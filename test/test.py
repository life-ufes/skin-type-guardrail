import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from functools import partial
from scipy.stats import bootstrap
from torch.utils.data import DataLoader
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    balanced_accuracy_score, 
    classification_report, 
    cohen_kappa_score, 
    f1_score, 
    mean_absolute_error,
    confusion_matrix
)

from train.train import EmbeddingModel, PatchDataset, get_transform, extract_embeddings_labels_filenames, validate_with_prototypes

from sacred import Experiment
from sacred.observers import FileStorageObserver

from config import DDI_PATCHES_DIR

ex = Experiment('skin_tone_prototype_evaluation')
ex.observers.append(FileStorageObserver('results_test/ours'))

@ex.config
def config():

    results_dir = './results/ours/ddi_arcface_m28.6_dim128_p20_clusteragglomerative/1'
    image_dir = DDI_PATCHES_DIR
    image_column = 'filename'

    external_test_file = None
    
    models = {
        "tf_efficientnetv2_s": "timm/tf_efficientnetv2_s.in21k_ft_in1k",
        #"mobilenetv3_large_100": "timm/mobilenetv3_large_100.ra_in1k",
        #"resnet50_a1_in1k": "timm/resnet50.a1_in1k",
        #"vit_small_patch16_dinov3_lvd1689m": "timm/vit_small_patch16_dinov3.lvd1689m",
    }
    
    batch_size = 8
    num_workers = 8
    embedding_dim = 128
    input_size = (256, 256)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    n_neighbors = 5
    metric = 'euclidean'
    n_folders = 5

    n_bootstrap_resamples = 10000
    confidence_level = 0.95
    random_state = 42

def get_prototype_embeddings(csv_path, model, image_dir, image_column, device):
    df = pd.read_csv(csv_path)
    dataset = PatchDataset(df, image_dir=image_dir, transform=get_transform(test=True, input_size=(224, 224)), image_column=image_column)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
    return extract_embeddings_labels_filenames(model, dataloader, device)

def calculate_bootstrap_ci(y_true, y_pred, metric_function, n_resamples=10000, alpha=0.95, random_state=42):
    indexes = np.arange(len(y_true))
    
    def statistic_wrapper(idx):
        return metric_function(y_true[idx], y_pred[idx])

    result = bootstrap(
        (indexes, ),
        statistic_wrapper,
        n_resamples=n_resamples, 
        confidence_level=alpha,
        method='percentile',
        random_state=random_state
    )

    original_score = metric_function(y_true, y_pred)
    return original_score, result.confidence_interval.low, result.confidence_interval.high, result.bootstrap_distribution

@ex.capture
def test_ensemble_of_folder_models(model_name_timm, model_folder_name, _run, image_dir, results_dir, random_state,
                      batch_size, num_workers, embedding_dim, device, image_column, input_size,
                      n_neighbors, metric, n_folders, n_bootstrap_resamples, confidence_level):
    
    model_directory = os.path.join(results_dir, model_folder_name)
    
    df_test = pd.read_csv(os.path.join(results_dir, "test_metadata.csv"))
    
    test_dataset = PatchDataset(df_test, image_dir=image_dir, transform=get_transform(test=True, input_size=input_size), image_column=image_column)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    y_test = None
    ensemble_probs = None
    test_filenames = None
    evaluation_results = []

    for fold_num in range(1, n_folders + 1): 
        fold_directory = os.path.join(model_directory, f"fold_{fold_num}")
        weights_path = os.path.join(fold_directory, "best_weights.pth")
        proto_csv_path = os.path.join(fold_directory, "selected_prototypes.csv")

        if not os.path.exists(weights_path): continue

        model = EmbeddingModel(model_name_timm, embedding_dim).to(device)
        model.load_state_dict(torch.load(weights_path, map_location=device))
        model.eval()
        prototype_embeddings, prototype_labels, _ = get_prototype_embeddings(proto_csv_path, model, image_dir, image_column, device)

        validation_kappa, validation_bacc, validation_mae, validation_f1 = validate_with_prototypes(device=device,
                                                                                            fold_dir=fold_directory,
                                                                                            dataloader=test_dataloader,
                                                                                            model=model,
                                                                                            prototype_embeddings=prototype_embeddings,
                                                                                            prototype_labels=prototype_labels,
                                                                                            save_metrics=False)
        
        print(f"Fold {fold_num} validation - Kappa: {validation_kappa:.4f}, BACC: {validation_bacc:.4f}, MAE: {validation_mae:.4f}, F1 Macro: {validation_f1:.4f}")

        X_test, y_test_fold, filenames_fold = extract_embeddings_labels_filenames(model, test_dataloader, device)

        if y_test is None:
            y_test = y_test_fold
            test_filenames = filenames_fold

        knn = KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric)
        knn.fit(prototype_embeddings, prototype_labels)

        probs = knn.predict_proba(X_test)
        if ensemble_probs is None: ensemble_probs = probs
        else: ensemble_probs += probs
        
        fold_preds = np.argmax(probs, axis=1)
        fold_metrics = {
            'fold': fold_num,
            'test_kappa': cohen_kappa_score(y_test, fold_preds, weights='quadratic'),
            'test_bacc': balanced_accuracy_score(y_test, fold_preds),
        }
        evaluation_results.append(fold_metrics)
        print(f"Fold {fold_num}: Kappa={fold_metrics['test_kappa']:.3f}")

    if ensemble_probs is None: return

    average_probs = ensemble_probs / len(evaluation_results)
    ensemble_preds = np.argmax(average_probs, axis=1)

    metrics_functions = {
        'kappa': partial(cohen_kappa_score, weights='quadratic'),
        'bacc': balanced_accuracy_score,
        'mae': mean_absolute_error,
        'f1_macro': partial(f1_score, average='macro')
    }

    results = {}
    print('\nResults with CI:')
    for name, metric_function in metrics_functions.items():
        score, low, high, distribution = calculate_bootstrap_ci(
            y_test, ensemble_preds, metric_function, n_bootstrap_resamples, confidence_level, random_state
        )
        results[name] = {'score': score, 'low': low, 'high': high, 'dist': distribution}
        print(f"{name}: {score:.4f}  [{low:.4f} - {high:.4f}]")

    print(classification_report(y_test, ensemble_preds, target_names=['Light', 'Dark']))

    plot_confusion_matrix(confusion_matrix(y_test, ensemble_preds), model_directory, model_folder_name)
    save_results(model_folder_name, model_directory, test_filenames, y_test, evaluation_results, average_probs, ensemble_preds, results)

def save_results(model_folder_name, model_directory, test_filenames, y_test_true, evaluation_results, average_probs, ensemble_preds, results):
    pd.DataFrame(evaluation_results).to_csv(
        os.path.join(model_directory, f"{model_folder_name}_test_metrics_per_fold_model.csv"), index=False
    )

    pd.DataFrame({
        f"{name}_dist": data['dist'] for name, data in results.items()
    }).to_csv(os.path.join(model_directory, f"{model_folder_name}_bootstrap_distributions.csv"), index=False)

    csv_data = {
        'filename': test_filenames,
        'target': y_test_true,
        'pred': ensemble_preds,
    }
    
    csv_data['prob_light'] = average_probs[:, 0]
    csv_data['prob_dark'] = average_probs[:, 1]
    
    pd.DataFrame(csv_data).to_csv(os.path.join(model_directory, f"{model_folder_name}_ensemble_final.csv"), index=False)

    pd.DataFrame([
        {'metric': name, 'score': data['score'], 'ci_lower': data['low'], 'ci_upper': data['high']}
        for name, data in results.items()
    ]).to_csv(os.path.join(model_directory, f"{model_folder_name}_ensemble_confidence_intervals.csv"), index=False)

def plot_confusion_matrix(cm, model_directory, model_folder_name):
    class_names = ['I-IV', 'V-VI']

    plt.figure(figsize=(8, 6))

    cm_normalized = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis])
    labels = np.asarray([
        f"{percentage:.1%}\n({count})" 
        for count, percentage in zip(cm.flatten(), cm_normalized.flatten())
    ]).reshape(cm.shape)

    sns.set_theme(font_scale=1.8)
    
    sns.heatmap(cm, annot=labels, fmt='', cmap='Blues', 
                xticklabels=class_names, 
                yticklabels=class_names)
    
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix')
    plt.tight_layout()
    
    plt.savefig(os.path.join(model_directory, f"{model_folder_name}_confusion_matrix.png"), dpi=300)
    plt.close()

    print(f"Confusion matrix saved to {os.path.join(model_directory, f'{model_folder_name}_confusion_matrix.png')}")
    print(cm)


@ex.main
def main(models):
    for model_folder_name, model_name_timm in models.items():
        test_ensemble_of_folder_models(model_name_timm=model_name_timm, model_folder_name=model_name_timm.split('/')[-1])

if __name__ == "__main__":
    ex.run()