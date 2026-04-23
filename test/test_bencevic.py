import os
import torch
import numpy as np
import pandas as pd

from functools import partial
from scipy.stats import bootstrap
from torch.utils.data import DataLoader
from sklearn.metrics import (
    balanced_accuracy_score, 
    classification_report, 
    cohen_kappa_score, 
    f1_score, 
    mean_absolute_error,
    confusion_matrix,
    roc_curve,
    auc
)

from train.train_bencevic import ClassificationModel, PatchDataset, get_transform
from test.test import calculate_bootstrap_ci, save_results, plot_confusion_matrix

from sacred import Experiment
from sacred.observers import FileStorageObserver
from config import DDI_PATCHES_DIR

ex = Experiment('skin_tone_classification_evaluation')
ex.observers.append(FileStorageObserver('results_test/bencevic'))

@ex.config
def config():

    results_dir = './results/bencevic/20260423_140431/1'
    image_dir = DDI_PATCHES_DIR
    image_column = 'filename'

    external_test_file = None
    
    models = {
        'vgg16_tv_in1k': 'timm/vgg16.tv_in1k',
    }
    
    batch_size = 8
    num_workers = 8
    linear_probe = False
    input_size = (256, 256)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    n_folders = 5

    n_bootstrap_resamples = 10000 
    confidence_level = 0.95
    random_state = 42

def extract_probabilities_labels_filenames(model, dataloader, device):
    model.eval()
    all_probs, all_labels, all_filenames = [], [], []
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            for inputs, labels, filenames in dataloader:
                inputs = inputs.to(device)
                logits = model(inputs)
                probs = torch.softmax(logits, dim=1)
                
                all_probs.append(probs.cpu().numpy())
                all_labels.append(labels.numpy())
                all_filenames.extend(filenames)

    return np.vstack(all_probs), np.concatenate(all_labels), np.array(all_filenames)

@ex.capture
def test_ensemble_of_folder_models(model_name_timm, model_folder_name, _run, image_dir, results_dir, random_state,
                      batch_size, num_workers, device, image_column, input_size, external_test_file,
                      n_folders, n_bootstrap_resamples, confidence_level, linear_probe):
    
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

        if not os.path.exists(weights_path): 
            print(f"Skipping fold {fold_num}, weights not found at {weights_path}")
            continue

        model = ClassificationModel(model_name_timm, n_classes=2).to(device)
        model.load_state_dict(torch.load(weights_path, map_location=device))
        
        print(f"Evaluating Fold {fold_num}...")

        probs, y_test_fold, filenames_fold = extract_probabilities_labels_filenames(model, test_dataloader, device)
        
        if y_test is None:
            y_test = y_test_fold
            test_filenames = filenames_fold

        if ensemble_probs is None: 
            ensemble_probs = probs
        else: 
            ensemble_probs += probs
        
        # Calculate Image-Level Fold Metrics
        fold_preds = np.argmax(probs, axis=1)
        fold_metrics = {
            'fold': fold_num,
            'test_kappa': cohen_kappa_score(y_test_fold, fold_preds, weights='quadratic'),
            'test_bacc': balanced_accuracy_score(y_test_fold, fold_preds),
        }
        evaluation_results.append(fold_metrics)
        print(f"Fold {fold_num} Metrics: Kappa={fold_metrics['test_kappa']:.4f}, BACC={fold_metrics['test_bacc']:.4f}")

    if ensemble_probs is None: return

    # Final Ensemble Averaging
    average_probs = ensemble_probs / len(evaluation_results)
    ensemble_preds = np.argmax(average_probs, axis=1)

    metrics_functions = {
        'kappa': partial(cohen_kappa_score, weights='quadratic'),
        'bacc': balanced_accuracy_score,
        'mae': mean_absolute_error,
        'f1_macro': partial(f1_score, average='macro')
    }
    
    results = {}
    print('\nFinal Aggregated Image-Level Results with CI:')
    for name, metric_function in metrics_functions.items():
        score, low, high, distribution = calculate_bootstrap_ci(
            y_test, ensemble_preds, metric_function, n_bootstrap_resamples, confidence_level, random_state
        )
        results[name] = {'score': score, 'low': low, 'high': high, 'dist': distribution}
        print(f"{name}: {score:.4f}  [{low:.4f} - {high:.4f}]")
        
    target_names = ['Light', 'Dark']
    print("\nClassification Report:")
    print(classification_report(y_test, ensemble_preds, target_names=target_names))

    plot_confusion_matrix(confusion_matrix(y_test, ensemble_preds), model_directory, model_folder_name)
    save_results(model_folder_name, model_directory, test_filenames, y_test, evaluation_results, average_probs, ensemble_preds, results)

@ex.main
def main(models):
    for model_folder_name, model_name_timm in models.items():
        test_ensemble_of_folder_models(model_name_timm=model_name_timm, model_folder_name=model_name_timm.split('/')[-1])

if __name__ == "__main__":
    ex.run()