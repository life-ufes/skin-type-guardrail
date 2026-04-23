import os
import timm
import torch
import numpy as np
import pandas as pd
import torch.nn as nn

from tqdm import tqdm
from torch import optim
from datetime import datetime
from torch.utils.data import DataLoader
from sklearn.model_selection import (
    StratifiedGroupKFold,
    StratifiedKFold,
    train_test_split,
)

from sklearn.metrics import (
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    mean_absolute_error,
    classification_report,
)

from sacred import Experiment
from sacred.observers import FileStorageObserver

from train.train import (
    get_sampler,
    get_transform,
    get_train_test,
    create_model_directory,
    train_epoch,
    validate_epoch,
    update_training_history,
    PatchDataset,
)
from config import DDI_PATCHES_DIR

ex = Experiment("skin_tone_standard_classification")
ex.observers.append(
    FileStorageObserver(
        os.path.join(
            "results/bencevic", datetime.now().strftime("%Y%m%d_%H%M%S")
        )
    )
)

@ex.config
def config():
    matmul_precision = "medium"
    test_metadata_filename = "test_metadata.csv"

    csv_path = f"ddi_patches_metadata.csv"
    image_dir = DDI_PATCHES_DIR
    test_split_column = "original_image"
    image_column = "filename"

    # Model
    model_list = [
        "timm/vgg16.tv_in1k",
    ]

    linear_probe = False
    input_size = (224, 224)

    # Training
    lr = 1e-4
    epochs = 50
    n_folds = 5
    batch_size = 24
    test_size = 0.2
    random_state = 42
    scheduler_patience = 5
    early_stop_patience = 10
    early_stop_metric = "val_loss"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers = 12
    weighted_sampler = True
    k_fold_group_by = "original_image"


class ClassificationModel(nn.Module):
    def __init__(self, model_name, n_classes=2):
        super().__init__()

        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0)

        with torch.no_grad():
            input_sample = torch.randn(1, 3, 224, 224)
            features = self.backbone(input_sample)
            input_size = features.shape[1]

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, n_classes),
        )

    def forward(self, x):
        x = self.backbone(x)
        logits = self.classifier(x)
        return logits


@ex.main
def main(model_list, n_folds):
    label_to_int = {
        12: 0,
        34: 0,
        56: 1,
    }

    for model in model_list:
        run_cross_validation(
            model_name=model,
            input_size=(224, 224) if not "dino" in model else (256, 256),
            label_to_int=label_to_int,
        )


@ex.capture
def run_cross_validation(
    model_name,
    csv_path,
    image_dir,
    test_split_column,
    test_metadata_filename,
    matmul_precision,
    weighted_sampler,
    test_size,
    random_state,
    batch_size,
    num_workers,
    image_column,
    k_fold_group_by,
    device,
    lr,
    scheduler_patience,
    epochs,
    early_stop_metric,
    early_stop_patience,
    n_folds,
    input_size,
    label_to_int,
):

    torch.set_float32_matmul_precision(matmul_precision)
    weighted_sampler = True
    results_directory = ex.current_run.observers[0].dir
    model_directory = create_model_directory(results_directory, model_name)
    train_df, _ = get_train_test(
        csv_path,
        test_split_column,
        test_metadata_filename,
        test_size,
        random_state,
        results_directory,
        label_to_int=label_to_int,
    )

    if k_fold_group_by is not None and k_fold_group_by in train_df.columns:
        kfold = StratifiedGroupKFold(
            n_splits=n_folds, shuffle=True, random_state=random_state
        )
    else:
        kfold = StratifiedKFold(
            n_splits=n_folds, shuffle=True, random_state=random_state
        )

    fold_results = []

    for fold, (train_indexes, validation_indexes) in enumerate(
        kfold.split(
            train_df,
            train_df["target"],
            groups=(
                train_df[k_fold_group_by]
                if k_fold_group_by is not None and k_fold_group_by in train_df.columns
                else None
            ),
        )
    ):
        fold_dir = os.path.join(model_directory, f"fold_{fold+1}")
        os.makedirs(fold_dir, exist_ok=True)

        fold_train_df = train_df.iloc[train_indexes]
        fold_val_df = train_df.iloc[validation_indexes]

        print(fold_train_df["target"].value_counts())

        train_dataset = PatchDataset(
            fold_train_df,
            image_dir,
            transform=get_transform(test=False, input_size=input_size),
            image_column=image_column,
        )
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=not weighted_sampler,
            num_workers=num_workers,
            sampler=get_sampler(fold_train_df) if weighted_sampler else None,
        )

        validation_dataset = PatchDataset(
            fold_val_df,
            image_dir,
            transform=get_transform(test=True, input_size=input_size),
            image_column=image_column,
        )
        validation_dataloader = DataLoader(
            validation_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=max(1, num_workers // 3),
            pin_memory=True,
            persistent_workers=True,
        )

        model = ClassificationModel(model_name=model_name).to(device)

        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)

        class_counts = fold_train_df["target"].value_counts().sort_index().values
        weights = 1.0 / class_counts
        weights = weights / weights.sum()

        criterion = nn.CrossEntropyLoss(
            weight=torch.from_numpy(weights).float().to(device)
        )

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.1, patience=scheduler_patience
        )
        scaler = torch.cuda.amp.GradScaler()

        patience_counter = 0
        min_loss = float("inf")
        weights_filename = os.path.join(fold_dir, "best_weights.pth")

        progress_bar = tqdm(range(epochs), desc=f"Fold {fold+1} Training", unit="epoch")
        history = []
        for epoch in progress_bar:

            train_loss = train_epoch(
                device, train_dataloader, model, optimizer, criterion, scaler
            )
            validation_loss = validate_epoch(
                device, validation_dataloader, model, criterion
            )

            update_training_history(
                fold_dir, optimizer, history, epoch, train_loss, validation_loss
            )
            scheduler.step(validation_loss)

            improved = validation_loss < min_loss
            if improved:
                min_loss = validation_loss
                patience_counter = 0
                torch.save(model.state_dict(), weights_filename)
            else:
                patience_counter += 1
                if patience_counter >= early_stop_patience:
                    progress_bar.write(f"Early stopping triggered at epoch {epoch+1}")
                    break

            progress_bar.set_postfix(
                {
                    "Tr_Loss": f"{train_loss:.4f}",
                    "Val_Loss": f"{validation_loss:.4f}",
                    "patience": f"{patience_counter}/{early_stop_patience}",
                }
            )

        progress_bar.write("Evaluating best model on validation set...")
        model.load_state_dict(torch.load(weights_filename))

        val_kappa, val_bacc, val_mae, val_f1 = evaluate_model(
            device, fold_dir, validation_dataloader, model
        )

        progress_bar.write(
            f"Fold {fold+1} results: kappa={val_kappa:.3f}, bacc={val_bacc:.3f}, mae={val_mae:.3f}, f1={val_f1:.3f}"
        )

        fold_results.append(
            {
                "fold": fold + 1,
                "kappa": val_kappa,
                "bacc": val_bacc,
                "mae": val_mae,
                "f1_score": val_f1,
            }
        )
        pd.DataFrame(fold_results).to_csv(
            os.path.join(model_directory, "cross_validation_results.csv"), index=False
        )

    print(pd.DataFrame(fold_results))


def evaluate_model(device, fold_dir, dataloader, model, save_metrics=True):
    model.eval()
    all_preds, all_labels, all_filenames = [], [], []

    with torch.no_grad():
        with torch.cuda.amp.autocast():
            for inputs, labels, filenames in dataloader:
                inputs = inputs.to(device)
                logits = model(inputs)
                preds = torch.argmax(logits, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())
                all_filenames.extend(filenames)

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)

    final_kappa = cohen_kappa_score(y_true, y_pred, weights="quadratic")
    final_bacc = balanced_accuracy_score(y_true, y_pred)
    final_mae = mean_absolute_error(y_true, y_pred)
    final_f1_score = f1_score(y_true, y_pred, average="macro", zero_division=0)

    if save_metrics:
        pd.DataFrame(
            {"filename": all_filenames, "y_true": y_true, "y_pred": y_pred}
        ).to_csv(os.path.join(fold_dir, "final_val_predictions.csv"), index=False)

        pd.DataFrame(
            {
                "metric": ["kappa", "balanced_accuracy", "mae", "f1_score"],
                "value": [final_kappa, final_bacc, final_mae, final_f1_score],
            }
        ).to_csv(os.path.join(fold_dir, "final_val_metrics.csv"), index=False)

        print(
            classification_report(
                y_true, y_pred, target_names=['1234', '56'], zero_division=0
            )
        )

    return final_kappa, final_bacc, final_mae, final_f1_score

if __name__ == "__main__":
    ex.run()
