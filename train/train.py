import os
import timm
import torch
import shutil
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from tqdm import tqdm
from torch import optim
from datetime import datetime
from torchvision import transforms
from pytorch_metric_learning import losses
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import AgglomerativeClustering, KMeans
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
    pairwise_distances_argmin_min,
    classification_report,
)

from sacred import Experiment
from sacred.observers import FileStorageObserver

from timm.data import create_transform
from config import DDI_PATCHES_DIR

ex = Experiment("skin_tone_prototype_classification")
ex.observers.append(
    FileStorageObserver(
        os.path.join("results/ours", datetime.now().strftime("%Y%m%d_%H%M%S"))
    )
)


@ex.config
def config():
    test_metadata_filename = "test_metadata.csv"

    csv_path = f"ddi_patches_metadata.csv"
    image_dir = DDI_PATCHES_DIR
    test_split_column = "original_image"
    image_column = "filename"

    # Models
    model_list = [
        "timm/mobilenetv3_large_100.ra_in1k",
        "timm/vit_small_patch16_dinov3.lvd1689m",
        "timm/tf_efficientnetv2_s.in21k_ft_in1k",
        "timm/resnet50.a1_in1k",
    ]

    embedding_dim = 128
    input_size = (224, 224)

    # Training
    lr = 1e-4
    epochs = 50
    n_folds = 5
    margin = 45
    batch_size = 24
    test_size = 0.2
    random_state = 42
    scheduler_patience = 5
    early_stop_patience = 10
    early_stop_metric = "val_loss"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers = 12
    loss = "arcface"
    weighted_sampler = True
    k_fold_group_by = "original_image"

    # Prototype Settings
    cluster_algorithm = "agglomerative"  # supported: 'kmeans', 'agglomerative'
    n_prototypes_per_class = 20


class PatchDataset(Dataset):
    def __init__(self, df, image_dir, transform=None, image_column="filename"):
        self.df = df
        self.image_dir = image_dir
        self.transform = transform
        self.image_paths = (
            self.df[image_column]
            .apply(lambda x: os.path.join(self.image_dir, str(x).strip()))
            .values
        )
        self.filenames = self.df[image_column].values
        self.labels = torch.tensor(self.df["target"].values, dtype=torch.long)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, self.labels[idx], self.filenames[idx]


class EmbeddingModel(nn.Module):
    def __init__(self, model_name, embedding_dim=128):
        super().__init__()
        self.backbone = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=0,
        )
        with torch.no_grad():
            input_sample = torch.randn(1, 3, 224, 224)
            features = self.backbone(input_sample)
            input_size = features.shape[1]

        self.projector = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
        )

    def forward(self, x):
        x = self.backbone(x)
        embeddings = self.projector(x)
        return F.normalize(embeddings, p=2, dim=1)


def get_transform(test=False, input_size=(224, 224)):
    if test:
        return transforms.Compose(
            [
                transforms.Resize(input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
    return transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(90),
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.RandomErasing(
                p=0.25, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=1
            ),
            transforms.RandomErasing(
                p=0.25, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0
            ),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def get_sampler(df, target_column="target"):
    class_counts = df[target_column].value_counts().sort_index()
    class_weights = 1.0 / class_counts

    # Map the class weight to every single sample in the dataframe
    sample_weights = df[target_column].map(class_weights).values

    # Create the sampler
    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=False,
    )
    return sampler


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
    weighted_sampler,
    test_size,
    random_state,
    batch_size,
    num_workers,
    image_column,
    cluster_algorithm,
    loss,
    k_fold_group_by,
    device,
    lr,
    scheduler_patience,
    embedding_dim,
    n_prototypes_per_class,
    margin,
    epochs,
    early_stop_metric,
    early_stop_patience,
    n_folds,
    input_size,
    label_to_int,
):

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
        print(
            f"Performing stratified group k-fold with grouping by '{k_fold_group_by}'"
        )
        kfold = StratifiedGroupKFold(
            n_splits=n_folds, shuffle=True, random_state=random_state
        )
    else:
        print("Performing stratified k-fold without grouping")
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

        # Train dataloader
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

        # Validation dataloader
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

        # Training configuration
        model = EmbeddingModel(
            model_name=model_name,
            embedding_dim=embedding_dim,
        ).to(device)

        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)

        criterion = losses.ArcFaceLoss(
            num_classes=2,
            embedding_size=embedding_dim,
            margin=margin,
            scale=64,
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

            # Learning rate scheduler
            scheduler.step(validation_loss)

            # Early stopping
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

            update_progress_bar(
                early_stop_patience,
                patience_counter,
                progress_bar,
                train_loss,
                validation_loss,
            )

        progress_bar.write("Selecting prototypes...")
        model.load_state_dict(torch.load(weights_filename))

        # Select prototypes from the trainig set, without augmentations
        prototype_dataloader = DataLoader(
            PatchDataset(
                fold_train_df,
                image_dir,
                transform=get_transform(test=True, input_size=input_size),
                image_column=image_column,
            ),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

        prototype_embeddings, prototype_labels, prototype_filenames = select_prototypes(
            model=model,
            dataloader=prototype_dataloader,
            method=cluster_algorithm,
            n_per_class=n_prototypes_per_class,
            device=device,
            random_state=random_state,
        )

        save_prototypes(image_dir, fold_dir, prototype_labels, prototype_filenames)

        # Validation using the k-nearest prototypes
        validation_metrics_dataloader = DataLoader(
            PatchDataset(
                fold_val_df,
                image_dir,
                transform=get_transform(True, input_size=input_size),
                image_column=image_column,
            ),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

        validation_kappa, validation_bacc, validation_mae, validation_f1 = (
            validate_with_prototypes(
                device=device,
                fold_dir=fold_dir,
                dataloader=validation_metrics_dataloader,
                model=model,
                prototype_embeddings=prototype_embeddings,
                prototype_labels=prototype_labels,
            )
        )

        progress_bar.write(
            f"Fold {fold+1} results: kappa={validation_kappa:.3f}, bacc={validation_bacc:.3f}, mae={validation_mae:.3f}, f1={validation_f1:.3f}"
        )

        # Save fold results
        fold_results.append(
            {
                "fold": fold + 1,
                "kappa": validation_kappa,
                "bacc": validation_bacc,
                "mae": validation_mae,
                "f1_score": validation_f1,
            }
        )
        pd.DataFrame(fold_results).to_csv(
            os.path.join(model_directory, "cross_validation_results.csv"), index=False
        )

    print(pd.DataFrame(fold_results))


def create_model_directory(results_dir, model_name):
    model_safe_name = (
        model_name.split("/")[1] if "/" in model_name else model_name
    )  # Remove timm/
    model_save_dir = os.path.join(results_dir, model_safe_name)
    os.makedirs(model_save_dir, exist_ok=True)
    return model_save_dir


def get_train_test(
    csv_path,
    test_split_column,
    test_metadata_filename,
    test_size,
    random_state,
    results_dir,
    label_to_int,
):
    df = pd.read_csv(csv_path)
    df["target"] = df["skin_tone"].map(label_to_int)
    df = df.dropna(subset=["target"])

    unique_ids = df[test_split_column].unique()
    if test_size == 0.0:
        test_ids = []
        train_ids = unique_ids
    else:
        train_ids, test_ids = train_test_split(
            unique_ids, test_size=test_size, random_state=random_state
        )

    train_df = df[df[test_split_column].isin(train_ids)]
    test_df = df[df[test_split_column].isin(test_ids)]
    test_df.to_csv(os.path.join(results_dir, test_metadata_filename), index=False)
    return train_df, test_df


def train_epoch(device, train_dataloader, model, optimizer, criterion, scaler):
    model.train()

    train_loss = 0.0
    for inputs, labels, _ in train_dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            embeddings = model(inputs)
            loss = criterion(embeddings, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item()

    return train_loss / len(train_dataloader)


def validate_epoch(
    device,
    validation_dataloader,
    model,
    criterion,
    bank_loader=None,
    val_metric_loader=None,
):
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        with torch.cuda.amp.autocast():
            for inputs, labels, _ in validation_dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                embeddings = model(inputs)
                loss = criterion(embeddings, labels)
                val_loss += loss.item()
    avg_val_loss = val_loss / len(validation_dataloader)
    return avg_val_loss


def update_training_history(
    fold_dir, optimizer, history, epoch, train_loss, validation_loss
):
    epoch_metrics = {
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "val_loss": validation_loss,
        "lr": optimizer.param_groups[0]["lr"],
    }

    history.append(epoch_metrics)
    pd.DataFrame(history).to_csv(os.path.join(fold_dir, "history.csv"), index=False)


def update_progress_bar(
    early_stop_patience, patience_counter, progress_bar, train_loss, validation_loss
):
    progress_bar.set_postfix(
        {
            "Tr_Loss": f"{train_loss:.4f}",
            "Val_Loss": f"{validation_loss:.4f}",
            "patience": f"{patience_counter}/{early_stop_patience}",
        }
    )


def select_prototypes(
    model, dataloader, n_per_class, device, method="agglomerative", random_state=42
):

    embeddings, labels, filenames = extract_embeddings_labels_filenames(
        model, dataloader, device
    )

    selected_prototypes = []
    selected_labels = []
    selected_filenames = []

    unique_labels = np.unique(labels)
    unique_labels.sort()

    for label in unique_labels:
        label_mask = labels == label
        label_embeddings = embeddings[label_mask]
        label_filenames = filenames[label_mask]

        if method == "kmeans":
            cluster_method = KMeans(
                n_clusters=n_per_class, random_state=random_state, n_init=10
            )
            cluster_method.fit(label_embeddings)
            centers = cluster_method.cluster_centers_

        elif method == "agglomerative":
            cluster_method = AgglomerativeClustering(
                n_clusters=n_per_class, linkage="ward"
            )
            cluster_labels = cluster_method.fit_predict(label_embeddings)

            centers = np.array(
                [
                    label_embeddings[cluster_labels == i].mean(axis=0)
                    for i in range(n_per_class)
                ]
            )
        else:
            raise NotImplementedError(f"The {method} is not supported")

        closest_embedding_indexes, _ = pairwise_distances_argmin_min(
            centers, label_embeddings
        )

        for index in closest_embedding_indexes:
            selected_prototypes.append(label_embeddings[index])
            selected_labels.append(label)
            selected_filenames.append(label_filenames[index])

    return (
        torch.tensor(np.array(selected_prototypes), device=device),
        torch.tensor(np.array(selected_labels), device=device),
        selected_filenames,
    )


def extract_embeddings_labels_filenames(model, dataloader, device):
    model.eval()
    all_embeddings, all_labels, all_filenames = [], [], []
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            for inputs, batch_labels, batch_filenames in dataloader:
                inputs = inputs.to(device)
                batch_embeddings = model(inputs)
                all_embeddings.append(batch_embeddings.cpu().numpy())
                all_labels.append(batch_labels.numpy())
                all_filenames.extend(batch_filenames)

    return (
        np.vstack(all_embeddings),
        np.concatenate(all_labels),
        np.array(all_filenames),
    )


def save_prototypes(image_dir, fold_dir, prototype_labels, prototype_filenames):
    csv_save_path = os.path.join(fold_dir, "selected_prototypes.csv")

    int_to_label = {0: "1234", 1: "56"}

    labels = prototype_labels.cpu().numpy()

    pd.DataFrame(
        {
            "filename": prototype_filenames,
            "target": labels,
            "skin_tone": [int_to_label[l] for l in labels],
        }
    ).to_csv(csv_save_path, index=False)

    save_prototype_images(
        prototype_filenames, labels, image_dir, fold_dir, int_to_label
    )


def save_prototype_images(filenames, labels, source_images_dir, fold_dir, int_to_label):
    save_dir = os.path.join(fold_dir, "prototype_images")

    for filename, label in zip(filenames, labels):
        class_folder = os.path.join(save_dir, str(int_to_label[label]))
        os.makedirs(class_folder, exist_ok=True)

        source_path = os.path.join(source_images_dir, filename)
        destination_path = os.path.join(class_folder, os.path.basename(filename))
        shutil.copy2(source_path, destination_path)


def validate_with_prototypes(
    device,
    fold_dir,
    dataloader,
    model,
    prototype_embeddings,
    prototype_labels,
    save_metrics=True,
):
    y_true, y_pred = classify_with_k_neighbors_prototypes(
        model, dataloader, prototype_embeddings, prototype_labels, device
    )

    final_kappa = cohen_kappa_score(y_true, y_pred, weights="quadratic")
    final_bacc = balanced_accuracy_score(y_true, y_pred)
    final_mae = mean_absolute_error(y_true, y_pred)
    final_f1_score = f1_score(y_true, y_pred, average="macro", zero_division=0)

    if save_metrics:
        metrics_df = pd.DataFrame(
            {
                "filename": dataloader.dataset.filenames,
                "y_true": y_true,
                "y_pred": y_pred,
            }
        )
        metrics_df.to_csv(
            os.path.join(fold_dir, "final_val_predictions.csv"), index=False
        )

        val_metrics_df = pd.DataFrame(
            {
                "metric": ["kappa", "balanced_accuracy", "mae", "f1_score"],
                "value": [final_kappa, final_bacc, final_mae, final_f1_score],
            }
        )
        val_metrics_df.to_csv(
            os.path.join(fold_dir, "final_val_metrics.csv"), index=False
        )
        print(
            classification_report(
                y_true,
                y_pred,
                target_names=[str(k) for k in range(len(prototype_labels.unique()))],
            )
        )
    return final_kappa, final_bacc, final_mae, final_f1_score


def classify_with_k_neighbors_prototypes(
    model, loader, prototypes, labels, device, k=5
):
    knn = KNeighborsClassifier(n_neighbors=k, metric="euclidean")

    if isinstance(prototypes, torch.Tensor):
        prototypes = prototypes.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    knn.fit(prototypes, labels)

    X_test, y_true, _ = extract_embeddings_labels_filenames(model, loader, device)

    y_pred = knn.predict(X_test)

    return y_true, y_pred


if __name__ == "__main__":
    ex.run()
