"""
Image Classifier with Transfer Learning (ResNet-18).

Supports Fashion-MNIST (default) or CIFAR-10. Uses a pretrained ImageNet
backbone, data augmentation, training curves, and evaluation metrics.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms
from tqdm import tqdm

CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

FASHION_CLASSES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_transforms(grayscale: bool = False) -> tuple[transforms.Compose, transforms.Compose]:
    """ImageNet-normalized transforms; train set uses data augmentation."""
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    to_rgb = [transforms.Grayscale(num_output_channels=3)] if grayscale else []

    train_tf = transforms.Compose(
        to_rgb
        + [
            transforms.Resize(224),
            transforms.RandomCrop(224, padding=16),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    eval_tf = transforms.Compose(
        to_rgb
        + [
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    return train_tf, eval_tf


def filter_by_classes(dataset, class_indices: list[int]) -> Subset:
    """Keep only samples whose label is in class_indices; remap labels 0..K-1."""
    targets = dataset.targets if hasattr(dataset, "targets") else dataset.labels
    indices = [i for i, y in enumerate(targets) if int(y) in class_indices]
    remap = {old: new for new, old in enumerate(class_indices)}

    class RemappedSubset(Subset):
        def __getitem__(self, idx):
            img, label = super().__getitem__(idx)
            return img, remap[int(label)]

        def __getitems__(self, indices):
            return [self.__getitem__(idx) for idx in indices]

    return RemappedSubset(dataset, indices)


def build_model(num_classes: int, freeze_backbone: bool = True) -> nn.Module:
    """ResNet-18 pretrained on ImageNet with a new classifier head."""
    weights = models.ResNet18_Weights.IMAGENET1K_V1
    model = models.resnet18(weights=weights)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, num_classes),
    )
    return model


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="Train", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()
    running_loss = 0.0
    all_preds: list[int] = []
    all_labels: list[int] = []

    for images, labels in tqdm(loader, desc="Eval", leave=False):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    acc = accuracy_score(y_true, y_pred)
    return running_loss / len(y_true), acc, y_true, y_pred


def plot_curves(history: dict, out_path: Path) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, history["train_loss"], label="Train", marker="o")
    axes[0].plot(epochs, history["val_loss"], label="Val", marker="s")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history["train_acc"], label="Train", marker="o")
    axes[1].plot(epochs, history["val_acc"], label="Val", marker="s")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved training curves -> {out_path}")


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    out_path: Path,
) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion Matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0 if cm.max() else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved confusion matrix -> {out_path}")


def resolve_classes(dataset_name: str, classes_arg: str) -> tuple[list[str], list[int]]:
    all_classes = FASHION_CLASSES if dataset_name == "fashionmnist" else CIFAR10_CLASSES
    if classes_arg.strip().lower() == "all":
        return all_classes, list(range(len(all_classes)))

    class_names = [c.strip() for c in classes_arg.split(",") if c.strip()]
    unknown = [c for c in class_names if c not in all_classes]
    if unknown:
        raise ValueError(f"Unknown classes: {unknown}. Choose from {all_classes}")
    class_indices = [all_classes.index(c) for c in class_names]
    return class_names, class_indices


def load_datasets(dataset_name: str, data_dir: Path, train_tf, eval_tf):
    if dataset_name == "fashionmnist":
        train_full = datasets.FashionMNIST(
            root=data_dir, train=True, download=True, transform=train_tf
        )
        test_full = datasets.FashionMNIST(
            root=data_dir, train=False, download=True, transform=eval_tf
        )
    else:
        train_full = datasets.CIFAR10(
            root=data_dir, train=True, download=True, transform=train_tf
        )
        test_full = datasets.CIFAR10(
            root=data_dir, train=False, download=True, transform=eval_tf
        )
    return train_full, test_full


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ResNet-18 image classifier")
    parser.add_argument(
        "--dataset",
        type=str,
        default="fashionmnist",
        choices=["fashionmnist", "cifar10"],
        help="Dataset to train on (default: fashionmnist)",
    )
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--output-dir", type=str, default="./outputs")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--classes",
        type=str,
        default="Trouser,Dress,Sandal,Sneaker,Bag",
        help="Comma-separated class names (subset). Use 'all' for all classes.",
    )
    parser.add_argument(
        "--unfreeze",
        action="store_true",
        help="Fine-tune the full backbone (default: freeze backbone, train head only).",
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=0,
        help="Optional cap on training samples (0 = use all).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    class_names, class_indices = resolve_classes(args.dataset, args.classes)
    num_classes = len(class_names)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Dataset: {args.dataset}")
    print(f"Classes ({num_classes}): {class_names}")

    grayscale = args.dataset == "fashionmnist"
    train_tf, eval_tf = get_transforms(grayscale=grayscale)
    train_full, test_full = load_datasets(args.dataset, data_dir, train_tf, eval_tf)

    train_ds = filter_by_classes(train_full, class_indices)
    test_ds = filter_by_classes(test_full, class_indices)

    n = len(train_ds)
    indices = list(range(n))
    random.shuffle(indices)
    if args.max_train_samples > 0:
        indices = indices[: args.max_train_samples]
        n = len(indices)
    val_size = max(1, int(0.1 * n))
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    train_subset = Subset(train_ds, train_indices)
    val_subset = Subset(train_ds, val_indices)

    train_loader = DataLoader(
        train_subset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    print(f"Train: {len(train_subset)} | Val: {len(val_subset)} | Test: {len(test_ds)}")

    model = build_model(num_classes, freeze_backbone=not args.unfreeze).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0.0
    best_path = output_dir / "best_model.pt"

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(
            f"  train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_names": class_names,
                    "num_classes": num_classes,
                    "dataset": args.dataset,
                    "epoch": epoch,
                    "val_acc": val_acc,
                    "freeze_backbone": not args.unfreeze,
                },
                best_path,
            )
            print(f"  Saved best model (val_acc={val_acc:.4f})")

    plot_curves(history, output_dir / "training_curves.png")

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss, test_acc, y_true, y_pred = evaluate(
        model, test_loader, criterion, device
    )
    f1 = f1_score(y_true, y_pred, average="macro")
    report = classification_report(y_true, y_pred, target_names=class_names)

    print("\n===== Test Results =====")
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Macro F1: {f1:.4f}")
    print(report)

    plot_confusion_matrix(
        y_true, y_pred, class_names, output_dir / "confusion_matrix.png"
    )

    metrics = {
        "dataset": args.dataset,
        "test_loss": float(test_loss),
        "test_accuracy": float(test_acc),
        "macro_f1": float(f1),
        "best_val_accuracy": float(best_val_acc),
        "class_names": class_names,
        "epochs": args.epochs,
        "history": history,
        "classification_report": report,
    }
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics -> {metrics_path}")
    print(f"Saved model    -> {best_path}")


if __name__ == "__main__":
    main()
