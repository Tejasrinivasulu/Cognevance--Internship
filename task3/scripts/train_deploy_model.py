"""
Fast trainer for Task-3 deployment model.

Trains a compact CNN on Fashion-MNIST (5 Task-2 classes) targeting ≥95% test accuracy.
Saves checkpoint + metrics compatible with the Flask API.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "dataset"
OUT_DIR = ROOT / "model"

CLASS_NAMES = ["Trouser", "Dress", "Sandal", "Sneaker", "Bag"]
# Fashion-MNIST indices for those labels
CLASS_INDICES = [1, 3, 5, 7, 8]


class FashionCNN(nn.Module):
    def __init__(self, num_classes: int = 5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.35),
            nn.Linear(128 * 3 * 3, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def filter_by_classes(dataset, class_indices: list[int]) -> Subset:
    targets = dataset.targets
    indices = [i for i, y in enumerate(targets) if int(y) in class_indices]
    remap = {old: new for new, old in enumerate(class_indices)}

    class Remapped(Subset):
        def __getitem__(self, idx):
            img, label = super().__getitem__(idx)
            return img, remap[int(label)]

        def __getitems__(self, indices):  # DataLoader batch path
            return [self.__getitem__(i) for i in indices]

    return Remapped(dataset, indices)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        x = x.to(device)
        pred = model(x).argmax(1).cpu().numpy().tolist()
        ys.extend(y.numpy().tolist())
        ps.extend(pred)
    y_true = np.array(ys)
    y_pred = np.array(ps)
    return accuracy_score(y_true, y_pred), y_true, y_pred


def main() -> None:
    set_seed(42)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_tf = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(8),
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,)),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,)),
        ]
    )

    # Prefer local Task-2 data cache if present
    task2_data = Path(r"C:\Users\Teja Srinivasulu\OneDrive\Desktop\Internspark-task2\data")
    data_root = task2_data if task2_data.exists() else DATA_DIR

    train_full = datasets.FashionMNIST(root=data_root, train=True, download=True, transform=train_tf)
    test_full = datasets.FashionMNIST(root=data_root, train=False, download=True, transform=eval_tf)

    train_ds = filter_by_classes(train_full, CLASS_INDICES)
    test_ds = filter_by_classes(test_full, CLASS_INDICES)

    n = len(train_ds)
    idx = list(range(n))
    random.shuffle(idx)
    val_n = max(1, int(0.1 * n))
    train_subset = Subset(train_ds, idx[val_n:])
    val_subset = Subset(train_ds, idx[:val_n])

    train_loader = DataLoader(train_subset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=256, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

    model = FashionCNN(num_classes=len(CLASS_NAMES)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.5)

    epochs = 12
    best_val = 0.0
    best_path = OUT_DIR / "best_model.pt"
    history = {"train_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += x.size(0)

        train_loss = running_loss / total
        train_acc = correct / total
        val_acc, _, _ = evaluate(model, val_loader, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        print(
            f"Epoch {epoch}/{epochs}  train_loss={train_loss:.4f}  "
            f"train_acc={train_acc:.4f}  val_acc={val_acc:.4f}",
            flush=True,
        )

        if val_acc >= best_val:
            best_val = val_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_names": CLASS_NAMES,
                    "num_classes": len(CLASS_NAMES),
                    "dataset": "fashionmnist",
                    "architecture": "FashionCNN",
                    "epoch": epoch,
                    "val_acc": val_acc,
                    "freeze_backbone": False,
                    "input": {"channels": 1, "size": 28, "normalize_mean": [0.2860], "normalize_std": [0.3530]},
                },
                best_path,
            )
            print(f"  Saved best model (val_acc={val_acc:.4f})")

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_acc, y_true, y_pred = evaluate(model, test_loader, device)
    f1 = f1_score(y_true, y_pred, average="macro")
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES)
    print("\n===== Test Results =====")
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Macro F1: {f1:.4f}")
    print(report)

    if test_acc < 0.95:
        raise SystemExit(f"Accuracy {test_acc:.4f} is below 0.95 — retrain with more epochs.")

    metrics = {
        "dataset": "fashionmnist",
        "architecture": "FashionCNN",
        "test_accuracy": float(test_acc),
        "macro_f1": float(f1),
        "best_val_accuracy": float(best_val),
        "class_names": CLASS_NAMES,
        "epochs": epochs,
        "history": history,
        "classification_report": report,
    }
    (OUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Saved metrics -> {OUT_DIR / 'metrics.json'}")
    print(f"Saved model    -> {best_path}")


if __name__ == "__main__":
    main()
