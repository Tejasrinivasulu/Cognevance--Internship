"""
Run inference with a trained CIFAR-10 ResNet-18 classifier.

Examples:
  python inference.py --image path/to/photo.jpg
  python inference.py --image path/to/photo.jpg --model outputs/best_model.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


def get_eval_transform() -> transforms.Compose:
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    return transforms.Compose(
        [
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    # RGB conversion is handled when opening the image (.convert("RGB"))


def build_model(num_classes: int, freeze_backbone: bool = True) -> nn.Module:
    model = models.resnet18(weights=None)
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, num_classes),
    )
    return model


def load_checkpoint(model_path: Path, device: torch.device):
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    class_names = checkpoint["class_names"]
    num_classes = checkpoint["num_classes"]
    freeze = checkpoint.get("freeze_backbone", True)
    model = build_model(num_classes, freeze_backbone=freeze).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, class_names


@torch.no_grad()
def predict(image_path: Path, model_path: Path, top_k: int = 3) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, class_names = load_checkpoint(model_path, device)
    transform = get_eval_transform()

    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)
    logits = model(tensor)
    probs = torch.softmax(logits, dim=1)[0]
    top_k = min(top_k, len(class_names))
    values, indices = torch.topk(probs, top_k)

    print(f"Image: {image_path}")
    print(f"Model: {model_path}")
    print("Predictions:")
    for rank, (score, idx) in enumerate(zip(values.tolist(), indices.tolist()), start=1):
        print(f"  {rank}. {class_names[idx]:12s}  {score * 100:6.2f}%")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CIFAR-10 classifier inference")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument(
        "--model",
        type=str,
        default="outputs/best_model.pt",
        help="Path to trained checkpoint",
    )
    parser.add_argument("--top-k", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_path = Path(args.image)
    model_path = Path(args.model)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}. Train first with: python train.py"
        )
    predict(image_path, model_path, top_k=args.top_k)


if __name__ == "__main__":
    main()
