"""Trening, walidacja i test klasyfikatora ClothingCNN."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import ClothingCNN

# torchvision uses (height, width); CLI values are provided as width, height.
DEFAULT_IMAGE_SIZE = (256, 192)
DEFAULT_CLASSES = ("Backpacks", "Cups", "Jeans", "Shirts", "Watches")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_datasets(
    dataset_dir: Path, image_size: tuple[int, int]
) -> tuple[Any, Any, Any]:
    train_transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    evaluation_transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    train_set = datasets.ImageFolder(dataset_dir / "train", train_transform)
    val_set = datasets.ImageFolder(dataset_dir / "val", evaluation_transform)
    test_set = datasets.ImageFolder(dataset_dir / "test", evaluation_transform)
    if train_set.classes != val_set.classes or train_set.classes != test_set.classes:
        raise ValueError("Foldery train, val i test muszą zawierać te same klasy.")
    if not train_set or not val_set or not test_set:
        raise ValueError("Każdy z folderów train, val i test musi zawierać zdjęcia.")
    return train_set, val_set, test_set


def make_loader(dataset: Any, batch_size: int, workers: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
    )


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    correct = 0
    total = 0
    context = torch.enable_grad() if training else torch.inference_mode()

    with context:
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
    return total_loss / total, correct / total


def save_checkpoint(
    path: Path,
    model: nn.Module,
    classes: list[str],
    image_size: tuple[int, int],
    epoch: int,
    val_accuracy: float,
) -> None:
    torch.save(
        {
            "model_state": model.state_dict(),
            "classes": classes,
            "image_size": image_size,
            "epoch": epoch,
            "val_accuracy": val_accuracy,
        },
        path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trenuje ClothingCNN od zera.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("runs/clothing_cnn"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--image-size",
        nargs=2,
        type=int,
        metavar=("WIDTH", "HEIGHT"),
        default=(DEFAULT_IMAGE_SIZE[1], DEFAULT_IMAGE_SIZE[0]),
        help=(
            "Rozmiar wejściowy obrazu jako szerokość wysokość "
            "(domyślnie: 192 256)."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.workers < 0:
        raise ValueError("epochs i batch-size muszą być dodatnie, workers nieujemne.")
    if any(size < 1 for size in args.image_size):
        raise ValueError("Szerokość i wysokość obrazu muszą być dodatnie.")
    image_size = (args.image_size[1], args.image_size[0])
    set_seed(args.seed)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
        if args.device != "auto"
        else "cpu"
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Wybrano CUDA, ale GPU nie jest dostępne.")

    train_set, val_set, test_set = create_datasets(args.dataset, image_size)
    train_loader = make_loader(train_set, args.batch_size, args.workers, True)
    val_loader = make_loader(val_set, args.batch_size, args.workers, False)
    test_loader = make_loader(test_set, args.batch_size, args.workers, False)
    model = ClothingCNN(len(train_set.classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )
    args.output.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, float]] = []
    best_accuracy = -1.0
    epochs_without_improvement = 0

    print(f"Urządzenie: {device}; klasy: {', '.join(train_set.classes)}")
    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = run_epoch(
            model, train_loader, criterion, device, optimizer
        )
        val_loss, val_accuracy = run_epoch(model, val_loader, criterion, device)
        scheduler.step(val_accuracy)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
            }
        )
        print(
            f"Epoch {epoch:03d}/{args.epochs}: "
            f"train_loss={train_loss:.4f} train_acc={train_accuracy:.3f} "
            f"val_loss={val_loss:.4f} val_acc={val_accuracy:.3f}",
            flush=True,
        )
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            epochs_without_improvement = 0
            save_checkpoint(
                args.output / "best_model.pt",
                model,
                train_set.classes,
                image_size,
                epoch,
                val_accuracy,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print("Early stopping.", flush=True)
                break

    checkpoint = torch.load(args.output / "best_model.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    test_loss, test_accuracy = run_epoch(model, test_loader, criterion, device)
    results = {
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "best_val_accuracy": checkpoint["val_accuracy"],
        "classes": train_set.classes,
    }
    (args.output / "history.json").write_text(
        json.dumps({"epochs": history, "results": results}, indent=2),
        encoding="utf-8",
    )
    print(f"Test: loss={test_loss:.4f}, accuracy={test_accuracy:.3f}")
    print(f"Zapisano model i historię w: {args.output.resolve()}")


if __name__ == "__main__":
    main()
