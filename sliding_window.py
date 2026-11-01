"""Prosta detekcja wieloobiektowa przez sliding window."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from torchvision import transforms

from model import ClothingCNN


def intersection_over_union(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
    left = max(box_a[0], box_b[0])
    top = max(box_a[1], box_b[1])
    right = min(box_a[2], box_b[2])
    bottom = min(box_a[3], box_b[3])
    intersection = max(0, right - left) * max(0, bottom - top)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def suppress_overlaps(
    detections: list[tuple[tuple[int, int, int, int], str, float]], threshold: float
) -> list[tuple[tuple[int, int, int, int], str, float]]:
    kept: list[tuple[tuple[int, int, int, int], str, float]] = []
    for detection in sorted(detections, key=lambda item: item[2], reverse=True):
        if all(
            detection[1] != previous[1]
            or intersection_over_union(detection[0], previous[0]) < threshold
            for previous in kept
        ):
            kept.append(detection)
    return kept


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Wykrywa ubrania metodą sliding window.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("detection_result.jpg"))
    parser.add_argument("--window-width", type=int, default=256)
    parser.add_argument("--window-height", type=int, default=256)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--threshold", type=float, default=0.85)
    parser.add_argument("--nms-iou", type=float, default=0.3)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def window_positions(length: int, window: int, stride: int) -> list[int]:
    if length <= window:
        return [0]
    positions = list(range(0, length - window + 1, stride))
    last_position = length - window
    if positions[-1] != last_position:
        positions.append(last_position)
    return positions


def main() -> None:
    args = parse_args()
    if args.window_width < 1 or args.window_height < 1 or args.stride < 1:
        raise ValueError("Rozmiary okna i stride muszą być dodatnie.")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Wybrano CUDA, ale GPU nie jest dostępne.")
    device = torch.device(
        "cuda"
        if args.device == "cuda"
        or (args.device == "auto" and torch.cuda.is_available())
        else "cpu"
    )
    checkpoint = torch.load(args.model, map_location=device)
    classes: list[str] = checkpoint["classes"]
    model = ClothingCNN(len(classes)).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    image = Image.open(args.image).convert("RGB")
    transform = transforms.Compose(
        [
            transforms.Resize(tuple(checkpoint.get("image_size", (80, 60)))),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    detections: list[tuple[tuple[int, int, int, int], str, float]] = []
    windows: list[tuple[tuple[int, int, int, int], Image.Image]] = []
    for top in window_positions(image.height, args.window_height, args.stride):
        for left in window_positions(image.width, args.window_width, args.stride):
            right = min(left + args.window_width, image.width)
            bottom = min(top + args.window_height, image.height)
            box = (left, top, right, bottom)
            windows.append((box, image.crop(box)))
    with torch.inference_mode():
        for start in range(0, len(windows), 64):
            batch = torch.stack([transform(crop) for _, crop in windows[start : start + 64]])
            probabilities = model(batch.to(device)).softmax(dim=1)
            values, indices = probabilities.max(dim=1)
            for (box, _), value, index in zip(
                windows[start : start + 64], values.cpu(), indices.cpu()
            ):
                if value.item() >= args.threshold:
                    detections.append((box, classes[index.item()], value.item()))
    detections = suppress_overlaps(detections, args.nms_iou)
    draw = ImageDraw.Draw(image)
    for (left, top, right, bottom), label, confidence in detections:
        draw.rectangle((left, top, right, bottom), outline="red", width=3)
        draw.text((left + 4, top + 4), f"{label} {confidence:.2f}", fill="red")
    image.save(args.output)
    print(f"Wykrycia: {len(detections)}; zapisano: {args.output.resolve()}")


if __name__ == "__main__":
    main()
