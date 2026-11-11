"""Podział zdjęć na zbiory treningowy, walidacyjny i testowy."""

from __future__ import annotations

import argparse
import csv
import random
import shutil
from collections import defaultdict
from pathlib import Path

from PIL import Image

LABELS = ("Shirts", "Jeans", "Caps", "Backpacks", "Watches")
IMAGE_EXTENSIONS = {".jpg"}
SPLITS = ("train", "val", "test")
LABELS_BY_KEY = {label.casefold(): label for label in LABELS}


def read_records(csv_path: Path, images_dir: Path) -> dict[str, list[Path]]:
    """Wczytuje rekordy CSV i zwraca ścieżki zdjęć pogrupowane po etykietach."""
    records: dict[str, list[Path]] = defaultdict(list)
    label_counts = {label: 0 for label in LABELS}
    seen_images: set[Path] = set()
    image_index = build_image_index(images_dir)
    missing_images: list[tuple[int, str]] = []

    with csv_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.reader(csv_file)
        for row_number, row in enumerate(reader, start=1):
            if not row or all(not cell.strip() for cell in row):
                continue
            if row_number == 1 and len(row) >= 5:
                first, fifth = row[0].strip().lower(), row[4].strip().lower()
                if first == "id" and fifth == "articletype":
                    continue
            if len(row) < 5:
                raise ValueError(
                    f"Wiersz {row_number} w pliku CSV ma mniej niż 5 kolumn."
                )

            image_id = row[0].strip()
            label = LABELS_BY_KEY.get("".join(row[4].split()).casefold())
            if label is None:
                continue
            label_counts[label] += 1
            if not image_id:
                raise ValueError(f"Brak nazwy zdjęcia w wierszu {row_number}.")

            image_path = find_image(image_index, image_id)
            if image_path is None:
                missing_images.append((row_number, image_id))
                continue
            resolved_path = image_path.resolve()
            if resolved_path in seen_images:
                raise ValueError(
                    f"Zdjęcie występuje więcej niż raz w danych: {image_path.name}"
                )
            seen_images.add(resolved_path)
            records[label].append(image_path)

    if missing_images:
        preview = ", ".join(
            f"'{image_id}' (wiersz {row_number})"
            for row_number, image_id in missing_images[:10]
        )
        suffix = (
            f" i {len(missing_images) - 10} kolejnych"
            if len(missing_images) > 10
            else ""
        )
        print(
            f"Pominięto {len(missing_images)} brakujących zdjęć: {preview}{suffix}",
            flush=True,
        )

    if not records:
        raise ValueError("Nie znaleziono zdjęć dla wybranych etykiet.")
    print(
        "Rekordy wybranych klas w CSV: "
        + ", ".join(f"{label}={label_counts[label]}" for label in LABELS),
        flush=True,
    )
    return records


def build_image_index(images_dir: Path) -> dict[str, list[Path]]:
    """Buduje indeks nazw zdjęć, aby nie skanować folderu dla każdego wiersza CSV."""
    index: dict[str, list[Path]] = defaultdict(list)
    for path in images_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            index[path.name.casefold()].append(path)
            index[path.stem.casefold()].append(path)
    return index


def find_image(
    image_index: dict[str, list[Path]], image_id: str
) -> Path | None:
    """Znajduje zdjęcie po pełnej nazwie lub po nazwie bez rozszerzenia."""
    normalized_id = Path(image_id).name.casefold()
    candidates = image_index.get(normalized_id, [])
    if not Path(image_id).suffix:
        candidates = image_index.get(Path(image_id).stem.casefold(), [])
    candidates = sorted(set(candidates))
    if not candidates:
        return None
    if len(candidates) > 1:
        names = ", ".join(path.name for path in candidates)
        raise ValueError(f"Niejednoznaczna nazwa '{image_id}': {names}")
    return candidates[0]


def split_records(
    records: dict[str, list[Path]], seed: int
) -> dict[str, dict[str, list[Path]]]:
    """Dzieli zdjęcia osobno w każdej klasie, zachowując stratyfikację."""
    rng = random.Random(seed)
    result: dict[str, dict[str, list[Path]]] = {}

    for label, paths in records.items():
        shuffled = list(paths)
        rng.shuffle(shuffled)
        train_end = int(len(shuffled) * 0.8)
        val_end = train_end + int(len(shuffled) * 0.1)
        result[label] = {
            "train": shuffled[:train_end],
            "val": shuffled[train_end:val_end],
            "test": shuffled[val_end:],
        }
    return result


def copy_splits(
    split_records_by_label: dict[str, dict[str, list[Path]]],
    output_dir: Path,
    resize_max_size: int | None = None,
) -> dict[str, int]:
    """Tworzy strukturę katalogów i kopiuje lub skaluje zdjęcia."""
    counts = {split: 0 for split in SPLITS}
    for label in LABELS:
        split_paths = split_records_by_label.get(
            label, {split: [] for split in SPLITS}
        )
        for split, paths in split_paths.items():
            target_dir = output_dir / split / label
            target_dir.mkdir(parents=True, exist_ok=True)
            for source_path in paths:
                target_path = target_dir / source_path.name
                if resize_max_size is None:
                    shutil.copy2(source_path, target_path)
                else:
                    with Image.open(source_path) as image:
                        image.thumbnail(
                            (resize_max_size, resize_max_size),
                            Image.Resampling.LANCZOS,
                        )
                        image.save(target_path, quality=95, optimize=True)
                counts[split] += 1
    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dzieli zdjęcia na train/val/test dla pięciu klas ubrań."
    )
    parser.add_argument("--images", type=Path, required=True, help="Folder ze zdjęciami.")
    parser.add_argument("--csv", dest="csv_path", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dataset"),
        help="Folder wynikowy (domyślnie: dataset).",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Ziarno losowania (domyślnie: 42)."
    )
    parser.add_argument(
        "--resize",
        type=int,
        metavar="PIXELS",
        help=(
            "Maksymalny rozmiar dłuższego boku zdjęcia w pikselach; "
            "zachowuje proporcje i nie powiększa zdjęć."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.images.is_dir():
        raise NotADirectoryError(f"Folder zdjęć nie istnieje: {args.images}")
    if not args.csv_path.is_file():
        raise FileNotFoundError(f"Plik CSV nie istnieje: {args.csv_path}")
    if args.resize is not None and args.resize <= 0:
        raise ValueError("Wartość --resize musi być większa od zera.")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(
            f"Folder wynikowy nie jest pusty: {args.output}. "
            "Wybierz pusty folder, aby nie mieszać zbiorów."
        )
    args.output.mkdir(parents=True, exist_ok=True)

    print("Indeksowanie zdjęć i odczyt CSV...", flush=True)
    records = read_records(args.csv_path, args.images)
    split_records_by_label = split_records(records, args.seed)
    counts = copy_splits(split_records_by_label, args.output, args.resize)

    print(f"Gotowe. Zapisano dane w: {args.output.resolve()}")
    for split in SPLITS:
        print(f"  {split}: {counts[split]} zdjęć")
    print("Klasy: " + ", ".join(sorted(records)))


if __name__ == "__main__":
    main()
