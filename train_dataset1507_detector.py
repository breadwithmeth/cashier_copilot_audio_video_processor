from __future__ import annotations

import argparse
import os
import random
import shutil
from collections import Counter
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".cache/matplotlib").resolve()))

from PIL import Image
from ultralytics import YOLO


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare dataset1507 YOLO txt labels and train a YOLO detector."
    )
    parser.add_argument("--source-dir", type=Path, default=Path("dataset1507"))
    parser.add_argument("--images-dir", type=Path, default=Path("dataset1507/frames2"))
    parser.add_argument(
        "--labels-dir",
        type=Path,
        default=Path("dataset1507/labels_my-project-name_2026-07-15-09-59-17"),
    )
    parser.add_argument("--classes", type=Path, default=Path("labels.txt"))
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("dataset1507/yolo_dataset"),
    )
    parser.add_argument("--model", default="weights/yolov8m-worldv2.pt")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--project", default="runs/dataset1507_detector")
    parser.add_argument("--name", default="dataset1507")
    parser.add_argument("--prepare-only", action="store_true")
    return parser.parse_args()


def read_classes(path: Path) -> list[str]:
    names = [
        line.strip().replace(" ", "_")
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not names:
        raise RuntimeError(f"No class names found in {path}")
    return names


def valid_image(path: Path) -> bool:
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except Exception:
        return False


def reset_dataset_dir(dataset_dir: Path) -> None:
    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)
    for split in ("train", "val"):
        (dataset_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (dataset_dir / "labels" / split).mkdir(parents=True, exist_ok=True)


def label_counts(label_path: Path) -> Counter:
    counts = Counter()
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if not parts:
            continue
        counts[int(parts[0])] += 1
    return counts


def find_labeled_images(images_dir: Path, labels_dir: Path) -> list[tuple[Path, Path]]:
    samples = []
    for label_path in sorted(labels_dir.glob("*.txt")):
        image_path = next(
            (
                images_dir / f"{label_path.stem}{extension}"
                for extension in IMAGE_EXTENSIONS
                if (images_dir / f"{label_path.stem}{extension}").exists()
            ),
            None,
        )
        if image_path is None:
            raise FileNotFoundError(f"No image found for label file {label_path}")
        if not valid_image(image_path):
            raise RuntimeError(f"Invalid image file: {image_path}")
        samples.append((image_path, label_path))
    return samples


def split_samples(
    samples: list[tuple[Path, Path]],
    val_ratio: float,
    seed: int,
) -> tuple[list[tuple[Path, Path]], list[tuple[Path, Path]]]:
    if not 0 < val_ratio < 0.5:
        raise ValueError("--val-ratio must be between 0 and 0.5")
    if len(samples) < 2:
        raise RuntimeError("Need at least 2 labeled images to train")

    shuffled = samples[:]
    random.Random(seed).shuffle(shuffled)
    val_count = max(1, int(len(shuffled) * val_ratio))
    return shuffled[val_count:], shuffled[:val_count]


def copy_sample(
    image_path: Path,
    label_path: Path,
    dataset_dir: Path,
    split: str,
) -> None:
    shutil.copy2(image_path, dataset_dir / "images" / split / image_path.name)
    shutil.copy2(label_path, dataset_dir / "labels" / split / label_path.name)


def write_data_yaml(dataset_dir: Path, class_names: list[str]) -> Path:
    data_yaml = dataset_dir / "dataset1507.yaml"
    lines = [
        f"path: {dataset_dir.resolve()}",
        "train: images/train",
        "val: images/val",
        "names:",
    ]
    lines.extend(f"  {index}: {name}" for index, name in enumerate(class_names))
    lines.append("")
    data_yaml.write_text("\n".join(lines), encoding="utf-8")
    return data_yaml


def prepare_dataset(args: argparse.Namespace) -> Path:
    class_names = read_classes(args.classes)
    samples = find_labeled_images(args.images_dir, args.labels_dir)
    train_samples, val_samples = split_samples(samples, args.val_ratio, args.seed)

    reset_dataset_dir(args.dataset_dir)
    split_counts = {"train": Counter(), "val": Counter()}
    for image_path, label_path in train_samples:
        copy_sample(image_path, label_path, args.dataset_dir, "train")
        split_counts["train"].update(label_counts(label_path))
    for image_path, label_path in val_samples:
        copy_sample(image_path, label_path, args.dataset_dir, "val")
        split_counts["val"].update(label_counts(label_path))

    data_yaml = write_data_yaml(args.dataset_dir, class_names)
    print(
        f"Prepared YOLO dataset: {len(train_samples)} train images, "
        f"{len(val_samples)} val images -> {data_yaml}"
    )
    print(
        "Train labels:",
        {class_names[class_id]: count for class_id, count in split_counts["train"].items()},
    )
    print(
        "Val labels:",
        {class_names[class_id]: count for class_id, count in split_counts["val"].items()},
    )
    return data_yaml


def train(args: argparse.Namespace, data_yaml: Path) -> None:
    model = YOLO(args.model)
    train_kwargs = {
        "data": str(data_yaml),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "project": str(Path(args.project).resolve()),
        "name": args.name,
        "exist_ok": True,
    }
    if args.device is not None:
        train_kwargs["device"] = args.device
    model.train(**train_kwargs)


def main() -> None:
    args = parse_args()
    data_yaml = prepare_dataset(args)
    if not args.prepare_only:
        train(args, data_yaml)


if __name__ == "__main__":
    main()
