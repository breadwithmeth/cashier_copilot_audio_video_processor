from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".cache/matplotlib").resolve()))

from PIL import Image
from ultralytics import YOLO


CLASS_NAMES = [
    "bottle",
    "can",
    "food",
    "tobacco",
    "receipt",
    "barcode_scanner",
    "id_card",
    "digital_id",
    "shopping_bag",
    "bank_card",
    "business_card",
    "basket",
]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare dataset1 COCO annotations and train a YOLO detector."
    )
    parser.add_argument("--source-dir", type=Path, default=Path("dataset1"))
    parser.add_argument(
        "--annotations",
        type=Path,
        default=Path("dataset1/labels_my-project-name_2026-07-14-01-01-44.json"),
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("dataset1/yolo_dataset"),
    )
    parser.add_argument(
        "--model",
        default="weights/yolov8s-worldv2.pt",
        help="Base YOLO detection checkpoint or architecture YAML.",
    )
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--project", default="runs/dataset1_detector")
    parser.add_argument("--name", default="dataset1")
    parser.add_argument("--prepare-only", action="store_true")
    return parser.parse_args()


def load_coco(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


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


def yolo_bbox(annotation: dict, image_width: int, image_height: int) -> tuple[float, ...]:
    x, y, width, height = annotation["bbox"]
    x_center = (x + width / 2) / image_width
    y_center = (y + height / 2) / image_height

    return (
        max(0.0, min(1.0, x_center)),
        max(0.0, min(1.0, y_center)),
        max(0.0, min(1.0, width / image_width)),
        max(0.0, min(1.0, height / image_height)),
    )


def split_images(images: list[dict], val_ratio: float, seed: int) -> tuple[list[dict], list[dict]]:
    if not 0 < val_ratio < 0.5:
        raise ValueError("--val-ratio must be between 0 and 0.5")

    shuffled = images[:]
    random.Random(seed).shuffle(shuffled)
    val_count = max(1, int(len(shuffled) * val_ratio))

    return shuffled[val_count:], shuffled[:val_count]


def write_sample(
    image: dict,
    annotations: list[dict],
    category_to_class: dict[int, int],
    source_dir: Path,
    dataset_dir: Path,
    split: str,
) -> Counter:
    image_path = source_dir / image["file_name"]
    if not image_path.exists():
        raise FileNotFoundError(f"Image listed in annotations is missing: {image_path}")
    if image_path.suffix.lower() not in IMAGE_EXTENSIONS or not valid_image(image_path):
        raise RuntimeError(f"Invalid image file: {image_path}")

    target_stem = image_path.stem
    image_target = dataset_dir / "images" / split / image_path.name
    label_target = dataset_dir / "labels" / split / f"{target_stem}.txt"
    shutil.copy2(image_path, image_target)

    label_lines = []
    counts = Counter()
    for annotation in annotations:
        class_id = category_to_class.get(annotation["category_id"])
        if class_id is None:
            continue

        x_center, y_center, width, height = yolo_bbox(
            annotation,
            image_width=image["width"],
            image_height=image["height"],
        )
        if width <= 0 or height <= 0:
            continue

        label_lines.append(
            f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
        )
        counts[CLASS_NAMES[class_id]] += 1

    label_target.write_text(
        "\n".join(label_lines) + ("\n" if label_lines else ""),
        encoding="utf-8",
    )
    return counts


def write_data_yaml(dataset_dir: Path) -> Path:
    data_yaml = dataset_dir / "dataset1.yaml"
    lines = [
        f"path: {dataset_dir.resolve()}",
        "train: images/train",
        "val: images/val",
        "names:",
    ]
    lines.extend(f"  {index}: {name}" for index, name in enumerate(CLASS_NAMES))
    lines.append("")
    data_yaml.write_text("\n".join(lines), encoding="utf-8")
    return data_yaml


def prepare_dataset(args: argparse.Namespace) -> Path:
    coco = load_coco(args.annotations)
    category_names = {category["id"]: category["name"] for category in coco["categories"]}
    category_to_class = {
        category_id: CLASS_NAMES.index(name)
        for category_id, name in category_names.items()
        if name in CLASS_NAMES
    }
    annotations_by_image = defaultdict(list)
    for annotation in coco["annotations"]:
        annotations_by_image[annotation["image_id"]].append(annotation)

    images = sorted(coco["images"], key=lambda item: item["file_name"])
    if len(images) < 2:
        raise RuntimeError("Need at least 2 annotated images to train")

    train_images, val_images = split_images(images, args.val_ratio, args.seed)
    reset_dataset_dir(args.dataset_dir)

    split_counts = {"train": Counter(), "val": Counter()}
    for image in train_images:
        split_counts["train"].update(
            write_sample(
                image,
                annotations_by_image[image["id"]],
                category_to_class,
                args.source_dir,
                args.dataset_dir,
                "train",
            )
        )
    for image in val_images:
        split_counts["val"].update(
            write_sample(
                image,
                annotations_by_image[image["id"]],
                category_to_class,
                args.source_dir,
                args.dataset_dir,
                "val",
            )
        )

    data_yaml = write_data_yaml(args.dataset_dir)
    print(
        f"Prepared YOLO dataset: {len(train_images)} train images, "
        f"{len(val_images)} val images -> {data_yaml}"
    )
    print(f"Train labels: {dict(split_counts['train'])}")
    print(f"Val labels: {dict(split_counts['val'])}")
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
