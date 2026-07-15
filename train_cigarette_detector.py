from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import Counter
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".cache/matplotlib").resolve()))

from PIL import Image
from ultralytics import YOLO


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLASS_ID = 0
CLASS_NAME = "tobacco"
SOURCE_SPLITS = {
    "train": "train",
    "valid": "val",
    "test": "val",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge all COCO datasets under cigarette_dataset into one YOLO "
            "dataset and train a one-class tobacco detector."
        )
    )
    parser.add_argument("--source-dir", type=Path, default=Path("cigarette_dataset"))
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("cigarette_dataset/yolo_tobacco_dataset"),
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
    parser.add_argument("--project", default="runs/cigarette_detector")
    parser.add_argument("--name", default="tobacco_3datasets")
    parser.add_argument(
        "--copy-images",
        action="store_true",
        help="Copy images instead of hardlinking them into the prepared dataset.",
    )
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


def write_sample(
    image: dict,
    annotations: list[dict],
    source_split_dir: Path,
    dataset_dir: Path,
    target_split: str,
    dataset_name: str,
    source_split: str,
    copy_images: bool,
) -> Counter:
    image_path = source_split_dir / image["file_name"]
    if not image_path.exists():
        raise FileNotFoundError(f"Image listed in annotations is missing: {image_path}")
    if image_path.suffix.lower() not in IMAGE_EXTENSIONS or not valid_image(image_path):
        raise RuntimeError(f"Invalid image file: {image_path}")

    safe_dataset = dataset_name.replace(" ", "_").replace("/", "_")
    target_stem = f"{safe_dataset}_{source_split}_{image_path.stem}"
    image_target = dataset_dir / "images" / target_split / f"{target_stem}{image_path.suffix.lower()}"
    label_target = dataset_dir / "labels" / target_split / f"{target_stem}.txt"
    if copy_images:
        shutil.copy2(image_path, image_target)
    else:
        image_target.hardlink_to(image_path.resolve())

    lines = []
    counts = Counter()
    for annotation in annotations:
        x_center, y_center, width, height = yolo_bbox(
            annotation,
            image_width=image["width"],
            image_height=image["height"],
        )
        if width <= 0 or height <= 0:
            continue
        lines.append(
            f"{CLASS_ID} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
        )
        counts[CLASS_NAME] += 1

    label_target.write_text(
        "\n".join(lines) + ("\n" if lines else ""),
        encoding="utf-8",
    )
    return counts


def write_data_yaml(dataset_dir: Path) -> Path:
    data_yaml = dataset_dir / "tobacco.yaml"
    data_yaml.write_text(
        "\n".join(
            [
                f"path: {dataset_dir.resolve()}",
                "train: images/train",
                "val: images/val",
                "names:",
                f"  {CLASS_ID}: {CLASS_NAME}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return data_yaml


def prepare_dataset(args: argparse.Namespace) -> Path:
    reset_dataset_dir(args.dataset_dir)
    split_counts = {"train": Counter(), "val": Counter()}
    split_images = Counter()

    annotation_paths = sorted(args.source_dir.glob("*/*/_annotations.coco.json"))
    if not annotation_paths:
        raise RuntimeError(f"No COCO annotation files found under {args.source_dir}")

    for annotation_path in annotation_paths:
        dataset_name = annotation_path.parents[1].name
        source_split = annotation_path.parent.name
        target_split = SOURCE_SPLITS.get(source_split)
        if target_split is None:
            continue

        coco = load_coco(annotation_path)
        annotations_by_image = {}
        for annotation in coco.get("annotations", []):
            annotations_by_image.setdefault(annotation["image_id"], []).append(annotation)

        for image in sorted(coco.get("images", []), key=lambda item: item["file_name"]):
            split_counts[target_split].update(
                write_sample(
                    image=image,
                    annotations=annotations_by_image.get(image["id"], []),
                    source_split_dir=annotation_path.parent,
                    dataset_dir=args.dataset_dir,
                    target_split=target_split,
                    dataset_name=dataset_name,
                    source_split=source_split,
                    copy_images=args.copy_images,
                )
            )
            split_images[target_split] += 1

    data_yaml = write_data_yaml(args.dataset_dir)
    print(
        "Prepared YOLO tobacco dataset: "
        f"{split_images['train']} train images, {split_images['val']} val images"
    )
    print(f"Train labels: {dict(split_counts['train'])}")
    print(f"Val labels: {dict(split_counts['val'])}")
    print(f"Data YAML: {data_yaml}")
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
