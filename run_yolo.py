from __future__ import annotations

import argparse
import glob
import os
import shutil
from collections import Counter
from pathlib import Path

import torch
from datasets import load_dataset
from ultralytics import YOLO


DATASET_NAME = "benjamintli/retail-product-checkout"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download 10% of retail-product-checkout, convert to YOLO, train, and run one prediction."
    )
    parser.add_argument("--dataset", default=DATASET_NAME)
    parser.add_argument("--train-split", default="train[:10%]")
    parser.add_argument("--val-split", default="validation[:10%]")
    parser.add_argument("--base-dir", type=Path, default=Path("hf_retail_yolo_dataset"))
    parser.add_argument("--yaml", type=Path, default=Path("hf_retail_data.yaml"))
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--project", default="retail_yolo_local")
    parser.add_argument("--name", default="10_epochs_mps")
    parser.add_argument("--prepare-only", action="store_true")
    return parser.parse_args()


def choose_device(value: str) -> str:
    if value != "auto":
        return value
    return "mps" if torch.backends.mps.is_available() else "cpu"


def reset_yolo_dirs(base_dir: Path) -> None:
    if base_dir.exists():
        shutil.rmtree(base_dir)
    for split in ("train", "val"):
        (base_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (base_dir / "labels" / split).mkdir(parents=True, exist_ok=True)


def collect_category_ids(ds: dict) -> dict[int, int]:
    category_ids = set()
    for hf_split in ds.values():
        for item in hf_split:
            bboxes = item["objects"]["bbox"]
            categories = item["objects"].get("category", [0] * len(bboxes))
            category_ids.update(int(category) for category in categories)
    if not category_ids:
        category_ids.add(0)
    return {category_id: index for index, category_id in enumerate(sorted(category_ids))}


def convert_split(hf_split, split_name: str, base_dir: Path, category_to_class: dict[int, int]) -> Counter:
    counts = Counter()
    for index, item in enumerate(hf_split):
        image = item["image"]
        if image.mode != "RGB":
            image = image.convert("RGB")
        image_path = base_dir / "images" / split_name / f"{index:06d}.jpg"
        image.save(image_path, quality=95)

        width, height = image.size
        label_path = base_dir / "labels" / split_name / f"{index:06d}.txt"
        lines = []
        bboxes = item["objects"]["bbox"]
        categories = item["objects"].get("category", [0] * len(bboxes))
        for bbox, category in zip(bboxes, categories):
            x_min, y_min, box_width, box_height = [float(value) for value in bbox]
            if box_width <= 0 or box_height <= 0:
                continue
            x_center = (x_min + box_width / 2) / width
            y_center = (y_min + box_height / 2) / height
            norm_width = box_width / width
            norm_height = box_height / height
            class_id = category_to_class[int(category)]
            lines.append(
                f"{class_id} "
                f"{_clip01(x_center):.6f} {_clip01(y_center):.6f} "
                f"{_clip01(norm_width):.6f} {_clip01(norm_height):.6f}"
            )
            counts[class_id] += 1
        label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return counts


def write_yaml(path: Path, base_dir: Path, category_to_class: dict[int, int]) -> None:
    names = {
        class_id: f"category_{category_id}"
        for category_id, class_id in category_to_class.items()
    }
    lines = [
        f"path: {base_dir.resolve()}",
        "train: images/train",
        "val: images/val",
        "names:",
    ]
    lines.extend(f"  {index}: {names[index]}" for index in range(len(names)))
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def prepare_dataset(args: argparse.Namespace) -> tuple[Path, dict[int, int]]:
    print("Downloading Hugging Face dataset slices...")
    ds = load_dataset(
        args.dataset,
        split={
            "train": args.train_split,
            "validation": args.val_split,
        },
    )
    reset_yolo_dirs(args.base_dir)
    category_to_class = collect_category_ids(ds)

    print("Converting dataset to YOLO format...")
    train_counts = convert_split(ds["train"], "train", args.base_dir, category_to_class)
    val_counts = convert_split(ds["validation"], "val", args.base_dir, category_to_class)
    write_yaml(args.yaml, args.base_dir, category_to_class)

    print(f"Prepared YOLO dataset: {len(ds['train'])} train, {len(ds['validation'])} val")
    print(f"Classes: {len(category_to_class)}")
    print(f"Train labels: {dict(train_counts)}")
    print(f"Val labels: {dict(val_counts)}")
    print(f"YAML: {args.yaml}")
    return args.yaml, category_to_class


def train_and_predict(args: argparse.Namespace, yaml_path: Path) -> None:
    device = choose_device(args.device)
    print(f"Training YOLO: model={args.model}, epochs={args.epochs}, device={device}")
    model = YOLO(args.model)
    model.train(
        data=str(yaml_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        project=args.project,
        name=args.name,
        exist_ok=True,
    )
    print("Training finished.")

    best_model_path = Path(args.project) / args.name / "weights" / "best.pt"
    if not best_model_path.exists():
        print(f"Could not find trained weights: {best_model_path}")
        return

    test_images = sorted(glob.glob(str(args.base_dir / "images" / "val" / "*.jpg")))
    if not test_images:
        print("No validation images found for prediction.")
        return

    YOLO(str(best_model_path)).predict(
        source=test_images[0],
        save=True,
        conf=0.3,
        project=args.project,
        name="test_predict",
        exist_ok=True,
    )
    print(f"Prediction saved to: {Path(args.project) / 'test_predict'}")


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def main() -> None:
    args = parse_args()
    yaml_path, _ = prepare_dataset(args)
    if not args.prepare_only:
        train_and_predict(args, yaml_path)


if __name__ == "__main__":
    main()
