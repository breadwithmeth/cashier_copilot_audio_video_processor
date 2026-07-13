from __future__ import annotations

import argparse
import os
import random
import shutil
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".cache/matplotlib").resolve()))

from PIL import Image
from ultralytics import YOLO


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
PRODUCT_CLASS_ID = 0
PRODUCT_CLASS_NAME = "product"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare crop dataset and train a YOLO detector with one class: product."
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("dataset_output"),
        help="Directory with collected product crops.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("dataset_output/yolo_product_dataset"),
        help="Output YOLO dataset directory.",
    )
    parser.add_argument(
        "--model",
        default="yolov8n.yaml",
        help="Base YOLO detection model checkpoint or architecture YAML.",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default=None)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--project",
        default="runs/product_detector",
        help="Ultralytics training output project directory.",
    )
    parser.add_argument("--name", default="product")
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Only create the YOLO dataset, do not start training.",
    )
    return parser.parse_args()


def find_images(source_dir: Path, dataset_dir: Path) -> list[Path]:
    dataset_dir = dataset_dir.resolve()
    images = []
    for path in source_dir.rglob("*"):
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        resolved = path.resolve()
        if dataset_dir == resolved or dataset_dir in resolved.parents:
            continue
        images.append(path)
    return sorted(images)


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


def copy_sample(source: Path, dataset_dir: Path, split: str, index: int) -> None:
    suffix = source.suffix.lower()
    target_name = f"product_{index:06d}{suffix}"
    image_target = dataset_dir / "images" / split / target_name
    label_target = dataset_dir / "labels" / split / f"product_{index:06d}.txt"

    shutil.copy2(source, image_target)
    label_target.write_text(
        f"{PRODUCT_CLASS_ID} 0.5 0.5 0.98 0.98\n",
        encoding="utf-8",
    )


def write_data_yaml(dataset_dir: Path) -> Path:
    data_yaml = dataset_dir / "product.yaml"
    data_yaml.write_text(
        "\n".join(
            [
                f"path: {dataset_dir.resolve()}",
                "train: images/train",
                "val: images/val",
                "names:",
                f"  0: {PRODUCT_CLASS_NAME}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return data_yaml


def prepare_dataset(
    source_dir: Path,
    dataset_dir: Path,
    val_ratio: float,
    seed: int,
) -> Path:
    if not 0 < val_ratio < 0.5:
        raise ValueError("--val-ratio must be between 0 and 0.5")

    images = [path for path in find_images(source_dir, dataset_dir) if valid_image(path)]
    if len(images) < 2:
        raise RuntimeError(f"Need at least 2 valid images in {source_dir}")

    random.Random(seed).shuffle(images)
    val_count = max(1, int(len(images) * val_ratio))
    train_images = images[val_count:]
    val_images = images[:val_count]

    reset_dataset_dir(dataset_dir)
    for index, image_path in enumerate(train_images):
        copy_sample(image_path, dataset_dir, "train", index)
    for index, image_path in enumerate(val_images):
        copy_sample(image_path, dataset_dir, "val", index)

    data_yaml = write_data_yaml(dataset_dir)
    print(
        f"Prepared YOLO dataset: {len(train_images)} train, "
        f"{len(val_images)} val -> {data_yaml}"
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
    data_yaml = prepare_dataset(
        source_dir=args.source_dir,
        dataset_dir=args.dataset_dir,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    if not args.prepare_only:
        train(args, data_yaml)


if __name__ == "__main__":
    main()
