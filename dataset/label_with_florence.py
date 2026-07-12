from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from config import DATASET_DIR, FLORENCE_MODEL
from dataset.phi4_collector import FlorenceDatasetCollector


def label_pending(dataset_dir: Path, model_name: str) -> int:
    pending = dataset_dir / "pending"
    images = sorted(pending.glob("*.jpg")) if pending.exists() else []
    if not images:
        print(f"No pending images in {pending}")
        return 0

    runtime = object.__new__(FlorenceDatasetCollector)
    runtime.model_name = model_name
    runtime.camera_name = "dataset-labeler"
    model, processor, generation, device = runtime._load_florence()
    metadata_path = dataset_dir / "metadata.jsonl"

    labeled = 0
    for image_path in images:
        sidecar_path = image_path.with_suffix(".json")
        info = (json.loads(sidecar_path.read_text(encoding="utf-8"))
                if sidecar_path.exists() else {})
        try:
            label = runtime._classify(
                image_path, model, processor, generation, device)
        except Exception as error:
            print(f"Failed {image_path.name}: {error}")
            continue

        class_dir = dataset_dir / "images" / label
        class_dir.mkdir(parents=True, exist_ok=True)
        final_path = class_dir / image_path.name
        image_path.replace(final_path)
        sidecar_path.unlink(missing_ok=True)
        info.update({
            "image": str(final_path.relative_to(dataset_dir)),
            "label": label,
            "status": "labeled",
            "labeled_at": datetime.now().astimezone().isoformat(),
            "labeler": model_name,
        })
        with metadata_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(info, ensure_ascii=False) + "\n")
        labeled += 1
        print(f"[{labeled}/{len(images)}] {image_path.name} -> {label}")
    return labeled


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Label pending object crops with Florence-2")
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR)
    parser.add_argument("--model", default=FLORENCE_MODEL)
    args = parser.parse_args()
    count = label_pending(args.dataset_dir, args.model)
    print(f"Labeled images: {count}")


if __name__ == "__main__":
    main()
