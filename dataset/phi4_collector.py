from __future__ import annotations

import json
import queue
import re
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
from PIL import Image


class FlorenceDatasetCollector:
    """Keep the best crop per track and label it with Florence-2.

    The historical class name is retained to avoid breaking imports.
    """

    def __init__(self, camera_name: str, output_dir: Path, model_name: str,
                 track_timeout: float = 2.0):
        self.camera_name = camera_name
        self.output_dir = output_dir
        self.model_name = model_name
        self.track_timeout = track_timeout
        self._tracks: dict[int | str, dict] = {}
        self._jobs: queue.Queue = queue.Queue()
        self._running = True
        self._worker = threading.Thread(target=self._label_worker, daemon=True)
        self._worker.start()

    def observe(self, frame, objects) -> None:
        now = time.monotonic()
        height, width = frame.shape[:2]
        active_ids = set()

        for index, obj in enumerate(objects):
            track_id = obj.track_id if obj.track_id is not None else f"temp_{index}"
            active_ids.add(track_id)
            x1, y1, x2, y2 = obj.bbox
            padding = max(8, int(max(x2 - x1, y2 - y1) * 0.08))
            x1, y1 = max(0, x1 - padding), max(0, y1 - padding)
            x2, y2 = min(width, x2 + padding), min(height, y2 + padding)
            if x2 <= x1 or y2 <= y1:
                continue
            crop = frame[y1:y2, x1:x2]
            sharpness = cv2.Laplacian(
                cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
            score = float(obj.confidence) * (x2 - x1) * (y2 - y1) * (1 + sharpness / 1000)
            track = self._tracks.get(track_id)
            if track is None or score > track["score"]:
                self._tracks[track_id] = {
                    "crop": crop.copy(),
                    "score": score,
                    "confidence": float(obj.confidence),
                    "last_seen": now,
                }
            else:
                track["last_seen"] = now

        expired = [track_id for track_id, track in self._tracks.items()
                   if track_id not in active_ids and
                   now - track["last_seen"] >= self.track_timeout]
        for track_id in expired:
            self._jobs.put((track_id, self._tracks.pop(track_id)))

    def stop(self) -> None:
        self._running = False
        for track_id, track in list(self._tracks.items()):
            self._jobs.put((track_id, track))
        self._tracks.clear()
        self._jobs.put(None)
        self._worker.join(timeout=30)

    def _label_worker(self) -> None:
        model = processor = generation_config = device = None
        while True:
            job = self._jobs.get()
            if job is None:
                return
            track_id, track = job
            pending_path = self._save_pending(track_id, track["crop"])
            label = "unlabeled"
            error = None
            try:
                if model is None:
                    model, processor, generation_config, device = self._load_florence()
                label = self._classify(
                    pending_path, model, processor, generation_config, device)
            except Exception as exc:
                error = str(exc)
                print(f"[{self.camera_name}] Florence-2 labeling error: {exc}")
            self._finish_sample(pending_path, label, track_id,
                                track["confidence"], error)

    def _load_florence(self):
        import torch
        from transformers import (AutoModelForCausalLM, AutoProcessor,
                                  GenerationConfig)

        device = "mps" if torch.backends.mps.is_available() else "cpu"
        dtype = torch.float16 if device == "mps" else torch.float32
        processor = AutoProcessor.from_pretrained(
            self.model_name, trust_remote_code=True, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name, trust_remote_code=True,
            torch_dtype=dtype, _attn_implementation="eager",
            local_files_only=True,
        ).to(device).eval()
        generation_config = None
        print(f"[{self.camera_name}] Florence-2 loaded on {device}")
        return model, processor, generation_config, device

    @staticmethod
    def _classify(image_path, model, processor, generation_config, device):
        import torch

        prompt = "<CAPTION>"
        image = Image.open(image_path).convert("RGB")
        inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
        inputs["pixel_values"] = inputs["pixel_values"].to(
            dtype=next(model.parameters()).dtype)
        with torch.inference_mode():
            generation_options = {
                "max_new_tokens": 64,
                "num_beams": 3,
                "do_sample": False,
            }
            if generation_config is not None:
                generation_options["generation_config"] = generation_config
            generated = model.generate(**inputs, **generation_options)
        response = processor.batch_decode(
            generated, skip_special_tokens=True,
            clean_up_tokenization_spaces=False)[0]
        parsed = processor.post_process_generation(
            response,
            task=prompt,
            image_size=image.size,
        )
        response = str(parsed.get(prompt, response))
        label = re.sub(r"[^a-z0-9]+", "_", response.lower()).strip("_")
        return label[:80] or "unknown"

    def _save_pending(self, track_id, crop) -> Path:
        pending = self.output_dir / "pending"
        pending.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        path = pending / f"{self.camera_name}_track_{track_id}_{stamp}.jpg"
        if not cv2.imwrite(str(path), crop):
            raise RuntimeError(f"Could not save dataset crop: {path}")
        return path

    def _finish_sample(self, pending_path, label, track_id, confidence, error):
        class_dir = self.output_dir / "images" / label
        class_dir.mkdir(parents=True, exist_ok=True)
        final_path = class_dir / pending_path.name
        pending_path.replace(final_path)
        record = {
            "image": str(final_path.relative_to(self.output_dir)),
            "label": label,
            "camera": self.camera_name,
            "track_id": track_id,
            "confidence": round(confidence, 4),
            "created_at": datetime.now().astimezone().isoformat(),
            "labeler": self.model_name,
        }
        if error:
            record["labeling_error"] = error
        metadata = self.output_dir / "metadata.jsonl"
        with metadata.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"[{self.camera_name}] Dataset sample: {final_path}")


# Backward-compatible alias for code written before switching from Phi-4.
Phi4DatasetCollector = FlorenceDatasetCollector
