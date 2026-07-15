from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import cv2


class ObjectDatasetCollector:
    """Save one high-quality crop per completed tracker ID."""

    def __init__(self, camera_name: str, output_dir: Path,
                 track_timeout: float = 2.0):
        self.camera_name = camera_name
        self.output_dir = output_dir
        self.track_timeout = track_timeout
        self._tracks: dict[int | str, dict] = {}

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
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
            score = float(obj.confidence) * crop.size * (1 + sharpness / 1000)
            track = self._tracks.get(track_id)
            if track is None or score > track["score"]:
                self._tracks[track_id] = {
                    "crop": crop.copy(), "score": score,
                    "confidence": float(obj.confidence),
                    "class_name": obj.class_name,
                    "last_seen": now,
                }
            else:
                track["last_seen"] = now

        expired = [track_id for track_id, track in self._tracks.items()
                   if track_id not in active_ids and
                   now - track["last_seen"] >= self.track_timeout]
        for track_id in expired:
            self._save(track_id, self._tracks.pop(track_id))

    def stop(self) -> None:
        for track_id, track in list(self._tracks.items()):
            self._save(track_id, track)
        self._tracks.clear()

    def _save(self, track_id, track) -> None:
        pending = self.output_dir / "pending"
        pending.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        stem = f"{self.camera_name}_track_{track_id}_{stamp}"
        image_path = pending / f"{stem}.jpg"
        if not cv2.imwrite(str(image_path), track["crop"]):
            raise RuntimeError(f"Could not save dataset crop: {image_path}")
        class_name = track.get("class_name", "product")
        sidecar = {
            "camera": self.camera_name,
            "track_id": track_id,
            "label": class_name,
            "class_name": class_name,
            "confidence": round(track["confidence"], 4),
            "created_at": datetime.now().astimezone().isoformat(),
            "status": "pending",
        }
        image_path.with_suffix(".json").write_text(
            json.dumps(sidecar, ensure_ascii=False, indent=2), encoding="utf-8")
        image_path.with_suffix(".txt").write_text(
            "0 0.5 0.5 0.98 0.98\n",
            encoding="utf-8",
        )
        print(f"[{self.camera_name}] Dataset crop saved: {image_path}")
