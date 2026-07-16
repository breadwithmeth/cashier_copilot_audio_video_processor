from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


@dataclass(frozen=True)
class ClipClassification:
    label: str
    confidence: float


class ClipCropClassifier:
    def __init__(
        self,
        model_name: str,
        labels_path: Path,
        device: str = "auto",
    ):
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self.device = self._torch_device(torch, device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device).eval()
        self.labels = self._read_labels(labels_path)
        self.prompts = [self._prompt(label) for label in self.labels]

        inputs = self.processor(
            text=self.prompts,
            return_tensors="pt",
            padding=True,
        ).to(self.device)
        with torch.inference_mode():
            text_features = self.model.get_text_features(**inputs)
            self.text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    def classify(self, image_bgr: np.ndarray) -> ClipClassification | None:
        import torch

        if image_bgr.size == 0:
            return None

        image = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        inputs = self.processor(
            images=image,
            return_tensors="pt",
        ).to(self.device)
        with torch.inference_mode():
            image_features = self.model.get_image_features(**inputs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            logits = image_features @ self.text_features.T
            probabilities = logits.softmax(dim=-1)[0]
            score, index = probabilities.max(dim=0)

        return ClipClassification(
            label=self.labels[int(index.item())].replace(" ", "_"),
            confidence=float(score.item()),
        )

    @staticmethod
    def _read_labels(path: Path) -> list[str]:
        labels = [
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not labels:
            raise RuntimeError(f"No CLIP labels found in {path}")
        return labels

    @staticmethod
    def _prompt(label: str) -> str:
        return f"a photo of a {label}"

    @staticmethod
    def _torch_device(torch, device: str) -> str:
        if device != "auto":
            return device
        return "mps" if torch.backends.mps.is_available() else "cpu"
