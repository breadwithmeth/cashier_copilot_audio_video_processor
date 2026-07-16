from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from huggingface_hub.utils import EntryNotFoundError, LocalEntryNotFoundError
from PIL import Image


@dataclass(frozen=True)
class ClipClassification:
    label: str
    confidence: float
    is_negative: bool = False


class ClipCropClassifier:
    def __init__(
        self,
        model_name: str,
        labels_path: Path,
        negative_labels: list[str] | None = None,
        device: str = "auto",
    ):
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self.device = self._torch_device(torch, device)
        self.processor = self._load_processor(CLIPProcessor, model_name)
        self.model = self._load_model(CLIPModel, model_name).to(self.device).eval()
        self.labels = self._read_labels(labels_path)
        self.negative_labels = [label.strip() for label in (negative_labels or []) if label.strip()]
        self._negative_label_set = {
            self._normalize_label(label)
            for label in self.negative_labels
        }
        self.labels = self.labels + self.negative_labels
        self.prompts = [self._prompt(label) for label in self.labels]

        inputs = self.processor(
            text=self.prompts,
            return_tensors="pt",
            padding=True,
        ).to(self.device)
        with torch.inference_mode():
            text_features = self._feature_tensor(self.model.get_text_features(**inputs))
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
            image_features = self._feature_tensor(self.model.get_image_features(**inputs))
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            logits = image_features @ self.text_features.T
            probabilities = logits.softmax(dim=-1)[0]
            score, index = probabilities.max(dim=0)

        return ClipClassification(
            label=self._normalize_label(self.labels[int(index.item())]),
            confidence=float(score.item()),
            is_negative=self._normalize_label(self.labels[int(index.item())])
            in self._negative_label_set,
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
    def _normalize_label(label: str) -> str:
        return label.replace(" ", "_")

    @staticmethod
    def _feature_tensor(output):
        if hasattr(output, "norm"):
            return output
        for attr in ("text_embeds", "image_embeds", "pooler_output", "last_hidden_state"):
            value = getattr(output, attr, None)
            if value is None:
                continue
            if attr == "last_hidden_state":
                return value[:, 0]
            return value
        if isinstance(output, (tuple, list)) and output:
            return output[0]
        raise TypeError(f"Unsupported CLIP feature output type: {type(output)!r}")

    @staticmethod
    def _load_processor(processor_cls, model_name: str):
        try:
            return processor_cls.from_pretrained(model_name, local_files_only=True)
        except (OSError, EntryNotFoundError, LocalEntryNotFoundError):
            return processor_cls.from_pretrained(model_name)

    @staticmethod
    def _load_model(model_cls, model_name: str):
        try:
            return model_cls.from_pretrained(model_name, local_files_only=True)
        except (OSError, EntryNotFoundError, LocalEntryNotFoundError):
            return model_cls.from_pretrained(model_name)

    @staticmethod
    def _torch_device(torch, device: str) -> str:
        if device != "auto":
            return device
        return "mps" if torch.backends.mps.is_available() else "cpu"
