from __future__ import annotations

import re
import time
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image


@dataclass(frozen=True)
class VlmClassification:
    label: str
    confidence: float = 0.5


class VlmCropClassifier:
    def __init__(
        self,
        model_name: str,
        prompt: str,
        device: str = "auto",
        interval_seconds: float = 2.0,
        local_files_only: bool = True,
    ):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.prompt = prompt
        self.interval_seconds = interval_seconds
        self.device = self._torch_device(torch, device)
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            local_files_only=local_files_only,
        )
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            local_files_only=local_files_only,
        ).to(self.device).eval()
        self._last_at_by_key = {}
        self._cache = {}

    def classify(self, image_bgr: np.ndarray, cache_key: str) -> VlmClassification | None:
        if image_bgr.size == 0:
            return None

        now = time.monotonic()
        last_at = self._last_at_by_key.get(cache_key)
        cached = self._cache.get(cache_key)
        if cached is not None and last_at is not None and now - last_at < self.interval_seconds:
            return cached

        classification = self._classify_uncached(image_bgr)
        self._last_at_by_key[cache_key] = now
        self._cache[cache_key] = classification
        return classification

    def _classify_uncached(self, image_bgr: np.ndarray) -> VlmClassification | None:
        import torch

        image = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": self.prompt},
                ],
            },
        ]
        prompt = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
        )
        inputs = self.processor(
            text=prompt,
            images=[image],
            return_tensors="pt",
        ).to(self.device)

        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=16,
                do_sample=False,
            )

        input_length = inputs["input_ids"].shape[-1]
        text = self.processor.decode(
            generated_ids[0][input_length:],
            skip_special_tokens=True,
        )
        label = self._normalize_label(text)
        print(f"Qwen VLM response: raw={text.strip()!r} label={label!r}")
        if not label:
            return None

        return VlmClassification(label=label)

    @staticmethod
    def _normalize_label(text: str) -> str:
        text = text.strip().splitlines()[0].strip().lower()
        
        # Extract product name from common sentence patterns
        # e.g., "The visible retail product is a milk carton." -> "milk_carton"
        # e.g., "The main visible retail product is a glass_bottle." -> "glass_bottle"
        # e.g., "A milk carton is visible." -> "milk_carton"
        patterns = [
            r"is\s+a\s+([a-z0-9_\-\s]+)[\.\!\s]*$",
            r"is\s+an\s+([a-z0-9_\-\s]+)[\.\!\s]*$",
            r"product\s+is\s+([a-z0-9_\-\s]+)[\.\!\s]*$",
            r"item\s+is\s+([a-z0-9_\-\s]+)[\.\!\s]*$",
            r"visible\s+(?:retail\s+)?product\s+is\s+([a-z0-9_\-\s]+)[\.\!\s]*$",
            r"main\s+visible\s+(?:retail\s+)?product\s+is\s+([a-z0-9_\-\s]+)[\.\!\s]*$",
            r"^([a-z0-9_\-\s]+)[\.\!\s]*$",  # fallback: just the text
        ]
        
        extracted = None
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                extracted = match.group(1).strip()
                break
        
        if extracted is None:
            extracted = text
        
        # Clean up: keep alphanumeric, spaces, hyphens, underscores
        extracted = re.sub(r"[^a-z0-9_\-\s]+", "", extracted)
        # Replace spaces/hyphens with underscores
        extracted = re.sub(r"[\s\-]+", "_", extracted)
        extracted = re.sub(r"_+", "_", extracted).strip("_")
        
        # Limit to ~3 words (max ~64 chars)
        words = extracted.split("_")
        if len(words) > 3:
            extracted = "_".join(words[:3])
        
        return extracted[:64]

    @staticmethod
    def _torch_device(torch, device: str) -> str:
        if device != "auto":
            return device
        return "mps" if torch.backends.mps.is_available() else "cpu"
