import json
import os
import re
import time

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO, YOLOWorld

from config import (
    SCAN_BACKEND,
    SCAN_DEVICE,
    SCAN_MODEL_PATH,
    SCAN_CONFIDENCE,
    SCAN_IMAGE_SIZE,
    SCAN_WORLD_PROMPTS,
    SCAN_OWLV2_MODEL,
    SCAN_OWLV2_CLASSES,
    SCAN_OWLV2_PROMPTS,
    SCAN_OMDET_MODEL,
    SCAN_OMDET_CLASSES,
    SCAN_OMDET_PROMPTS,
    SCAN_SMOLVLM_INTERVAL_SECONDS,
    SCAN_SMOLVLM_MODEL,
    SCAN_SMOLVLM_PROMPT,
    SCAN_SMOLVLM_PROMPTS,
)

from models.detection import Detection, ScanResult
from vision.roi import bbox_center_in_roi, crop_roi, offset_bbox


GENERIC_SCAN_CLASSES = {
    "retail product",
    "store product",
    "bar market product",
    "alcohol product",
    "beverage product",
    "food package",
    "snack package",
    "glass bottle",
    "plastic bottle",
    "aluminum can",
    "metal can",
    "carton box",
    "paper box",
}

DUPLICATE_IOU_THRESHOLD = 0.65
SPECIFIC_CLASS_BONUS = 0.03


class ScanDetector:
    def __init__(self, roi):
        self.roi = roi
        self.backend = SCAN_BACKEND
        self.model = None
        self.processor = None
        self.device = None
        self._smolvlm_last_at = 0.0
        self._smolvlm_cached_objects = []

        if self.backend == "yolo":
            self.model = YOLO(str(SCAN_MODEL_PATH))
        elif self.backend == "yolo_world":
            self.model = YOLOWorld(str(SCAN_MODEL_PATH))
            if SCAN_WORLD_PROMPTS:
                self.model.set_classes(SCAN_WORLD_PROMPTS)
        elif self.backend == "owlv2":
            self._load_owlv2()
        elif self.backend == "omdet_turbo":
            self._load_omdet_turbo()
        elif self.backend == "smolvlm":
            self._load_smolvlm()
        else:
            raise ValueError(
                "SCAN_BACKEND must be one of: "
                "yolo, yolo_world, owlv2, omdet_turbo, smolvlm"
            )

    def detect(self, frame) -> ScanResult:
        roi_frame, clipped_roi = crop_roi(frame, self.roi)

        x1, y1, _, _ = clipped_roi

        if roi_frame.size == 0:
            return ScanResult(objects=[], process_ms=0)

        start = time.time()

        if self.backend == "owlv2":
            return self._detect_owlv2(
                roi_frame=roi_frame,
                clipped_roi=clipped_roi,
                start=start,
            )
        if self.backend == "omdet_turbo":
            return self._detect_omdet_turbo(
                roi_frame=roi_frame,
                clipped_roi=clipped_roi,
                start=start,
            )
        if self.backend == "smolvlm":
            return self._detect_smolvlm(
                roi_frame=roi_frame,
                clipped_roi=clipped_roi,
                start=start,
            )

        return self._detect_ultralytics(
            roi_frame=roi_frame,
            clipped_roi=clipped_roi,
            start=start,
        )

    def _detect_ultralytics(self, roi_frame, clipped_roi, start) -> ScanResult:
        x1, y1, _, _ = clipped_roi

        results = self.model.track(
            source=roi_frame,
            conf=SCAN_CONFIDENCE,
            imgsz=SCAN_IMAGE_SIZE,
            persist=True,
            tracker="trackers/bytetrack_retail.yaml",
            verbose=False,
        )

        process_ms = int((time.time() - start) * 1000)

        objects = []

        for result in results:
            if result.boxes is None:
                continue

            masks = getattr(result, "masks", None)
            mask_polygons = getattr(masks, "xy", None) if masks is not None else None

            for index, box in enumerate(result.boxes):
                cls = int(box.cls.item())
                conf = float(box.conf.item())
                track_id = (
                    int(box.id.item())
                    if box.id is not None
                    else None
                )

                bx1, by1, bx2, by2 = box.xyxy[0].tolist()

                local_bbox = (
                    int(bx1),
                    int(by1),
                    int(bx2),
                    int(by2),
                )

                if not bbox_center_in_roi(local_bbox, self.roi, x1, y1):
                    continue

                full_bbox = offset_bbox(
                    local_bbox,
                    x1,
                    y1,
                )
                polygon = self._object_polygon(
                    mask_polygons,
                    index,
                    local_bbox,
                    x1,
                    y1,
                )

                objects.append(
                    Detection(
                        class_name=result.names.get(cls, "product"),
                        confidence=conf,
                        bbox=full_bbox,
                        roi_name="scan_zone",
                        track_id=track_id,
                        polygon=polygon,
                    )
                )

        return ScanResult(
            objects=self._deduplicate_objects(objects),
            process_ms=process_ms,
        )

    def _load_owlv2(self) -> None:
        import torch
        from transformers import Owlv2ForObjectDetection, Owlv2Processor

        self.device = self._torch_device(torch)
        self.processor = Owlv2Processor.from_pretrained(SCAN_OWLV2_MODEL)
        self.model = Owlv2ForObjectDetection.from_pretrained(
            SCAN_OWLV2_MODEL
        ).to(self.device).eval()

    def _load_omdet_turbo(self) -> None:
        import torch
        from transformers import OmDetTurboForObjectDetection, OmDetTurboProcessor

        self.device = self._torch_device(torch)
        self.processor = OmDetTurboProcessor.from_pretrained(SCAN_OMDET_MODEL)
        self.model = OmDetTurboForObjectDetection.from_pretrained(
            SCAN_OMDET_MODEL
        ).to(self.device).eval()

    def _load_smolvlm(self) -> None:
        import torch
        from transformers import AutoModelForVision2Seq, AutoProcessor

        self.device = self._torch_device(torch)
        dtype = torch.float16 if self.device == "mps" else torch.float32
        self.processor = AutoProcessor.from_pretrained(SCAN_SMOLVLM_MODEL)
        self.model = AutoModelForVision2Seq.from_pretrained(
            SCAN_SMOLVLM_MODEL,
            torch_dtype=dtype,
            _attn_implementation="eager",
        ).to(self.device).eval()

    @staticmethod
    def _torch_device(torch):
        if SCAN_DEVICE != "auto":
            return SCAN_DEVICE

        return "mps" if torch.backends.mps.is_available() else "cpu"

    def _detect_owlv2(self, roi_frame, clipped_roi, start) -> ScanResult:
        import torch

        x1, y1, _, _ = clipped_roi
        image = Image.fromarray(cv2.cvtColor(roi_frame, cv2.COLOR_BGR2RGB))
        texts = [SCAN_OWLV2_PROMPTS]
        inputs = self.processor(
            text=texts,
            images=image,
            return_tensors="pt",
        ).to(self.device)

        with torch.inference_mode():
            outputs = self.model(**inputs)

        target_sizes = torch.tensor(
            [image.size[::-1]],
            device=self.device,
        )
        results = self.processor.post_process_object_detection(
            outputs=outputs,
            target_sizes=target_sizes,
            threshold=SCAN_CONFIDENCE,
        )[0]

        objects = []
        for box, score, label in zip(
            results["boxes"],
            results["scores"],
            results["labels"],
        ):
            bx1, by1, bx2, by2 = box.tolist()
            local_bbox = (
                int(bx1),
                int(by1),
                int(bx2),
                int(by2),
            )

            if not bbox_center_in_roi(local_bbox, self.roi, x1, y1):
                continue

            full_bbox = offset_bbox(local_bbox, x1, y1)
            class_name = SCAN_OWLV2_CLASSES[int(label.item())]["label"].replace(" ", "_")
            polygon = self._bbox_polygon(local_bbox, x1, y1)

            objects.append(
                Detection(
                    class_name=class_name,
                    confidence=float(score.item()),
                    bbox=full_bbox,
                    roi_name="scan_zone",
                    track_id=None,
                    polygon=polygon,
                )
            )

        process_ms = int((time.time() - start) * 1000)

        return ScanResult(
            objects=self._deduplicate_objects(objects),
            process_ms=process_ms,
        )

    def _detect_omdet_turbo(self, roi_frame, clipped_roi, start) -> ScanResult:
        import torch

        x1, y1, _, _ = clipped_roi
        image = Image.fromarray(cv2.cvtColor(roi_frame, cv2.COLOR_BGR2RGB))
        inputs = self.processor(
            images=image,
            text=SCAN_OMDET_PROMPTS,
            return_tensors="pt",
        ).to(self.device)

        with torch.inference_mode():
            outputs = self.model(**inputs)

        target_sizes = torch.tensor(
            [image.size[::-1]],
            device=self.device,
        )
        results = self.processor.post_process_grounded_object_detection(
            outputs=outputs,
            classes=[SCAN_OMDET_PROMPTS],
            score_threshold=SCAN_CONFIDENCE,
            target_sizes=target_sizes,
        )[0]

        objects = []
        for box, score, label in zip(
            results["boxes"],
            results["scores"],
            results["classes"],
        ):
            bx1, by1, bx2, by2 = box.tolist()
            local_bbox = (
                int(bx1),
                int(by1),
                int(bx2),
                int(by2),
            )

            if not bbox_center_in_roi(local_bbox, self.roi, x1, y1):
                continue

            full_bbox = offset_bbox(local_bbox, x1, y1)
            class_name = self._omdet_label(label)
            polygon = self._bbox_polygon(local_bbox, x1, y1)

            objects.append(
                Detection(
                    class_name=class_name,
                    confidence=float(score.item()),
                    bbox=full_bbox,
                    roi_name="scan_zone",
                    track_id=None,
                    polygon=polygon,
                )
            )

        process_ms = int((time.time() - start) * 1000)

        return ScanResult(
            objects=self._deduplicate_objects(objects),
            process_ms=process_ms,
        )

    def _detect_smolvlm(self, roi_frame, clipped_roi, start) -> ScanResult:
        import torch

        now = time.monotonic()
        if now - self._smolvlm_last_at < SCAN_SMOLVLM_INTERVAL_SECONDS:
            return ScanResult(
                objects=list(self._smolvlm_cached_objects),
                process_ms=0,
            )

        image = Image.fromarray(cv2.cvtColor(roi_frame, cv2.COLOR_BGR2RGB))
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": SCAN_SMOLVLM_PROMPT},
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
                max_new_tokens=80,
                do_sample=False,
            )

        input_length = inputs["input_ids"].shape[-1]
        text = self.processor.decode(
            generated_ids[0][input_length:],
            skip_special_tokens=True,
        )
        labels = self._parse_smolvlm_labels(text)
        objects = self._smolvlm_objects(labels, clipped_roi)
        self._smolvlm_last_at = now
        self._smolvlm_cached_objects = objects

        process_ms = int((time.time() - start) * 1000)

        return ScanResult(objects=objects, process_ms=process_ms)

    @classmethod
    def _deduplicate_objects(cls, objects):
        kept = []

        for obj in sorted(objects, key=cls._dedupe_rank, reverse=True):
            if any(
                cls._iou(obj.bbox, kept_obj.bbox) >= DUPLICATE_IOU_THRESHOLD
                for kept_obj in kept
            ):
                continue

            kept.append(obj)

        return kept

    @classmethod
    def _dedupe_rank(cls, detection):
        return (
            detection.confidence
            + SPECIFIC_CLASS_BONUS * cls._specificity(detection.class_name),
            detection.confidence,
        )

    @staticmethod
    def _specificity(class_name):
        if class_name in GENERIC_SCAN_CLASSES:
            return 0

        return min(2, max(0, len(class_name.split()) - 1))

    @staticmethod
    def _prompt_label(prompts, label):
        if hasattr(label, "item"):
            label = label.item()
        if isinstance(label, int):
            return prompts[label].replace(" ", "_")

        return str(label).replace(" ", "_")

    @staticmethod
    def _omdet_label(label):
        if hasattr(label, "item"):
            label = label.item()
        if isinstance(label, int):
            return SCAN_OMDET_CLASSES[label]["label"].replace(" ", "_")

        label_text = str(label)
        for item in SCAN_OMDET_CLASSES:
            if label_text == item["prompt"]:
                return item["label"].replace(" ", "_")

        return label_text.replace(" ", "_")

    @staticmethod
    def _parse_smolvlm_labels(text):
        allowed = {label.lower(): label for label in SCAN_SMOLVLM_PROMPTS}
        labels = []

        try:
            match = re.search(r"\[[\s\S]*\]", text)
            parsed = json.loads(match.group(0) if match else text)
            if isinstance(parsed, list):
                candidates = parsed
            else:
                candidates = []
        except (json.JSONDecodeError, TypeError):
            lowered = text.lower()
            candidates = [
                label
                for label in SCAN_SMOLVLM_PROMPTS
                if label.lower() in lowered
            ]

        for candidate in candidates:
            key = str(candidate).strip().lower().replace("_", " ")
            label = allowed.get(key)
            if label and label not in labels:
                labels.append(label)

        return labels

    @classmethod
    def _smolvlm_objects(cls, labels, clipped_roi):
        x1, y1, x2, y2 = clipped_roi
        bbox = (int(x1), int(y1), int(x2), int(y2))
        polygon = cls._bbox_polygon((0, 0, x2 - x1, y2 - y1), x1, y1)

        return [
            Detection(
                class_name=label.replace(" ", "_"),
                confidence=0.5,
                bbox=bbox,
                roi_name="scan_zone",
                track_id=None,
                polygon=polygon,
            )
            for label in labels
        ]

    @staticmethod
    def _iou(first, second):
        ax1, ay1, ax2, ay2 = first
        bx1, by1, bx2, by2 = second

        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)

        inter_width = max(0, ix2 - ix1)
        inter_height = max(0, iy2 - iy1)
        inter_area = inter_width * inter_height

        first_area = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        second_area = max(0, bx2 - bx1) * max(0, by2 - by1)
        union_area = first_area + second_area - inter_area

        if union_area <= 0:
            return 0.0

        return inter_area / union_area

    @classmethod
    def _object_polygon(cls, mask_polygons, index, local_bbox, offset_x, offset_y):
        if mask_polygons is not None and index < len(mask_polygons):
            mask_polygon = cls._simplify_mask_polygon(mask_polygons[index])
            if len(mask_polygon) >= 3:
                return [
                    (int(x + offset_x), int(y + offset_y))
                    for x, y in mask_polygon
                ]

        return cls._bbox_polygon(local_bbox, offset_x, offset_y)

    @staticmethod
    def _simplify_mask_polygon(mask_polygon):
        points = np.asarray(mask_polygon, dtype=np.float32)
        if len(points) < 3:
            return []

        epsilon = 0.01 * cv2.arcLength(points, closed=True)
        simplified = cv2.approxPolyDP(points, epsilon, closed=True)

        return [
            (float(point[0][0]), float(point[0][1]))
            for point in simplified
        ]

    @staticmethod
    def _bbox_polygon(bbox, offset_x, offset_y):
        x1, y1, x2, y2 = bbox

        return [
            (int(x1 + offset_x), int(y1 + offset_y)),
            (int(x2 + offset_x), int(y1 + offset_y)),
            (int(x2 + offset_x), int(y2 + offset_y)),
            (int(x1 + offset_x), int(y2 + offset_y)),
        ]
