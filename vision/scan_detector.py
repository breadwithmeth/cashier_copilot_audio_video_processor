import json
import os
import re
import threading
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
    SCAN_CLIP_CLASSIFICATION_ENABLED,
    SCAN_CLIP_LABELS_PATH,
    SCAN_CLIP_MIN_CONFIDENCE,
    SCAN_CLIP_MODEL,
    SCAN_CLIP_NEGATIVE_LABELS,
    SCAN_VLM_CLASSIFICATION_ENABLED,
    SCAN_VLM_CLASSIFICATION_INTERVAL_SECONDS,
    SCAN_VLM_CLASSIFICATION_LOCAL_FILES_ONLY,
    SCAN_VLM_CLASSIFICATION_MAX_OBJECTS,
    SCAN_VLM_CLASSIFICATION_MIN_CLIP_CONFIDENCE,
    SCAN_VLM_CLASSIFICATION_MODE,
    SCAN_VLM_CLASSIFICATION_MODEL,
    SCAN_VLM_CLASSIFICATION_PROMPT,
    SCAN_WORLD_PROMPTS,
    SCAN_OWLV2_MODEL,
    SCAN_OWLV2_CLASSES,
    SCAN_OWLV2_PROMPTS,
    SCAN_OMDET_MODEL,
    SCAN_OMDET_CLASSES,
    SCAN_OMDET_PROMPTS,
    SCAN_RT_DETR_V2_MODEL,
    SCAN_SMOLVLM_INTERVAL_SECONDS,
    SCAN_SMOLVLM_MODEL,
    SCAN_SMOLVLM_PROMPT,
    SCAN_SMOLVLM_PROMPTS,
)

from models.detection import Detection, ScanResult
from vision.clip_classifier import ClipCropClassifier
from vision.roi import bbox_center_in_roi, crop_roi, offset_bbox, scale_roi
from vision.vlm_crop_classifier import VlmCropClassifier


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
SINGLE_INSTANCE_CLASSES = {
    "scanner",
    "barcode_scanner",
}
SCANNER_TRACK_ID = 1
RESERVED_TRACK_ID_OFFSET = 100000


class ScanDetector:
    def __init__(self, roi):
        self.roi = roi
        self.backend = SCAN_BACKEND
        self.model = None
        self.processor = None
        self.device = None
        self.clip_classifier = None
        self.vlm_classifier = None
        self._clip_cache = {}
        self._vlm_cache = {}
        self._vlm_lock = threading.Lock()
        self._vlm_loading = False
        self._vlm_load_failed = False
        self._smolvlm_lock = threading.Lock()
        self._smolvlm_loading = False
        self._smolvlm_load_failed = False
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
        elif self.backend == "rt_detr_v2":
            self._load_rt_detr_v2()
        elif self.backend == "smolvlm":
            print(
                "SmolVLM detector will load in background on first use: "
                f"{SCAN_SMOLVLM_MODEL}"
            )
        else:
            raise ValueError(
                "SCAN_BACKEND must be one of: "
                "yolo, yolo_world, owlv2, omdet_turbo, rt_detr_v2, smolvlm"
            )

        if SCAN_CLIP_CLASSIFICATION_ENABLED:
            self.clip_classifier = ClipCropClassifier(
                model_name=SCAN_CLIP_MODEL,
                labels_path=SCAN_CLIP_LABELS_PATH,
                negative_labels=SCAN_CLIP_NEGATIVE_LABELS,
                device=SCAN_DEVICE,
            )

        if SCAN_VLM_CLASSIFICATION_ENABLED:
            print(
                "VLM crop classifier will load in background on first use: "
                f"{SCAN_VLM_CLASSIFICATION_MODEL}"
            )

    def detect(self, frame) -> ScanResult:
        active_roi = scale_roi(self.roi, frame)
        roi_frame, clipped_roi = crop_roi(frame, active_roi)

        x1, y1, _, _ = clipped_roi

        if roi_frame.size == 0:
            return ScanResult(objects=[], process_ms=0)

        start = time.time()
        result = None

        if self.backend == "owlv2":
            result = self._detect_owlv2(
                roi_frame=roi_frame,
                clipped_roi=clipped_roi,
                active_roi=active_roi,
                start=start,
            )
        elif self.backend == "omdet_turbo":
            result = self._detect_omdet_turbo(
                roi_frame=roi_frame,
                clipped_roi=clipped_roi,
                active_roi=active_roi,
                start=start,
            )
        elif self.backend == "rt_detr_v2":
            result = self._detect_rt_detr_v2(
                roi_frame=roi_frame,
                clipped_roi=clipped_roi,
                active_roi=active_roi,
                start=start,
            )
        elif self.backend == "smolvlm":
            result = self._detect_smolvlm(
                roi_frame=roi_frame,
                clipped_roi=clipped_roi,
                start=start,
            )
        else:
            result = self._detect_ultralytics(
                roi_frame=roi_frame,
                clipped_roi=clipped_roi,
                active_roi=active_roi,
                start=start,
            )

        return self._classify_objects(frame, result)

    def _detect_ultralytics(self, roi_frame, clipped_roi, active_roi, start) -> ScanResult:
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

                if not bbox_center_in_roi(local_bbox, active_roi, x1, y1):
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

    def _load_rt_detr_v2(self) -> None:
        import torch
        from transformers import (
            AutoImageProcessor,
            AutoModelForObjectDetection,
            RTDetrImageProcessor,
        )

        self.device = self._torch_device(torch)
        try:
            self.processor = AutoImageProcessor.from_pretrained(SCAN_RT_DETR_V2_MODEL)
        except OSError:
            # Some community checkpoints do not ship preprocessor_config.json.
            # RT-DETR defaults match the common 640x640 resize/rescale pipeline.
            self.processor = RTDetrImageProcessor()
        try:
            self.model = AutoModelForObjectDetection.from_pretrained(
                SCAN_RT_DETR_V2_MODEL
            ).to(self.device).eval()
        except OSError as error:
            raise RuntimeError(
                f"Cannot load RT-DETR v2 model {SCAN_RT_DETR_V2_MODEL!r}. "
                "The Hugging Face repository must contain standard Transformers "
                "weights such as model.safetensors or pytorch_model.bin."
            ) from error

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

    def _detect_owlv2(self, roi_frame, clipped_roi, active_roi, start) -> ScanResult:
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

            if not bbox_center_in_roi(local_bbox, active_roi, x1, y1):
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

    def _detect_omdet_turbo(self, roi_frame, clipped_roi, active_roi, start) -> ScanResult:
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

            if not bbox_center_in_roi(local_bbox, active_roi, x1, y1):
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

    def _detect_rt_detr_v2(self, roi_frame, clipped_roi, active_roi, start) -> ScanResult:
        import torch

        x1, y1, _, _ = clipped_roi
        image = Image.fromarray(cv2.cvtColor(roi_frame, cv2.COLOR_BGR2RGB))
        inputs = self.processor(
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

            if not bbox_center_in_roi(local_bbox, active_roi, x1, y1):
                continue

            full_bbox = offset_bbox(local_bbox, x1, y1)
            class_name = self._model_label(int(label.item()))
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

        if self.model is None or self.processor is None:
            self._ensure_smolvlm_loading()
            return ScanResult(
                objects=list(self._smolvlm_cached_objects),
                process_ms=int((time.time() - start) * 1000),
            )

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

    def _ensure_smolvlm_loading(self) -> None:
        if self._smolvlm_load_failed or self._smolvlm_loading:
            return

        with self._smolvlm_lock:
            if (
                self.model is not None
                or self._smolvlm_loading
                or self._smolvlm_load_failed
            ):
                return
            self._smolvlm_loading = True

        thread = threading.Thread(
            target=self._load_smolvlm_worker,
            name="smolvlm-loader",
            daemon=True,
        )
        thread.start()

    def _load_smolvlm_worker(self) -> None:
        try:
            self._load_smolvlm()
        except Exception as error:
            with self._smolvlm_lock:
                self._smolvlm_load_failed = True
                self._smolvlm_loading = False
            print(f"SmolVLM detector disabled: cannot load {SCAN_SMOLVLM_MODEL}: {error}")
            return

        with self._smolvlm_lock:
            self._smolvlm_loading = False
        print(f"SmolVLM detector loaded: {SCAN_SMOLVLM_MODEL}")

    def _classify_objects(self, frame, result: ScanResult) -> ScanResult:
        if not result.objects:
            return ScanResult(
                objects=self._limit_single_instance_classes(result.objects),
                process_ms=result.process_ms,
            )

        classified = []
        vlm_used = 0
        for obj in result.objects:
            classification = self._classify_object(frame, obj)
            clip_detection = self._detection_from_classification(obj, classification)
            if self._is_vlm_excluded(obj, clip_detection):
                classified.append(clip_detection or obj)
                continue

            if clip_detection is not None and not (
                SCAN_VLM_CLASSIFICATION_ENABLED
                and self._should_use_vlm(classification)
            ):
                classified.append(clip_detection)
            else:
                vlm_classification = None
                if vlm_used < SCAN_VLM_CLASSIFICATION_MAX_OBJECTS:
                    vlm_classification = self._classify_object_with_vlm(
                        frame=frame,
                        obj=obj,
                    )
                    if vlm_classification is not None:
                        vlm_used += 1

                if vlm_classification is not None and vlm_classification.label != "background":
                    classified.append(
                        self._detection_with_class_name(obj, vlm_classification.label)
                    )
                elif clip_detection is not None:
                    classified.append(clip_detection)
                else:
                    classified.append(obj)

        return ScanResult(
            objects=self._limit_single_instance_classes(classified),
            process_ms=result.process_ms,
        )

    def _classify_object(self, frame, obj):
        if self.clip_classifier is None:
            return None

        key = self._clip_key(obj)
        cached = self._clip_cache.get(key)
        if cached is not None:
            return cached

        crop = self._object_crop(frame, obj)
        classification = self.clip_classifier.classify(crop)
        self._clip_cache[key] = classification
        return classification

    def _classify_object_with_vlm(self, frame, obj):
        if not SCAN_VLM_CLASSIFICATION_ENABLED:
            return None

        if self.vlm_classifier is None:
            self._ensure_vlm_classifier_loading()
            return None

        key = self._clip_key(obj)
        cached = self._vlm_cache.get(key)
        if cached is not None:
            return cached

        crop = self._object_crop(frame, obj)
        classification = self.vlm_classifier.classify(crop, cache_key=key)
        self._vlm_cache[key] = classification
        return classification

    @classmethod
    def _detection_from_classification(cls, obj, classification):
        if (
            classification is None
            or classification.is_negative
            or classification.confidence < SCAN_CLIP_MIN_CONFIDENCE
        ):
            return None

        return cls._detection_with_class_name(obj, classification.label)

    @staticmethod
    def _detection_with_class_name(obj, class_name):
        return Detection(
            class_name=class_name,
            confidence=obj.confidence,
            bbox=obj.bbox,
            roi_name=obj.roi_name,
            track_id=obj.track_id,
            polygon=obj.polygon,
        )

    @classmethod
    def _is_vlm_excluded(cls, obj, clip_detection):
        class_names = [obj.class_name]
        if clip_detection is not None:
            class_names.append(clip_detection.class_name)

        return any(
            cls._normalized_class_name(class_name) in SINGLE_INSTANCE_CLASSES
            for class_name in class_names
        )

    def _ensure_vlm_classifier_loading(self) -> None:
        if self._vlm_load_failed or self._vlm_loading:
            return

        with self._vlm_lock:
            if (
                self.vlm_classifier is not None
                or self._vlm_loading
                or self._vlm_load_failed
            ):
                return
            self._vlm_loading = True

        thread = threading.Thread(
            target=self._load_vlm_classifier_worker,
            name="vlm-crop-loader",
            daemon=True,
        )
        thread.start()

    def _load_vlm_classifier_worker(self) -> None:
        try:
            classifier = VlmCropClassifier(
                model_name=SCAN_VLM_CLASSIFICATION_MODEL,
                prompt=SCAN_VLM_CLASSIFICATION_PROMPT,
                device=SCAN_DEVICE,
                interval_seconds=SCAN_VLM_CLASSIFICATION_INTERVAL_SECONDS,
                local_files_only=SCAN_VLM_CLASSIFICATION_LOCAL_FILES_ONLY,
            )
        except Exception as error:
            with self._vlm_lock:
                self._vlm_load_failed = True
                self._vlm_loading = False
            print(
                "VLM crop classifier disabled: "
                f"cannot load {SCAN_VLM_CLASSIFICATION_MODEL}: {error}"
            )
            return

        with self._vlm_lock:
            self.vlm_classifier = classifier
            self._vlm_loading = False
        print(f"VLM crop classifier loaded: {SCAN_VLM_CLASSIFICATION_MODEL}")

    @staticmethod
    def _should_use_vlm(classification):
        if SCAN_VLM_CLASSIFICATION_MODE == "all":
            return True
        if classification.is_negative:
            return True
        return classification.confidence < SCAN_VLM_CLASSIFICATION_MIN_CLIP_CONFIDENCE

    @staticmethod
    def _object_crop(frame, obj):
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = obj.bbox
        x1 = max(0, min(width, x1))
        x2 = max(0, min(width, x2))
        y1 = max(0, min(height, y1))
        y2 = max(0, min(height, y2))
        return frame[y1:y2, x1:x2]

    @staticmethod
    def _clip_key(obj):
        if obj.track_id is not None:
            return f"id:{obj.track_id}"
        x1, y1, x2, y2 = obj.bbox
        return (
            f"{obj.class_name}:"
            f"{int(((x1 + x2) / 2) // 40)}:"
            f"{int(((y1 + y2) / 2) // 40)}"
        )

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
    def _limit_single_instance_classes(cls, objects):
        best_by_class = {}
        filtered = []

        for obj in objects:
            normalized_class = cls._normalized_class_name(obj.class_name)
            if normalized_class not in SINGLE_INSTANCE_CLASSES:
                filtered.append(cls._avoid_reserved_track_id(obj))
                continue

            current = best_by_class.get(normalized_class)
            if current is None or obj.confidence > current.confidence:
                best_by_class[normalized_class] = cls._with_single_instance_track_id(obj)

        filtered.extend(best_by_class.values())
        return filtered

    @staticmethod
    def _with_single_instance_track_id(obj):
        return Detection(
            class_name=obj.class_name,
            confidence=obj.confidence,
            bbox=obj.bbox,
            roi_name=obj.roi_name,
            track_id=SCANNER_TRACK_ID,
            polygon=obj.polygon,
        )

    @staticmethod
    def _avoid_reserved_track_id(obj):
        if obj.track_id != SCANNER_TRACK_ID:
            return obj

        return Detection(
            class_name=obj.class_name,
            confidence=obj.confidence,
            bbox=obj.bbox,
            roi_name=obj.roi_name,
            track_id=obj.track_id + RESERVED_TRACK_ID_OFFSET,
            polygon=obj.polygon,
        )

    @staticmethod
    def _normalized_class_name(class_name):
        return str(class_name).strip().lower().replace(" ", "_")

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

    def _model_label(self, label_id):
        id2label = getattr(self.model.config, "id2label", {})
        label = id2label.get(label_id, str(label_id))
        return str(label).strip().lower().replace(" ", "_")

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
