import time

import cv2
import numpy as np
from ultralytics import YOLOWorld

from config import (
    SCAN_MODEL_PATH,
    SCAN_CONFIDENCE,
    SCAN_IMAGE_SIZE,
    SCAN_WORLD_PROMPTS,
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
        self.model = YOLOWorld(str(SCAN_MODEL_PATH))
        self.model.set_classes(SCAN_WORLD_PROMPTS)

    def detect(self, frame) -> ScanResult:
        roi_frame, clipped_roi = crop_roi(frame, self.roi)

        x1, y1, _, _ = clipped_roi

        if roi_frame.size == 0:
            return ScanResult(objects=[], process_ms=0)

        start = time.time()

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
