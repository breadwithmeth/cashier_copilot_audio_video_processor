import time

from ultralytics import YOLO

from config import (
    SCAN_MODEL_PATH,
    SCAN_CONFIDENCE,
    SCAN_IMAGE_SIZE,
)

from models.detection import Detection, ScanResult
from vision.roi import crop_roi, offset_bbox


class ScanDetector:
    def __init__(self, roi):
        self.roi = roi
        self.model = YOLO(str(SCAN_MODEL_PATH))

    def detect(self, frame) -> ScanResult:
        roi_frame, clipped_roi = crop_roi(frame, self.roi)

        x1, y1, _, _ = clipped_roi

        if roi_frame.size == 0:
            return ScanResult(objects=[], process_ms=0)

        start = time.time()

        results = self.model.predict(
            source=roi_frame,
            conf=SCAN_CONFIDENCE,
            imgsz=SCAN_IMAGE_SIZE,
            verbose=False,
        )

        process_ms = int((time.time() - start) * 1000)

        objects = []

        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                cls = int(box.cls.item())
                conf = float(box.conf.item())

                bx1, by1, bx2, by2 = box.xyxy[0].tolist()

                local_bbox = (
                    int(bx1),
                    int(by1),
                    int(bx2),
                    int(by2),
                )

                full_bbox = offset_bbox(
                    local_bbox,
                    x1,
                    y1,
                )

                class_name = self.model.names[cls]

                objects.append(
                    Detection(
                        class_name=class_name,
                        confidence=conf,
                        bbox=full_bbox,
                        roi_name="scan_zone",
                    )
                )

        return ScanResult(
            objects=objects,
            process_ms=process_ms,
        )