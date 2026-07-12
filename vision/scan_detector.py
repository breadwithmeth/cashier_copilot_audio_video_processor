import time

from ultralytics import YOLOWorld

from config import (
    SCAN_MODEL_PATH,
    SCAN_CONFIDENCE,
    SCAN_IMAGE_SIZE,
    SCAN_WORLD_PROMPTS,
)

from models.detection import Detection, ScanResult
from vision.roi import crop_roi, offset_bbox


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

            for box in result.boxes:
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

                full_bbox = offset_bbox(
                    local_bbox,
                    x1,
                    y1,
                )

                objects.append(
                    Detection(
                        class_name="object",
                        confidence=conf,
                        bbox=full_bbox,
                        roi_name="scan_zone",
                        track_id=track_id,
                    )
                )

        return ScanResult(
            objects=objects,
            process_ms=process_ms,
        )
