import time

from ultralytics import YOLO

from config import (
    POSE_MODEL_PATH,
    PERSON_CONFIDENCE,
    POSE_IMAGE_SIZE,
)

from models.person import PersonDetection, PersonResult
from vision.roi import crop_roi, offset_bbox


class PersonDetector:
    def __init__(
        self,
        customer_roi,
        cashier_roi,
    ):
        self.customer_roi = customer_roi
        self.cashier_roi = cashier_roi

        self.model = YOLO(POSE_MODEL_PATH)

    def detect(self, frame) -> PersonResult:
        customer_persons, customer_ms = self._detect_in_roi(
            frame=frame,
            roi=self.customer_roi,
            role="customer",
        )

        cashier_persons, cashier_ms = self._detect_in_roi(
            frame=frame,
            roi=self.cashier_roi,
            role="cashier",
        )

        persons = customer_persons + cashier_persons

        return PersonResult(
            customer_detected=len(customer_persons) > 0,
            cashier_detected=len(cashier_persons) > 0,
            persons=persons,
            customer_ms=customer_ms,
            cashier_ms=cashier_ms,
        )

    def _detect_in_roi(
        self,
        frame,
        roi,
        role: str,
    ):
        roi_frame, clipped_roi = crop_roi(frame, roi)

        x1, y1, _, _ = clipped_roi

        if roi_frame.size == 0:
            return [], 0

        start = time.time()

        results = self.model.predict(
            source=roi_frame,
            conf=PERSON_CONFIDENCE,
            imgsz=POSE_IMAGE_SIZE,
            verbose=False,
        )

        process_ms = int((time.time() - start) * 1000)

        persons = []

        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                cls = int(box.cls.item())
                conf = float(box.conf.item())

                class_name = self.model.names[cls]

                if class_name != "person":
                    continue

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

                persons.append(
                    PersonDetection(
                        role=role,
                        confidence=conf,
                        bbox=full_bbox,
                    )
                )

        return persons, process_ms