import time

from ultralytics import YOLO

from config import (
    POSE_MODEL_PATH,
    PERSON_CONFIDENCE,
    POSE_IMAGE_SIZE,
    POSE_KEYPOINT_CONFIDENCE,
)

from models.person import HandPose, PersonDetection, PersonResult
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
        persons, process_ms = self._detect_persons(frame)
        customer_detected = any(person.role == "customer" for person in persons)
        cashier_detected = any(person.role == "cashier" for person in persons)

        return PersonResult(
            customer_detected=customer_detected,
            cashier_detected=cashier_detected,
            persons=persons,
            process_ms=process_ms,
            customer_ms=process_ms if customer_detected else 0,
            cashier_ms=process_ms if cashier_detected else 0,
        )

    def _detect_persons(self, frame):
        roi_frame, clipped_roi = crop_roi(frame, self._combined_roi())

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

            for person_index, box in enumerate(result.boxes):
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

                hands = self._extract_hands(
                    result,
                    person_index,
                    offset_x=x1,
                    offset_y=y1,
                )

                for role in self._roles_for_bbox(full_bbox):
                    persons.append(
                        PersonDetection(
                            role=role,
                            confidence=conf,
                            bbox=full_bbox,
                            hands=hands,
                        )
                    )

        return persons, process_ms

    def _extract_hands(self, result, person_index, offset_x, offset_y):
        if result.keypoints is None or result.keypoints.xy is None:
            return []

        xy = result.keypoints.xy[person_index]
        confidence = result.keypoints.conf
        confidence = confidence[person_index] if confidence is not None else None

        # COCO pose indices: shoulders 5/6, elbows 7/8, wrists 9/10.
        hands = []
        for side, indices in (("left", (5, 7, 9)), ("right", (6, 8, 10))):
            points = []
            scores = []
            for index in indices:
                score = float(confidence[index].item()) if confidence is not None else 1.0
                scores.append(score)
                if score < POSE_KEYPOINT_CONFIDENCE:
                    points.append(None)
                    continue
                px, py = xy[index].tolist()
                points.append((int(px + offset_x), int(py + offset_y)))

            shoulder, elbow, wrist = points
            position = self._classify_hand(shoulder, elbow, wrist)
            hands.append(HandPose(
                side=side,
                position=position,
                shoulder=shoulder,
                elbow=elbow,
                wrist=wrist,
                confidence=min(scores),
            ))
        return hands

    @staticmethod
    def _classify_hand(shoulder, elbow, wrist):
        if shoulder is None or wrist is None:
            return "unknown"

        shoulder_x, shoulder_y = shoulder
        wrist_x, wrist_y = wrist
        arm_scale = max(30, abs(wrist_y - shoulder_y))

        if wrist_y < shoulder_y:
            return "raised"

        if abs(wrist_x - shoulder_x) > arm_scale * 0.75:
            return "extended"

        if elbow is not None and wrist_y < elbow[1]:
            return "bent"

        return "down"

    def _combined_roi(self):
        customer_x1, customer_y1, customer_x2, customer_y2 = self.customer_roi
        cashier_x1, cashier_y1, cashier_x2, cashier_y2 = self.cashier_roi

        return (
            min(customer_x1, cashier_x1),
            min(customer_y1, cashier_y1),
            max(customer_x2, cashier_x2),
            max(customer_y2, cashier_y2),
        )

    def _roles_for_bbox(self, bbox):
        x1, _, x2, y2 = bbox
        bottom_center = (
            int((x1 + x2) / 2),
            y2,
        )

        roles = []

        if self._point_in_roi(bottom_center, self.customer_roi):
            roles.append("customer")

        if self._point_in_roi(bottom_center, self.cashier_roi):
            roles.append("cashier")

        return roles

    @staticmethod
    def _point_in_roi(point, roi):
        x, y = point
        x1, y1, x2, y2 = roi

        return x1 <= x <= x2 and y1 <= y <= y2
