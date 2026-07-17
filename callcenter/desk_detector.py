import time
import cv2
from ultralytics import YOLO

from config import (
    DESK_MODEL_PATH,
    DESK_CONFIDENCE,
    DESK_IMAGE_SIZE,
    DESK_KEYPOINT_CONFIDENCE,
)

from models.person import HandPose, PersonDetection, PersonResult
from vision.person_action import SkeletonActionBuffer
from vision.roi import crop_roi, expand_roi, offset_bbox, scale_roi


UPPER_BODY_RATIO = 0.55


class DeskDetector:
    def __init__(
        self,
        table_roi,
    ):
        self.table_roi = table_roi

        self.model = YOLO(DESK_MODEL_PATH)
        self.action_buffer = SkeletonActionBuffer()

    def detect(self, frame) -> PersonResult:
        persons, process_ms = self._detect_persons(frame)
        agent_detected = any(person.role == "agent" for person in persons)
        customer_detected = any(person.role == "customer" for person in persons)

        return PersonResult(
            customer_detected=customer_detected,
            cashier_detected=agent_detected,  # Reusing cashier_detected for agent
            persons=persons,
            process_ms=process_ms,
            customer_ms=process_ms if customer_detected else 0,
            cashier_ms=process_ms if agent_detected else 0,
        )

    def _detect_persons(self, frame):
        table_roi = scale_roi(self.table_roi, frame)
        expanded_roi = expand_roi(table_roi, frame, margin=0.08)
        roi_frame, clipped_roi = crop_roi(frame, expanded_roi)

        x1, y1, _, _ = clipped_roi

        if roi_frame.size == 0:
            return [], 0

        start = time.time()

        results = self.model.track(
            source=roi_frame,
            conf=DESK_CONFIDENCE,
            imgsz=DESK_IMAGE_SIZE,
            persist=True,
            tracker="trackers/bytetrack_retail.yaml",
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
                track_id = (
                    int(box.id.item())
                    if box.id is not None
                    else None
                )

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
                upper_body_bbox = self._upper_body_bbox(full_bbox)

                hands = self._extract_hands(
                    result,
                    person_index,
                    offset_x=x1,
                    offset_y=y1,
                )
                keypoints = self._extract_keypoints(
                    result,
                    person_index,
                    offset_x=x1,
                    offset_y=y1,
                )
                action = self.action_buffer.update(
                    track_id=track_id,
                    bbox=upper_body_bbox,
                    keypoints=keypoints,
                )

                for role in self._roles_for_bbox(
                    upper_body_bbox,
                    table_roi=table_roi,
                ):
                    persons.append(
                        PersonDetection(
                            role=role,
                            confidence=conf,
                            bbox=upper_body_bbox,
                            hands=hands,
                            track_id=track_id,
                            keypoints=keypoints,
                            action=action.label,
                            action_confidence=action.confidence,
                        )
                    )

        self.action_buffer.prune()
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
                if score < DESK_KEYPOINT_CONFIDENCE:
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

    def _extract_keypoints(self, result, person_index, offset_x, offset_y):
        if result.keypoints is None or result.keypoints.xy is None:
            return None

        xy = result.keypoints.xy[person_index]
        confidence = result.keypoints.conf
        confidence = confidence[person_index] if confidence is not None else None

        keypoints = []
        for index, point in enumerate(xy):
            score = float(confidence[index].item()) if confidence is not None else 1.0
            if score < DESK_KEYPOINT_CONFIDENCE:
                keypoints.append(None)
                continue

            px, py = point.tolist()
            keypoints.append((int(px + offset_x), int(py + offset_y), score))

        return keypoints

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

    def _roles_for_bbox(self, bbox, table_roi):
        x1, y1, x2, y2 = bbox
        upper_body_center = (
            int((x1 + x2) / 2),
            int((y1 + y2) / 2),
        )

        roles = []

        if self._point_in_roi(upper_body_center, table_roi):
            roles.append("agent")
            roles.append("customer")

        return roles

    @staticmethod
    def _upper_body_bbox(bbox):
        x1, y1, x2, y2 = bbox
        height = max(0, y2 - y1)
        upper_y2 = y1 + int(height * UPPER_BODY_RATIO)

        return x1, y1, x2, upper_y2

    @staticmethod
    def _point_in_roi(point, roi):
        x, y = point
        x1, y1, x2, y2 = roi

        return x1 <= x <= x2 and y1 <= y <= y2
