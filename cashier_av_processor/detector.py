from __future__ import annotations

import json
import logging
import multiprocessing
import queue
import statistics
import time
from pathlib import Path
from typing import Any, Optional

from .db import CvEvent, DatabaseClient, DurableEventStore
from .messages import InferenceFrame

logger = logging.getLogger(__name__)


ALLOWED_EVENT_TYPES = {
    "item_in_bag",
    "hand_to_drawer",
    "phone_scanned_by_cashier",
    "document_presented",
    "item_return",
    "hand_to_scanner",
    "customer_present",
    "customer_left",
    "no_cashier",
    "cashier_present",
}

EVENT_COOLDOWN_MS = {
    "item_in_bag": 5000,
    "hand_to_drawer": 4000,
    "phone_scanned_by_cashier": 4000,
    "document_presented": 7000,
    "item_return": 5000,
    "hand_to_scanner": 4000,
    "cashier_present": 20000,
    "no_cashier": 20000,
}


class YoloDetector:
    def __init__(
        self,
        model_path: str,
        db_pool: DatabaseClient,
        spool_dir: Path | str = "spool",
        weights_version: Optional[str] = None,
        roi_reload_interval_s: float = 60.0,
        device: str = "cpu",
    ) -> None:
        from ultralytics import YOLO

        self.device = device
        self.model = YOLO(model_path)
        self.model.to(device)
        self.model_path = model_path
        self.model_name = "yolov11-cashier"
        self.weights_version = weights_version or Path(model_path).name
        self.db_pool = db_pool
        self.store = DurableEventStore(db_pool, Path(spool_dir))
        self.roi_polygons: dict[str, dict[str, Any]] = {}
        self.camera_pos: dict[str, str] = {}
        self.camera_state: dict[str, dict[str, bool]] = {}
        self.last_event_at_ms: dict[tuple[str, str, str], int] = {}
        self.roi_reload_interval_s = roi_reload_interval_s
        self.next_roi_reload_at = 0.0
        self.load_roi_polygons()

    def load_roi_polygons(self) -> None:
        from shapely.geometry import Polygon

        try:
            rows = self.store.load_active_cameras()
        except Exception:
            logger.exception("Failed to load ROI polygons from DB; will retry")
            self.next_roi_reload_at = time.monotonic() + self.roi_reload_interval_s
            return

        polygons_by_camera: dict[str, dict[str, Any]] = {}
        pos_by_camera: dict[str, str] = {}

        for camera_id, roi_config, pos_id in rows:
            config = self._coerce_roi_config(roi_config)
            camera_polygons: dict[str, Any] = {}
            for name, coords in self._iter_roi_entries(config):
                if not coords:
                    continue
                try:
                    camera_polygons[name] = Polygon(coords)
                except Exception:
                    logger.exception("Invalid ROI polygon camera=%s zone=%s", camera_id, name)

            polygons_by_camera[str(camera_id)] = camera_polygons
            pos_by_camera[str(camera_id)] = str(pos_id)

        self.roi_polygons = polygons_by_camera
        self.camera_pos = pos_by_camera
        self.next_roi_reload_at = time.monotonic() + self.roi_reload_interval_s
        logger.info("Loaded ROI polygons for %s active cameras", len(self.roi_polygons))

    def run_detection(self, frame, camera_id: str, timestamp_ms: int) -> float:
        """Run YOLOv11, map boxes to Shapely ROI polygons, and persist cv_events."""
        from shapely.geometry import Point

        if time.monotonic() >= self.next_roi_reload_at:
            self.load_roi_polygons()

        start = time.perf_counter()
        results = self.model(frame, verbose=False)
        inference_time_ms = (time.perf_counter() - start) * 1000.0
        polygons = self.roi_polygons.get(camera_id, {})
        customer_detected = False
        cashier_detected = False

        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue

            names = getattr(result, "names", {})
            for box in boxes:
                xyxy = self._box_xyxy(box)
                if xyxy is None:
                    continue

                x1, y1, x2, y2 = xyxy
                confidence = self._box_confidence(box)
                class_id = self._box_class_id(box)
                class_name = str(names.get(class_id, class_id))
                normalized_class = self._normalize_token(class_name)
                track_id = self._box_track_id(box)
                point = Point((x1 + x2) / 2.0, y2)
                bbox_payload = {
                    "bbox": [x1, y1, x2, y2],
                    "class_id": class_id,
                    "class_name": class_name,
                    "track_id": track_id,
                    "frame_id": timestamp_ms,
                }

                for zone_name, polygon in polygons.items():
                    if not polygon.intersects(point):
                        continue

                    normalized_zone = self._normalize_token(zone_name)
                    if self._is_customer(normalized_class) and self._is_customer_zone(normalized_zone):
                        customer_detected = True
                    if self._is_cashier(normalized_class) and self._is_cashier_zone(normalized_zone):
                        cashier_detected = True

                    event_type = self._event_type_for(normalized_class, normalized_zone)
                    if event_type is None:
                        continue

                    object_key = track_id or self._object_key(normalized_class, x1, y1, x2, y2)
                    if not self._should_emit(camera_id, event_type, object_key, timestamp_ms):
                        continue

                    event = CvEvent(
                        camera_id=camera_id,
                        event_type=event_type,
                        timestamp_ms=timestamp_ms,
                        confidence=confidence,
                        model_name=self.model_name,
                        weights_version=self.weights_version,
                        inference_time_ms=inference_time_ms,
                        bbox_jsonb={**bbox_payload, "roi": zone_name},
                        snapshot_path=None,
                    )
                    self.store.insert_cv_event(event)

        self._emit_presence_state_events(
            camera_id=camera_id,
            timestamp_ms=timestamp_ms,
            inference_time_ms=inference_time_ms,
            customer_detected=customer_detected,
            cashier_detected=cashier_detected,
            has_customer_zone=any(self._is_customer_zone(self._normalize_token(zone)) for zone in polygons),
            has_cashier_zone=any(self._is_cashier_zone(self._normalize_token(zone)) for zone in polygons),
        )
        self.store.replay_pending(max_events=25)
        return inference_time_ms

    @staticmethod
    def _iter_roi_entries(config: dict[str, Any]):
        for name, coords in config.items():
            if isinstance(coords, list) and len(coords) >= 3:
                yield str(name), coords

    @staticmethod
    def _coerce_roi_config(raw: Any) -> dict[str, Any]:
        if raw is None:
            return {}
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, str):
            return json.loads(raw)
        return dict(raw)

    def _event_type_for(self, normalized_class: str, normalized_zone: str) -> Optional[str]:
        if normalized_class == "hand" and normalized_zone in {"cash_drawer_zone", "drawer_zone"}:
            return "hand_to_drawer"
        if self._is_item(normalized_class) and normalized_zone in {
            "bag_zone",
            "bagging_zone",
            "packing_zone",
        }:
            return "item_in_bag"
        if self._is_phone(normalized_class) and normalized_zone in {
            "scanner_zone",
            "cashier_zone",
            "cashier_workplace_zone",
        }:
            return "phone_scanned_by_cashier"
        if self._is_document(normalized_class) and normalized_zone in {
            "scanner_zone",
            "customer_zone",
            "document_zone",
        }:
            return "document_presented"
        if self._is_item(normalized_class) and normalized_zone in {
            "return_zone",
            "scanner_zone",
            "cashier_zone",
        }:
            return "item_return"
        if normalized_class == "hand" and normalized_zone in {
            "scanner_zone",
            "cashier_zone",
            "cashier_workplace_zone",
        }:
            return "hand_to_scanner"
        return None

    def _emit_presence_state_events(
        self,
        camera_id: str,
        timestamp_ms: int,
        inference_time_ms: float,
        customer_detected: bool,
        cashier_detected: bool,
        has_customer_zone: bool,
        has_cashier_zone: bool,
    ) -> None:
        state = self.camera_state.setdefault(
            camera_id,
            {"customer_present": False, "cashier_present": False},
        )

        if has_customer_zone and customer_detected != state["customer_present"]:
            state["customer_present"] = customer_detected
            self._insert_state_event(
                camera_id,
                "customer_present" if customer_detected else "customer_left",
                timestamp_ms,
                inference_time_ms,
            )

        if has_cashier_zone:
            if cashier_detected != state["cashier_present"]:
                state["cashier_present"] = cashier_detected
                self._insert_state_event(
                    camera_id,
                    "cashier_present" if cashier_detected else "no_cashier",
                    timestamp_ms,
                    inference_time_ms,
                )
            elif cashier_detected:
                self._insert_state_event(
                    camera_id,
                    "cashier_present",
                    timestamp_ms,
                    inference_time_ms,
                    heartbeat=True,
                )
            else:
                self._insert_state_event(
                    camera_id,
                    "no_cashier",
                    timestamp_ms,
                    inference_time_ms,
                    heartbeat=True,
                )

    def _insert_state_event(
        self,
        camera_id: str,
        event_type: str,
        timestamp_ms: int,
        inference_time_ms: float,
        heartbeat: bool = False,
    ) -> None:
        if heartbeat and not self._should_emit(camera_id, event_type, "state", timestamp_ms):
            return
        if event_type not in ALLOWED_EVENT_TYPES:
            return
        if not heartbeat:
            self.last_event_at_ms[(camera_id, event_type, "state")] = timestamp_ms
        self.store.insert_cv_event(
            CvEvent(
                camera_id=camera_id,
                event_type=event_type,
                timestamp_ms=timestamp_ms,
                confidence=1.0,
                model_name=self.model_name,
                weights_version=self.weights_version,
                inference_time_ms=inference_time_ms,
                bbox_jsonb={"roi": None, "class_name": None, "state": event_type},
                snapshot_path=None,
            )
        )

    def _should_emit(
        self,
        camera_id: str,
        event_type: str,
        object_key: str,
        timestamp_ms: int,
    ) -> bool:
        if event_type not in ALLOWED_EVENT_TYPES:
            return False
        cooldown_ms = EVENT_COOLDOWN_MS.get(event_type, 0)
        key = (camera_id, event_type, object_key)
        last_timestamp_ms = self.last_event_at_ms.get(key)
        if last_timestamp_ms is not None and timestamp_ms - last_timestamp_ms < cooldown_ms:
            return False
        self.last_event_at_ms[key] = timestamp_ms
        return True

    @staticmethod
    def _normalize_token(value: str) -> str:
        return value.strip().lower().replace(" ", "_").replace("-", "_")

    @staticmethod
    def _is_customer(normalized_class: str) -> bool:
        return normalized_class in {"customer", "person", "client", "buyer"}

    @staticmethod
    def _is_cashier(normalized_class: str) -> bool:
        return normalized_class in {"cashier", "person", "employee", "operator"}

    @staticmethod
    def _is_item(normalized_class: str) -> bool:
        return normalized_class in {"item", "product", "goods", "object", "package"}

    @staticmethod
    def _is_phone(normalized_class: str) -> bool:
        return normalized_class in {"phone", "cell_phone", "mobile_phone", "qr", "qr_code"}

    @staticmethod
    def _is_document(normalized_class: str) -> bool:
        return normalized_class in {"document", "passport", "id_card", "id", "license"}

    @staticmethod
    def _is_customer_zone(normalized_zone: str) -> bool:
        return normalized_zone in {"customer_zone", "client_zone", "service_zone"}

    @staticmethod
    def _is_cashier_zone(normalized_zone: str) -> bool:
        return normalized_zone in {
            "cashier_zone",
            "cashier_workplace_zone",
            "workplace_zone",
            "operator_zone",
        }

    @staticmethod
    def _object_key(normalized_class: str, x1: float, y1: float, x2: float, y2: float) -> str:
        cx = int(round((x1 + x2) / 20.0))
        cy = int(round((y1 + y2) / 20.0))
        return f"{normalized_class}:{cx}:{cy}"

    @staticmethod
    def _box_xyxy(box: Any) -> Optional[tuple[float, float, float, float]]:
        try:
            raw = box.xyxy[0].detach().cpu().tolist()
        except AttributeError:
            raw = box.xyxy[0].tolist()
        except Exception:
            logger.exception("Failed to read YOLO bbox")
            return None
        return float(raw[0]), float(raw[1]), float(raw[2]), float(raw[3])

    @staticmethod
    def _box_confidence(box: Any) -> float:
        try:
            return float(box.conf[0].detach().cpu().item())
        except AttributeError:
            return float(box.conf[0])

    @staticmethod
    def _box_class_id(box: Any) -> int:
        try:
            return int(box.cls[0].detach().cpu().item())
        except AttributeError:
            return int(box.cls[0])

    @staticmethod
    def _box_track_id(box: Any) -> Optional[str]:
        raw_id = getattr(box, "id", None)
        if raw_id is None:
            return None
        try:
            value = raw_id[0].detach().cpu().item()
        except AttributeError:
            value = raw_id[0]
        except Exception:
            return None
        return str(int(value))


class YoloInferenceProcess(multiprocessing.Process):
    def __init__(
        self,
        inference_queue: multiprocessing.Queue,
        db_dsn: str,
        stop_event: multiprocessing.Event,
        model_path: str = "yolov11x.pt",
        device: str = "cpu",
        spool_dir: Path | str = "spool",
        profile_interval_s: float = 10.0,
    ) -> None:
        super().__init__(name="YoloInferenceProcess")
        self.inference_queue = inference_queue
        self.db_dsn = db_dsn
        self.stop_event = stop_event
        self.model_path = model_path
        self.device = device
        self.spool_dir = Path(spool_dir)
        self.profile_interval_s = profile_interval_s

    def run(self) -> None:
        import cv2
        import numpy as np

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(processName)s %(name)s: %(message)s",
        )
        db = DatabaseClient(self.db_dsn)
        detector = YoloDetector(self.model_path, db, self.spool_dir, device=self.device)
        inference_times: list[float] = []
        next_profile_at = time.monotonic() + self.profile_interval_s

        try:
            while not self.stop_event.is_set():
                try:
                    message = self.inference_queue.get(timeout=0.5)
                except queue.Empty:
                    continue

                if not isinstance(message, InferenceFrame):
                    logger.warning("Ignoring unexpected inference payload: %r", message)
                    continue

                frame = cv2.imdecode(
                    np.frombuffer(message.jpeg_bytes, np.uint8),
                    cv2.IMREAD_COLOR,
                )
                if frame is None:
                    logger.warning("Failed to decode inference frame camera=%s", message.camera_id)
                    continue

                try:
                    elapsed_ms = detector.run_detection(
                        frame=frame,
                        camera_id=message.camera_id,
                        timestamp_ms=message.timestamp_ms,
                    )
                    inference_times.append(elapsed_ms)
                except Exception:
                    logger.exception("YOLO inference failed camera=%s", message.camera_id)

                now = time.monotonic()
                if now >= next_profile_at:
                    self._log_profile(inference_times, self.device)
                    inference_times.clear()
                    next_profile_at = now + self.profile_interval_s
        finally:
            db.close()

    @staticmethod
    def _log_profile(inference_times: list[float], device: str) -> None:
        if inference_times:
            avg = statistics.fmean(inference_times)
        else:
            avg = 0.0

        if not device.startswith("cuda"):
            logger.info("YOLO profile avg_inference_time_ms=%.2f device=%s", avg, device)
            return

        gpu_info = "unavailable"
        try:
            import torch

            if torch.cuda.is_available():
                allocated_mb = torch.cuda.memory_allocated() / (1024 * 1024)
                reserved_mb = torch.cuda.memory_reserved() / (1024 * 1024)
                gpu_info = f"allocated_mb={allocated_mb:.1f} reserved_mb={reserved_mb:.1f}"
        except Exception:
            logger.debug("GPU memory profiling unavailable", exc_info=True)

        logger.info("YOLO profile avg_inference_time_ms=%.2f device=%s gpu=%s", avg, device, gpu_info)
