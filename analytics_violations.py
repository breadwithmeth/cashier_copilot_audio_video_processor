from __future__ import annotations

import json
import queue
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime

from config import (
    ANALYTICS_API_BASE_URL,
    ANALYTICS_API_KEY,
    ANALYTICS_REGISTER_CODE,
    ANALYTICS_SEND_TIMEOUT,
    ANALYTICS_STORE_CODE,
    ANALYTICS_VIOLATION_EVENTS_PATH,
    CASHIER_ABSENT_DURING_SERVICE_SECONDS,
    CUSTOMER_WAITING_VIOLATION_SECONDS,
    OBJECT_LEFT_IN_SCAN_ZONE_SECONDS,
    VIOLATION_EVENT_COOLDOWN_SECONDS,
)


def send_violation_event(
    *,
    camera_name: str,
    code: str,
    severity: str,
    message: str,
    started_at: float | str,
    ended_at: float | str,
    duration_seconds: float,
    payload: dict,
    source: str,
    correlation_id: str | None = None,
) -> dict:
    if not ANALYTICS_API_BASE_URL or not ANALYTICS_API_KEY:
        print(f"[{camera_name}] Violation event skipped: analytics API is not configured")
        return {"status": "skipped", "reason": "analytics_api_not_configured"}

    started_at_iso = _iso_datetime(started_at)
    ended_at_iso = _iso_datetime(ended_at)
    occurred_at_ms = int(_timestamp_seconds(ended_at) * 1000)
    id_parts = [camera_name, code.lower(), str(occurred_at_ms)]
    if correlation_id:
        id_parts.insert(2, correlation_id[:12])
    external_event_id = "-".join(id_parts)
    request_payload = {
        "externalEventId": external_event_id,
        "idempotencyKey": external_event_id,
        "cameraCode": camera_name,
        "eventType": code,
        "source": source,
        "occurredAt": ended_at_iso,
        "startedAt": started_at_iso,
        "endedAt": ended_at_iso,
        "severity": severity,
        "message": message,
        "payload": {
            **payload,
            "durationSeconds": round(duration_seconds, 3),
        },
    }
    if correlation_id:
        request_payload["correlationId"] = correlation_id
        request_payload["payload"]["correlationId"] = correlation_id
    if ANALYTICS_STORE_CODE:
        request_payload["storeCode"] = ANALYTICS_STORE_CODE
        request_payload["payload"]["storeCode"] = ANALYTICS_STORE_CODE
    if ANALYTICS_REGISTER_CODE:
        request_payload["registerCode"] = ANALYTICS_REGISTER_CODE

    url = f"{ANALYTICS_API_BASE_URL}{ANALYTICS_VIOLATION_EVENTS_PATH}"
    request = urllib.request.Request(
        url,
        data=json.dumps(request_payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "x-api-key": ANALYTICS_API_KEY,
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(
            request,
            timeout=ANALYTICS_SEND_TIMEOUT,
        ) as response:
            response_body = response.read().decode("utf-8", errors="replace")
        print(f"[{camera_name}] Violation event sent: {external_event_id}")
        return {
            "status": "sent",
            "endpoint": url,
            "externalEventId": external_event_id,
            "statusCode": response.status,
            "response": response_body[:1000],
        }
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        print(f"[{camera_name}] Violation event HTTP error {error.code}: {body[:300]}")
        return {
            "status": "failed",
            "endpoint": url,
            "externalEventId": external_event_id,
            "statusCode": error.code,
            "error": body[:1000],
        }
    except (OSError, urllib.error.URLError) as error:
        print(f"[{camera_name}] Violation event send error: {error}")
        return {
            "status": "failed",
            "endpoint": url,
            "externalEventId": external_event_id,
            "error": str(error),
        }


class VideoViolationMonitor:
    def __init__(
        self,
        camera_name: str,
        waiting_threshold_seconds: float = CUSTOMER_WAITING_VIOLATION_SECONDS,
        cashier_absent_threshold_seconds: float = CASHIER_ABSENT_DURING_SERVICE_SECONDS,
        object_left_threshold_seconds: float = OBJECT_LEFT_IN_SCAN_ZONE_SECONDS,
        cooldown_seconds: float = VIOLATION_EVENT_COOLDOWN_SECONDS,
    ):
        self.camera_name = camera_name
        self.waiting_threshold_seconds = waiting_threshold_seconds
        self.cashier_absent_threshold_seconds = cashier_absent_threshold_seconds
        self.object_left_threshold_seconds = object_left_threshold_seconds
        self.cooldown_seconds = cooldown_seconds
        self._active_since: dict[str, float] = {}
        self._last_sent_at: dict[str, float] = {}
        self._visit_key: str | None = None
        self._cashier_seen_in_visit = False
        self._object_seen_since: dict[str, float] = {}
        self._jobs: queue.Queue[dict | None] = queue.Queue()
        self._worker = threading.Thread(target=self._send_worker, daemon=True)
        self._worker.start()

    def evaluate(self, state) -> None:
        now = time.time()
        visit_key = self._visit_key_for_state(state)
        self._sync_visit(visit_key)
        if visit_key is None:
            return

        if state.cashier_is_present:
            self._cashier_seen_in_visit = True

        wait_seconds = state.get_customer_wait_seconds()
        customer_waiting_too_long = (
            state.customer_is_present
            and not state.cashier_is_present
            and wait_seconds >= self.waiting_threshold_seconds
        )

        self._update_rule(
            code="CUSTOMER_WAITING_TOO_LONG",
            active=customer_waiting_too_long,
            now=now,
            severity="warning",
            message="Клиент ожидает кассира больше 20 секунд",
            started_at=state.customer_present_since,
            duration_seconds=wait_seconds,
            visit_key=visit_key,
            payload={
                "visitKey": visit_key,
                "customerWaitSeconds": round(wait_seconds, 3),
                "thresholdSeconds": self.waiting_threshold_seconds,
                "cashierPresent": state.cashier_is_present,
                "customerPresent": state.customer_is_present,
                "checkoutStatus": state.status.value,
            },
        )
        self._evaluate_cashier_absent_during_service(state, now, visit_key)
        self._evaluate_objects_left_in_scan_zone(state, now, visit_key)

    def stop(self) -> None:
        self._jobs.put(None)
        self._worker.join(timeout=10)

    def _evaluate_cashier_absent_during_service(self, state, now, visit_key: str) -> None:
        absent_seconds = state.get_cashier_absent_seconds()
        active = (
            state.customer_is_present
            and self._cashier_seen_in_visit
            and not state.cashier_is_present
            and absent_seconds >= self.cashier_absent_threshold_seconds
        )
        self._update_rule(
            code="CASHIER_ABSENT_DURING_SERVICE",
            active=active,
            now=now,
            severity="critical",
            message="Кассир отсутствует во время обслуживания клиента",
            started_at=state.cashier_absent_since,
            duration_seconds=absent_seconds,
            visit_key=visit_key,
            payload={
                "visitKey": visit_key,
                "cashierAbsentSeconds": round(absent_seconds, 3),
                "thresholdSeconds": self.cashier_absent_threshold_seconds,
                "customerPresent": state.customer_is_present,
                "cashierSeenInVisit": self._cashier_seen_in_visit,
                "checkoutStatus": state.status.value,
            },
        )

    def _evaluate_objects_left_in_scan_zone(self, state, now, visit_key: str) -> None:
        current_keys = {self._object_key(obj) for obj in state.scan_objects}
        for object_key in current_keys:
            self._object_seen_since.setdefault(object_key, now)
        for object_key in list(self._object_seen_since):
            if object_key not in current_keys:
                self._object_seen_since.pop(object_key, None)
                self._active_since.pop(f"OBJECT_LEFT_IN_SCAN_ZONE:{object_key}", None)

        for object_key, seen_since in self._object_seen_since.items():
            duration_seconds = now - seen_since
            self._update_rule(
                code=f"OBJECT_LEFT_IN_SCAN_ZONE:{object_key}",
                event_code="OBJECT_LEFT_IN_SCAN_ZONE",
                active=duration_seconds >= self.object_left_threshold_seconds,
                now=now,
                severity="warning",
                message="Объект долго находится в зоне сканирования",
                started_at=seen_since,
                duration_seconds=duration_seconds,
                visit_key=visit_key,
                payload={
                    "visitKey": visit_key,
                    "objectKey": object_key,
                    "objectLeftSeconds": round(duration_seconds, 3),
                    "thresholdSeconds": self.object_left_threshold_seconds,
                    "checkoutStatus": state.status.value,
                },
            )

    def _update_rule(
        self,
        code: str,
        active: bool,
        now: float,
        severity: str,
        message: str,
        started_at: float | None,
        duration_seconds: float,
        visit_key: str,
        payload: dict,
        event_code: str | None = None,
    ) -> None:
        if not active:
            self._active_since.pop(code, None)
            return

        active_since = self._active_since.setdefault(code, started_at or now)
        sent_key = f"{visit_key}:{code}"
        last_sent_at = self._last_sent_at.get(sent_key)
        if last_sent_at is not None and now - last_sent_at < self.cooldown_seconds:
            return

        self._last_sent_at[sent_key] = now
        self._jobs.put({
            "code": event_code or code,
            "severity": severity,
            "message": message,
            "started_at": active_since,
            "ended_at": now,
            "duration_seconds": duration_seconds,
            "correlation_id": visit_key,
            "payload": payload,
        })

    def _send_worker(self) -> None:
        while True:
            event = self._jobs.get()
            if event is None:
                return
            self._send_event(event)

    def _send_event(self, event: dict) -> dict:
        return send_violation_event(
            camera_name=self.camera_name,
            code=event["code"],
            severity=event["severity"],
            message=event["message"],
            started_at=event["started_at"],
            ended_at=event["ended_at"],
            duration_seconds=event["duration_seconds"],
            payload=event["payload"],
            source="cashier_copilot_video",
            correlation_id=event.get("correlation_id"),
        )

    def _sync_visit(self, visit_key: str | None) -> None:
        if visit_key == self._visit_key:
            return
        self._visit_key = visit_key
        self._active_since.clear()
        self._object_seen_since.clear()
        self._cashier_seen_in_visit = False

    @staticmethod
    def _visit_key_for_state(state) -> str | None:
        if state.customer_left or not state.customer_is_present:
            return None
        started_at = state.visit_started_at or state.customer_present_since
        if started_at is None:
            return None
        return f"visit-{int(started_at * 1000)}"

    @staticmethod
    def _object_key(obj) -> str:
        if getattr(obj, "track_id", None) is not None:
            return f"id:{obj.track_id}"
        x1, y1, x2, y2 = obj.bbox
        center_x = int(((x1 + x2) / 2) // 40)
        center_y = int(((y1 + y2) / 2) // 40)
        return f"{obj.class_name}:{center_x}:{center_y}"


def _iso_datetime(value: float | str) -> str:
    if isinstance(value, str):
        return value
    return datetime.fromtimestamp(value).astimezone().isoformat()


def _timestamp_seconds(value: float | str) -> float:
    if isinstance(value, str):
        return datetime.fromisoformat(value).timestamp()
    return float(value)
