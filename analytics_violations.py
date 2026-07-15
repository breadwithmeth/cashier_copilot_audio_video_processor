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
    CUSTOMER_WAITING_VIOLATION_SECONDS,
    VIOLATION_EVENT_COOLDOWN_SECONDS,
)


class VideoViolationMonitor:
    def __init__(
        self,
        camera_name: str,
        waiting_threshold_seconds: float = CUSTOMER_WAITING_VIOLATION_SECONDS,
        cooldown_seconds: float = VIOLATION_EVENT_COOLDOWN_SECONDS,
    ):
        self.camera_name = camera_name
        self.waiting_threshold_seconds = waiting_threshold_seconds
        self.cooldown_seconds = cooldown_seconds
        self._active_since: dict[str, float] = {}
        self._last_sent_at: dict[str, float] = {}
        self._jobs: queue.Queue[dict | None] = queue.Queue()
        self._worker = threading.Thread(target=self._send_worker, daemon=True)
        self._worker.start()

    def evaluate(self, state) -> None:
        now = time.time()
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
            payload={
                "customerWaitSeconds": round(wait_seconds, 3),
                "thresholdSeconds": self.waiting_threshold_seconds,
                "cashierPresent": state.cashier_is_present,
                "customerPresent": state.customer_is_present,
                "checkoutStatus": state.status.value,
            },
        )

    def stop(self) -> None:
        self._jobs.put(None)
        self._worker.join(timeout=10)

    def _update_rule(
        self,
        code: str,
        active: bool,
        now: float,
        severity: str,
        message: str,
        started_at: float | None,
        duration_seconds: float,
        payload: dict,
    ) -> None:
        if not active:
            self._active_since.pop(code, None)
            return

        active_since = self._active_since.setdefault(code, started_at or now)
        last_sent_at = self._last_sent_at.get(code)
        if last_sent_at is not None and now - last_sent_at < self.cooldown_seconds:
            return

        self._last_sent_at[code] = now
        self._jobs.put({
            "code": code,
            "severity": severity,
            "message": message,
            "started_at": active_since,
            "ended_at": now,
            "duration_seconds": duration_seconds,
            "payload": payload,
        })

    def _send_worker(self) -> None:
        while True:
            event = self._jobs.get()
            if event is None:
                return
            self._send_event(event)

    def _send_event(self, event: dict) -> dict:
        if not ANALYTICS_API_BASE_URL or not ANALYTICS_API_KEY:
            print(f"[{self.camera_name}] Violation event skipped: analytics API is not configured")
            return {"status": "skipped", "reason": "analytics_api_not_configured"}

        started_at = datetime.fromtimestamp(event["started_at"]).astimezone().isoformat()
        ended_at = datetime.fromtimestamp(event["ended_at"]).astimezone().isoformat()
        occurred_at_ms = int(event["ended_at"] * 1000)
        external_event_id = f"{self.camera_name}-{event['code'].lower()}-{occurred_at_ms}"
        payload = {
            "externalEventId": external_event_id,
            "idempotencyKey": external_event_id,
            "cameraCode": self.camera_name,
            "eventType": event["code"],
            "source": "cashier_copilot_video",
            "occurredAt": ended_at,
            "startedAt": started_at,
            "endedAt": ended_at,
            "severity": event["severity"],
            "message": event["message"],
            "payload": {
                **event["payload"],
                "durationSeconds": round(event["duration_seconds"], 3),
            },
        }
        if ANALYTICS_STORE_CODE:
            payload["storeCode"] = ANALYTICS_STORE_CODE
            payload["payload"]["storeCode"] = ANALYTICS_STORE_CODE
        if ANALYTICS_REGISTER_CODE:
            payload["registerCode"] = ANALYTICS_REGISTER_CODE

        url = f"{ANALYTICS_API_BASE_URL}{ANALYTICS_VIOLATION_EVENTS_PATH}"
        request = urllib.request.Request(
            url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
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
            print(f"[{self.camera_name}] Violation event sent: {external_event_id}")
            return {
                "status": "sent",
                "endpoint": url,
                "externalEventId": external_event_id,
                "statusCode": response.status,
                "response": response_body[:1000],
            }
        except urllib.error.HTTPError as error:
            body = error.read().decode("utf-8", errors="replace")
            print(f"[{self.camera_name}] Violation event HTTP error {error.code}: {body[:300]}")
            return {
                "status": "failed",
                "endpoint": url,
                "externalEventId": external_event_id,
                "statusCode": error.code,
                "error": body[:1000],
            }
        except (OSError, urllib.error.URLError) as error:
            print(f"[{self.camera_name}] Violation event send error: {error}")
            return {
                "status": "failed",
                "endpoint": url,
                "externalEventId": external_event_id,
                "error": str(error),
            }
