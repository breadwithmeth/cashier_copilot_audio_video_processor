from __future__ import annotations

import json
import logging
import os
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, Optional

logger = logging.getLogger(__name__)


LOAD_ACTIVE_CAMERAS_SQL = """
SELECT id, roi_config, pos_id FROM cameras WHERE status = 'active';
"""

INSERT_CV_EVENT_SQL = """
INSERT INTO cv_events (
    camera_id,
    event_type,
    timestamp_ms,
    confidence,
    model_name,
    weights_version,
    inference_time_ms,
    bbox_jsonb,
    snapshot_path
)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s);
"""

CLAIM_VIDEO_EXPORT_TASK_SQL = """
WITH next_task AS (
    SELECT id
    FROM tasks
    WHERE status = 'pending' AND task_type = 'video_export'
    ORDER BY created_at ASC
    FOR UPDATE SKIP LOCKED
    LIMIT 1
)
UPDATE tasks
SET status = 'processing',
    updated_at = CURRENT_TIMESTAMP
WHERE id IN (SELECT id FROM next_task)
RETURNING id, task_type, camera_id, violation_id, payload;
"""

COMPLETE_SAVE_CLIP_TASK_SQL = """
UPDATE tasks
SET status = 'completed',
    result_path = %s,
    error_message = NULL,
    updated_at = CURRENT_TIMESTAMP
WHERE id = %s;
"""

FAIL_SAVE_CLIP_TASK_SQL = """
UPDATE tasks
SET status = 'failed',
    error_message = %s,
    updated_at = CURRENT_TIMESTAMP
WHERE id = %s;
"""

INSERT_SPEECH_TRANSCRIPT_SQL = """
INSERT INTO speech_transcripts (
    pos_id,
    transcript,
    timestamp_ms,
    duration_ms,
    confidence,
    model_name,
    weights_version
)
VALUES (%s, %s, %s, %s, %s, %s, %s);
"""


@dataclass(frozen=True)
class CvEvent:
    camera_id: str
    event_type: str
    timestamp_ms: int
    confidence: float
    model_name: str
    weights_version: str
    inference_time_ms: float
    bbox_jsonb: dict[str, Any]
    snapshot_path: Optional[str] = None


@dataclass(frozen=True)
class SpeechTranscript:
    pos_id: str
    transcript: str
    timestamp_ms: int
    duration_ms: int
    confidence: Optional[float]
    model_name: str
    weights_version: str


@dataclass(frozen=True)
class VideoExportTask:
    id: int
    task_type: str
    camera_id: str
    violation_id: Optional[int]
    payload: dict[str, Any]
    start_timestamp_ms: Optional[int]
    end_timestamp_ms: Optional[int]


class DatabaseClient:
    """Per-process psycopg2 ThreadedConnectionPool wrapper."""

    def __init__(self, dsn: str, minconn: int = 1, maxconn: int = 4) -> None:
        self.dsn = dsn
        self.minconn = minconn
        self.maxconn = maxconn
        self._pool = None

    def _ensure_pool(self):
        if self._pool is None:
            from psycopg2.pool import ThreadedConnectionPool

            self._pool = ThreadedConnectionPool(self.minconn, self.maxconn, self.dsn)
        return self._pool

    @contextmanager
    def connection(self) -> Iterator[Any]:
        pool = self._ensure_pool()
        conn = pool.getconn()
        close_conn = False
        try:
            yield conn
            conn.commit()
        except Exception:
            try:
                conn.rollback()
            except Exception:
                close_conn = True
                logger.exception("Failed to rollback PostgreSQL connection; closing it")
            raise
        finally:
            if getattr(conn, "closed", 0):
                close_conn = True
            pool.putconn(conn, close=close_conn)

    def close(self) -> None:
        if self._pool is not None:
            self._pool.closeall()
            self._pool = None

    def fetchall(self, sql: str, params: tuple[Any, ...] = ()) -> list[tuple[Any, ...]]:
        with self.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                return cur.fetchall()

    def fetchone(self, sql: str, params: tuple[Any, ...] = ()) -> Optional[tuple[Any, ...]]:
        with self.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                return cur.fetchone()

    def execute(self, sql: str, params: tuple[Any, ...] = ()) -> None:
        with self.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)


class DurableEventStore:
    """Writes analytics events to PostgreSQL and falls back to JSONL on DB outages."""

    def __init__(self, db: DatabaseClient, spool_dir: Path) -> None:
        self.db = db
        self.spool_dir = spool_dir
        self.spool_dir.mkdir(parents=True, exist_ok=True)
        self.spool_path = self.spool_dir / f"pending_events_{os.getpid()}.jsonl"

    def load_active_cameras(self) -> list[tuple[Any, Any, Any]]:
        return self.db.fetchall(LOAD_ACTIVE_CAMERAS_SQL)

    def insert_cv_event(self, event: CvEvent) -> None:
        try:
            self._insert_cv_event(event)
        except Exception:
            logger.exception("Failed to insert cv event; spooling to disk")
            self._spool("cv_event", asdict(event))

    def insert_speech_transcript(self, transcript: SpeechTranscript) -> None:
        try:
            self._insert_speech_transcript(transcript)
        except Exception:
            logger.exception("Failed to insert speech transcript; spooling to disk")
            self._spool("speech_transcript", asdict(transcript))

    def poll_clip_task(self) -> Optional[VideoExportTask]:
        row = self.db.fetchone(CLAIM_VIDEO_EXPORT_TASK_SQL)
        if row is None:
            return None
        task_id, task_type, camera_id, violation_id, payload = row
        payload_dict = self._coerce_payload(payload)
        start_ts = self._extract_timestamp_ms(
            payload_dict,
            ("start_timestamp_ms", "start_ts", "start_ms", "from_timestamp_ms", "from_ms"),
        )
        end_ts = self._extract_timestamp_ms(
            payload_dict,
            ("end_timestamp_ms", "end_ts", "end_ms", "to_timestamp_ms", "to_ms"),
        )
        return VideoExportTask(
            id=int(task_id),
            task_type=str(task_type),
            camera_id=str(camera_id),
            violation_id=int(violation_id) if violation_id is not None else None,
            payload=payload_dict,
            start_timestamp_ms=start_ts,
            end_timestamp_ms=end_ts,
        )

    def complete_clip_task(self, task_id: int, result_path: str) -> None:
        self.db.execute(COMPLETE_SAVE_CLIP_TASK_SQL, (result_path, task_id))

    def fail_clip_task(self, task_id: int, error_message: str) -> None:
        logger.error("Marking clip task failed task_id=%s error=%s", task_id, error_message)
        self.db.execute(FAIL_SAVE_CLIP_TASK_SQL, (error_message[:2000], task_id))

    def replay_pending(self, max_events: int = 100) -> None:
        if not self.spool_path.exists():
            return

        lines = self.spool_path.read_text(encoding="utf-8").splitlines()
        if not lines:
            self.spool_path.unlink(missing_ok=True)
            return

        remaining: list[str] = []
        replayed = 0
        for line in lines:
            if replayed >= max_events:
                remaining.append(line)
                continue
            try:
                record = json.loads(line)
                kind = record["kind"]
                payload = record["payload"]
                if kind == "cv_event":
                    self._insert_cv_event(CvEvent(**payload))
                elif kind == "speech_transcript":
                    self._insert_speech_transcript(SpeechTranscript(**payload))
                else:
                    logger.warning("Unknown spooled event kind: %s", kind)
                replayed += 1
            except Exception:
                logger.exception("Replay failed; keeping remaining spool entries")
                remaining.append(line)

        if remaining:
            self.spool_path.write_text("\n".join(remaining) + "\n", encoding="utf-8")
        else:
            self.spool_path.unlink(missing_ok=True)

        if replayed:
            logger.info("Replayed %s spooled DB events", replayed)

    def _insert_cv_event(self, event: CvEvent) -> None:
        from psycopg2.extras import Json

        self.db.execute(
            INSERT_CV_EVENT_SQL,
            (
                event.camera_id,
                event.event_type,
                event.timestamp_ms,
                event.confidence,
                event.model_name,
                event.weights_version,
                int(round(event.inference_time_ms)),
                Json(event.bbox_jsonb),
                event.snapshot_path or "",
            ),
        )

    def _insert_speech_transcript(self, transcript: SpeechTranscript) -> None:
        self.db.execute(
            INSERT_SPEECH_TRANSCRIPT_SQL,
            (
                transcript.pos_id,
                transcript.transcript,
                transcript.timestamp_ms,
                transcript.duration_ms,
                transcript.confidence if transcript.confidence is not None else 0.0,
                transcript.model_name,
                transcript.weights_version,
            ),
        )

    @staticmethod
    def _coerce_payload(raw: Any) -> dict[str, Any]:
        if raw is None:
            return {}
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, str):
            return json.loads(raw)
        return dict(raw)

    @staticmethod
    def _extract_timestamp_ms(payload: dict[str, Any], keys: tuple[str, ...]) -> Optional[int]:
        for key in keys:
            value = payload.get(key)
            if value in (None, ""):
                continue
            return int(value)

        clip_payload = payload.get("clip")
        if isinstance(clip_payload, dict):
            for key in keys:
                value = clip_payload.get(key)
                if value in (None, ""):
                    continue
                return int(value)
        return None

    def _spool(self, kind: str, payload: dict[str, Any]) -> None:
        record = {
            "kind": kind,
            "payload": payload,
            "spooled_at_ms": int(time.time() * 1000),
        }
        with self.spool_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
