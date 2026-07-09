from __future__ import annotations

import logging
import multiprocessing
import queue
import time
import uuid
from pathlib import Path
from typing import Optional

from .db import DatabaseClient, DurableEventStore
from .messages import ClipRequest, ClipResponse
from .video import VideoBuffer

logger = logging.getLogger(__name__)


class ClipExporter(multiprocessing.Process):
    def __init__(
        self,
        db_pool: DatabaseClient | str,
        video_buffers: dict[str, VideoBuffer] | None = None,
        clip_request_queue: Optional[multiprocessing.Queue] = None,
        clip_response_queue: Optional[multiprocessing.Queue] = None,
        stop_event: Optional[multiprocessing.Event] = None,
        output_dir: Path | str = "clips",
        spool_dir: Path | str = "spool",
        poll_interval_s: float = 1.0,
        response_timeout_s: float = 15.0,
    ) -> None:
        super().__init__(name="ClipExporterProcess")
        self.db_pool = db_pool
        self.video_buffers = video_buffers or {}
        self.clip_request_queue = clip_request_queue
        self.clip_response_queue = clip_response_queue
        self.stop_event = stop_event
        self.output_dir = Path(output_dir)
        self.spool_dir = Path(spool_dir)
        self.poll_interval_s = poll_interval_s
        self.response_timeout_s = response_timeout_s
        self._pending_responses: dict[str, ClipResponse] = {}

    def run(self) -> None:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(processName)s %(name)s: %(message)s",
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        db = self.db_pool if isinstance(self.db_pool, DatabaseClient) else DatabaseClient(self.db_pool)
        store = DurableEventStore(db, self.spool_dir)

        try:
            while self.stop_event is None or not self.stop_event.is_set():
                try:
                    task = store.poll_clip_task()
                except Exception:
                    logger.exception("Failed to poll clip tasks")
                    time.sleep(self.poll_interval_s)
                    continue

                if task is None:
                    time.sleep(self.poll_interval_s)
                    continue

                try:
                    output_path = self._export_task(
                        task.id,
                        task.camera_id,
                        task.violation_id,
                        task.start_timestamp_ms,
                        task.end_timestamp_ms,
                    )
                    store.complete_clip_task(task.id, str(output_path))
                    logger.info("Clip task completed task_id=%s path=%s", task.id, output_path)
                except Exception as exc:
                    logger.exception("Clip task failed task_id=%s", task.id)
                    try:
                        store.fail_clip_task(task.id, str(exc))
                    except Exception:
                        logger.exception("Failed to mark clip task failed task_id=%s", task.id)
        finally:
            db.close()

    def _export_task(
        self,
        task_id: int,
        camera_id: str,
        violation_id: Optional[int],
        start_ts: Optional[int],
        end_ts: Optional[int],
    ) -> Path:
        if start_ts is None or end_ts is None:
            raise RuntimeError(
                "video_export payload must include start_timestamp_ms and end_timestamp_ms"
            )
        if end_ts < start_ts:
            raise RuntimeError("video_export end_timestamp_ms must be >= start_timestamp_ms")

        frames = self._get_clip_frames(task_id, camera_id, start_ts, end_ts)
        if not frames:
            raise RuntimeError(f"no frames in buffer for camera={camera_id} range={start_ts}-{end_ts}")

        return self._write_mp4(task_id, camera_id, violation_id, start_ts, end_ts, frames)

    def _get_clip_frames(
        self,
        task_id: int,
        camera_id: str,
        start_ts: int,
        end_ts: int,
    ) -> list[tuple[int, bytes]]:
        buffer = self.video_buffers.get(camera_id)
        if buffer is not None:
            return buffer.get_clip(start_ts, end_ts)

        if self.clip_request_queue is None or self.clip_response_queue is None:
            raise RuntimeError("clip IPC queues are not configured")

        request_id = str(uuid.uuid4())
        request = ClipRequest(
            request_id=request_id,
            task_id=task_id,
            camera_id=camera_id,
            start_timestamp_ms=start_ts,
            end_timestamp_ms=end_ts,
        )
        self.clip_request_queue.put(request)
        response = self._wait_for_response(request_id)
        if response.error:
            raise RuntimeError(response.error)
        return response.frames

    def _wait_for_response(self, request_id: str) -> ClipResponse:
        cached = self._pending_responses.pop(request_id, None)
        if cached is not None:
            return cached

        deadline = time.monotonic() + self.response_timeout_s
        while time.monotonic() < deadline:
            try:
                response = self.clip_response_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if not isinstance(response, ClipResponse):
                logger.warning("Ignoring unexpected clip response payload: %r", response)
                continue

            if response.request_id == request_id:
                return response
            self._pending_responses[response.request_id] = response

        raise TimeoutError(f"clip response timeout request_id={request_id}")

    def _write_mp4(
        self,
        task_id: int,
        camera_id: str,
        violation_id: Optional[int],
        start_ts: int,
        end_ts: int,
        jpeg_frames: list[tuple[int, bytes]],
    ) -> Path:
        import cv2
        import numpy as np

        decoded = []
        for ts, jpeg_bytes in jpeg_frames:
            frame = cv2.imdecode(np.frombuffer(jpeg_bytes, np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                logger.warning("Skipping undecodable clip frame task_id=%s ts=%s", task_id, ts)
                continue
            decoded.append((ts, frame))

        if not decoded:
            raise RuntimeError("all selected JPEG frames failed to decode")

        height, width = decoded[0][1].shape[:2]
        violation_part = violation_id if violation_id is not None else "none"
        output_path = (
            self.output_dir
            / f"violation_{violation_part}_task_{task_id}_{camera_id}_{start_ts}_{end_ts}.mp4"
        )
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            25,
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"failed to open VideoWriter path={output_path}")

        try:
            for _, frame in decoded:
                if frame.shape[1] != width or frame.shape[0] != height:
                    frame = cv2.resize(frame, (width, height))
                writer.write(frame)
        finally:
            writer.release()

        return output_path
