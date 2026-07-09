from __future__ import annotations

import logging
import multiprocessing
import queue
import threading
import time
from collections import deque
from typing import Optional

from .messages import ClipRequest, ClipResponse, InferenceFrame

logger = logging.getLogger(__name__)


def utc_timestamp_ms() -> int:
    return int(time.time() * 1000)


class VideoBuffer:
    def __init__(
        self,
        camera_id: str,
        rtsp_url: str,
        fps: int = 25,
        buffer_seconds: int = 60,
        jpeg_quality: int = 85,
        inference_stride: int = 5,
    ) -> None:
        self.camera_id = camera_id
        self.rtsp_url = rtsp_url
        self.fps = fps
        self.buffer_seconds = buffer_seconds
        self.jpeg_quality = jpeg_quality
        self.inference_stride = inference_stride
        self.buffer: deque[tuple[int, bytes]] = deque(maxlen=fps * buffer_seconds)
        self.lock = threading.Lock()

    def start_capture(
        self,
        inference_queue: multiprocessing.Queue,
        clip_request_queue: Optional[multiprocessing.Queue] = None,
        clip_response_queue: Optional[multiprocessing.Queue] = None,
        stop_event: Optional[multiprocessing.Event] = None,
    ) -> None:
        """Capture frames, keep JPEG ring buffer, and send every Nth frame to AI queue."""
        import cv2

        cap = None
        frame_count = 0
        last_frame_at: Optional[float] = None
        next_drop_log_at = 0.0
        frame_interval_s = 1.0 / float(self.fps)

        while stop_event is None or not stop_event.is_set():
            self._drain_clip_requests(clip_request_queue, clip_response_queue)

            if cap is None or not cap.isOpened():
                cap = cv2.VideoCapture(self.rtsp_url)
                if not cap.isOpened():
                    logger.error("RTSP connect failed for camera=%s; retrying in 5s", self.camera_id)
                    time.sleep(5)
                    continue
                logger.info("RTSP capture opened for camera=%s", self.camera_id)

            ret, frame = cap.read()
            if not ret:
                logger.warning("RTSP read failed for camera=%s; reconnecting in 5s", self.camera_id)
                cap.release()
                cap = None
                time.sleep(5)
                continue

            now = time.perf_counter()
            if last_frame_at is not None and now - last_frame_at > frame_interval_s:
                skipped = max(1, int((now - last_frame_at) / frame_interval_s) - 1)
                if now >= next_drop_log_at:
                    logger.warning(
                        "Video capture lag camera=%s delta_ms=%.1f skipped_estimate=%s",
                        self.camera_id,
                        (now - last_frame_at) * 1000.0,
                        skipped,
                    )
                    next_drop_log_at = now + 10.0
            last_frame_at = now

            ts_ms = utc_timestamp_ms()
            ok, jpeg = cv2.imencode(
                ".jpg",
                frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), int(self.jpeg_quality)],
            )
            if not ok:
                logger.warning("JPEG encode failed for camera=%s timestamp_ms=%s", self.camera_id, ts_ms)
                continue

            jpeg_bytes = jpeg.tobytes()
            with self.lock:
                self.buffer.append((ts_ms, jpeg_bytes))

            if frame_count % self.inference_stride == 0:
                self._enqueue_inference_frame(inference_queue, ts_ms, jpeg_bytes)
            frame_count += 1

        if cap is not None:
            cap.release()
        logger.info("Video capture stopped for camera=%s", self.camera_id)

    def get_clip(self, start_ts: int, end_ts: int) -> list[tuple[int, bytes]]:
        """Return JPEG frames for a millisecond timestamp range."""
        with self.lock:
            return [(ts, data) for ts, data in self.buffer if start_ts <= ts <= end_ts]

    def _enqueue_inference_frame(
        self,
        inference_queue: multiprocessing.Queue,
        timestamp_ms: int,
        jpeg_bytes: bytes,
    ) -> None:
        message = InferenceFrame(
            camera_id=self.camera_id,
            timestamp_ms=timestamp_ms,
            jpeg_bytes=jpeg_bytes,
        )
        try:
            inference_queue.put_nowait(message)
        except queue.Full:
            logger.warning("Inference queue full; dropping frame camera=%s", self.camera_id)

    def _drain_clip_requests(
        self,
        clip_request_queue: Optional[multiprocessing.Queue],
        clip_response_queue: Optional[multiprocessing.Queue],
    ) -> None:
        if clip_request_queue is None or clip_response_queue is None:
            return

        while True:
            try:
                request = clip_request_queue.get_nowait()
            except queue.Empty:
                return

            if not isinstance(request, ClipRequest):
                logger.warning("Ignoring unexpected clip request payload: %r", request)
                continue

            if request.camera_id != self.camera_id:
                continue

            try:
                frames = self.get_clip(request.start_timestamp_ms, request.end_timestamp_ms)
                response = ClipResponse(
                    request_id=request.request_id,
                    task_id=request.task_id,
                    camera_id=request.camera_id,
                    frames=frames,
                )
            except Exception as exc:
                logger.exception("Failed to collect clip frames")
                response = ClipResponse(
                    request_id=request.request_id,
                    task_id=request.task_id,
                    camera_id=request.camera_id,
                    frames=[],
                    error=str(exc),
                )
            clip_response_queue.put(response)


class VideoGrabberProcess(multiprocessing.Process):
    def __init__(
        self,
        camera_id: str,
        rtsp_url: str,
        inference_queue: multiprocessing.Queue,
        clip_request_queue: multiprocessing.Queue,
        clip_response_queue: multiprocessing.Queue,
        stop_event: multiprocessing.Event,
        fps: int = 25,
        buffer_seconds: int = 60,
        jpeg_quality: int = 85,
        inference_stride: int = 5,
    ) -> None:
        super().__init__(name=f"VideoGrabberProcess-{camera_id}")
        self.camera_id = camera_id
        self.rtsp_url = rtsp_url
        self.inference_queue = inference_queue
        self.clip_request_queue = clip_request_queue
        self.clip_response_queue = clip_response_queue
        self.stop_event = stop_event
        self.fps = fps
        self.buffer_seconds = buffer_seconds
        self.jpeg_quality = jpeg_quality
        self.inference_stride = inference_stride

    def run(self) -> None:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(processName)s %(name)s: %(message)s",
        )
        buffer = VideoBuffer(
            camera_id=self.camera_id,
            rtsp_url=self.rtsp_url,
            fps=self.fps,
            buffer_seconds=self.buffer_seconds,
            jpeg_quality=self.jpeg_quality,
            inference_stride=self.inference_stride,
        )
        buffer.start_capture(
            inference_queue=self.inference_queue,
            clip_request_queue=self.clip_request_queue,
            clip_response_queue=self.clip_response_queue,
            stop_event=self.stop_event,
        )
