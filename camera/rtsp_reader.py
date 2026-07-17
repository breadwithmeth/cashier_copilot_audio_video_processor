import logging
import threading
import time

import cv2

logger = logging.getLogger(__name__)


class RTSPReader:

    def __init__(
        self,
        name: str,
        url: str,
    ):
        self.name = name
        self.url = url

        self._frame = None
        self._lock = threading.Lock()

        self._running = True
        self._capture = None

        self._frame_counter = 0
        self._last_frame_counter = 0
        self._consecutive_failures = 0
        self._reconnected = False

        self._thread = threading.Thread(
            target=self._worker,
            daemon=True,
        )

        self._thread.start()

    def stop(self):
        self._running = False

        if self._capture is not None:
            self._capture.release()

    def get_frame(self):
        with self._lock:

            if self._frame is None:
                return None

            frame = self._frame.copy()

        if self._reconnected:
            self._reconnected = False
            logger.warning(
                "RTSP stream reconnected — tracker state may be stale, "
                "consider resetting track IDs"
            )

        return frame

    def _connect(self):

        if self._capture is not None:
            self._capture.release()

        print(f"[{self.name}] Connecting...")

        self._capture = cv2.VideoCapture(
            self.url,
            cv2.CAP_FFMPEG,
        )

        self._capture.set(
            cv2.CAP_PROP_BUFFERSIZE,
            1,
        )

        if self._capture.isOpened():
            print(f"[{self.name}] Connected")
        else:
            print(f"[{self.name}] Connection failed")

    def _worker(self):

        while self._running:

            if (
                self._capture is None
                or
                not self._capture.isOpened()
            ):
                self._connect()

                time.sleep(1)

                continue

            ok, frame = self._capture.read()

            if not ok:

                print(f"[{self.name}] Stream lost")

                self._capture.release()
                self._capture = None

                self._consecutive_failures += 1
                backoff = min(2 ** self._consecutive_failures, 30)
                logger.warning(
                    f"[{self.name}] Stream read failed "
                    f"(consecutive failures: {self._consecutive_failures}), "
                    f"reconnecting in {backoff}s"
                )
                time.sleep(backoff)

                self._reconnected = True
                continue

            self._consecutive_failures = 0
            self._frame_counter += 1

            if self._last_frame_counter > 0:
                gap = self._frame_counter - self._last_frame_counter
                if gap > 1:
                    logger.warning(
                        f"[{self.name}] Frame gap detected: "
                        f"expected {self._last_frame_counter + 1}, "
                        f"got {self._frame_counter} (gap={gap})"
                    )

            self._last_frame_counter = self._frame_counter

            with self._lock:
                self._frame = frame

        if self._capture is not None:
            self._capture.release()