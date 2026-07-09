import threading
import time

import cv2


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

            return self._frame.copy()

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

                time.sleep(2)

                continue

            with self._lock:
                self._frame = frame

        if self._capture is not None:
            self._capture.release()