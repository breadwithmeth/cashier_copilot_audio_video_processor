"""Synchronous video file reader.

Drop-in replacement for :class:`camera.rtsp_reader.RTSPReader` when the
video source is a local file (mp4, avi, mov, …) instead of an RTSP stream.

Unlike ``RTSPReader`` it reads frames on demand from the calling thread —
no background thread, no reconnection logic, no buffer management.
"""

from __future__ import annotations

import cv2


class VideoFileReader:
    """Read frames sequentially from a local video file."""

    def __init__(self, name: str, url: str):
        self.name = name
        self.url = url

        self._capture = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        if not self._capture.isOpened():
            raise RuntimeError(
                f"[{name}] Cannot open video file: {url}"
            )

        print(f"[{name}] Video file opened: {url}")

    def get_frame(self):
        ok, frame = self._capture.read()
        if not ok:
            return None
        return frame

    def stop(self):
        if self._capture is not None:
            self._capture.release()
            self._capture = None
