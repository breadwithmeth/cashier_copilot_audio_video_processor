from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass


@dataclass(frozen=True)
class PersonAction:
    label: str
    confidence: float


@dataclass(frozen=True)
class SkeletonFrame:
    timestamp: float
    bbox: tuple[int, int, int, int]
    keypoints: list[tuple[int, int, float] | None]


class SkeletonActionBuffer:
    def __init__(
        self,
        window_seconds: float = 3.0,
        min_frames: int = 6,
    ):
        self.window_seconds = window_seconds
        self.min_frames = min_frames
        self._tracks = defaultdict(deque)

    def update(
        self,
        track_id: int | None,
        bbox: tuple[int, int, int, int],
        keypoints: list[tuple[int, int, float] | None] | None,
        now: float | None = None,
    ) -> PersonAction:
        if track_id is None or not keypoints:
            return PersonAction(label="unknown", confidence=0.0)

        now = now or time.time()
        frames = self._tracks[track_id]
        frames.append(SkeletonFrame(timestamp=now, bbox=bbox, keypoints=keypoints))

        while frames and now - frames[0].timestamp > self.window_seconds:
            frames.popleft()

        return self._classify(frames)

    def prune(self, max_idle_seconds: float = 10.0) -> None:
        now = time.time()
        stale_ids = [
            track_id
            for track_id, frames in self._tracks.items()
            if not frames or now - frames[-1].timestamp > max_idle_seconds
        ]
        for track_id in stale_ids:
            del self._tracks[track_id]

    def _classify(self, frames) -> PersonAction:
        if len(frames) < self.min_frames:
            return PersonAction(label="observing", confidence=0.2)

        first = frames[0]
        last = frames[-1]
        duration = max(0.001, last.timestamp - first.timestamp)

        first_center = self._bbox_center(first.bbox)
        last_center = self._bbox_center(last.bbox)
        bbox_height = max(1, last.bbox[3] - last.bbox[1])
        center_speed = self._distance(first_center, last_center) / bbox_height / duration

        wrist_motion = max(
            self._keypoint_motion(frames, 9, bbox_height),
            self._keypoint_motion(frames, 10, bbox_height),
        ) / duration

        if center_speed > 0.18:
            return PersonAction(label="walking", confidence=min(0.95, center_speed * 2.5))

        if wrist_motion > 0.22:
            return PersonAction(label="reaching", confidence=min(0.9, wrist_motion * 2.0))

        return PersonAction(label="standing", confidence=0.65)

    @staticmethod
    def _bbox_center(bbox):
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    @staticmethod
    def _distance(point_a, point_b):
        return ((point_a[0] - point_b[0]) ** 2 + (point_a[1] - point_b[1]) ** 2) ** 0.5

    @classmethod
    def _keypoint_motion(cls, frames, keypoint_index, scale):
        first_point = cls._first_valid_keypoint(frames, keypoint_index)
        last_point = cls._last_valid_keypoint(frames, keypoint_index)
        if first_point is None or last_point is None:
            return 0.0

        return cls._distance(first_point, last_point) / max(1, scale)

    @staticmethod
    def _first_valid_keypoint(frames, keypoint_index):
        for frame in frames:
            point = frame.keypoints[keypoint_index]
            if point is not None:
                return point[0], point[1]
        return None

    @staticmethod
    def _last_valid_keypoint(frames, keypoint_index):
        for frame in reversed(frames):
            point = frame.keypoints[keypoint_index]
            if point is not None:
                return point[0], point[1]
        return None
