from dataclasses import dataclass


@dataclass
class _Track:
    track_id: int
    class_name: str
    bbox: tuple[int, int, int, int]
    missed_frames: int = 0
    source_track_id: int | None = None


class ProductCounter:
    """Counts new products while keeping detections stable between frames."""

    def __init__(
        self,
        max_missed_frames: int = 45,
        min_iou: float = 0.2,
        max_center_distance: float = 150.0,
        ignored_classes: set[str] | None = None,
    ):
        self.max_missed_frames = max_missed_frames
        self.min_iou = min_iou
        self.max_center_distance = max_center_distance
        self.ignored_classes = ignored_classes or set()
        self.total = 0
        self.class_counts: dict[str, int] = {}
        self.visible_count = 0

        self._next_track_id = 1
        self._tracks: list[_Track] = []

    def update(self, detections: list) -> int:
        """Update active tracks and return the number of newly counted products."""
        detections = [
            detection
            for detection in detections
            if detection.class_name not in self.ignored_classes
        ]
        self.visible_count = len(detections)
        unmatched_tracks = set(range(len(self._tracks)))
        unmatched_detections = set(range(len(detections)))
        candidates = []

        for track_index, track in enumerate(self._tracks):
            for detection_index, detection in enumerate(detections):
                if (
                    track.source_track_id is not None
                    and detection.track_id is not None
                ):
                    if track.source_track_id == detection.track_id:
                        candidates.append((100.0, track_index, detection_index))
                        continue
                    # If the previous frame still contained this track, a
                    # different ID represents a genuinely separate object.
                    # After a detection gap ByteTrack may assign a new ID, so
                    # fall through to spatial re-identification.
                    if track.missed_frames == 0:
                        continue

                iou = self._iou(track.bbox, detection.bbox)
                if iou >= self.min_iou:
                    score = 2.0 + iou
                else:
                    distance = self._center_distance(track.bbox, detection.bbox)
                    if distance > self.max_center_distance:
                        continue
                    score = 1.0 - distance / self.max_center_distance

                candidates.append((score, track_index, detection_index))

        for _, track_index, detection_index in sorted(candidates, reverse=True):
            if (
                track_index not in unmatched_tracks
                or detection_index not in unmatched_detections
            ):
                continue

            track = self._tracks[track_index]
            track.bbox = detections[detection_index].bbox
            track.class_name = detections[detection_index].class_name
            track.source_track_id = detections[detection_index].track_id
            track.missed_frames = 0
            unmatched_tracks.remove(track_index)
            unmatched_detections.remove(detection_index)

        for track_index in unmatched_tracks:
            self._tracks[track_index].missed_frames += 1

        self._tracks = [
            track
            for track in self._tracks
            if track.missed_frames <= self.max_missed_frames
        ]

        for detection_index in unmatched_detections:
            detection = detections[detection_index]
            self._tracks.append(
                _Track(
                    track_id=self._next_track_id,
                    class_name=detection.class_name,
                    bbox=detection.bbox,
                    source_track_id=detection.track_id,
                )
            )
            self._next_track_id += 1
            self.total += 1
            self.class_counts[detection.class_name] = (
                self.class_counts.get(detection.class_name, 0) + 1
            )

        return len(unmatched_detections)

    def reset(self):
        self.total = 0
        self.class_counts.clear()
        self.visible_count = 0
        self._tracks.clear()
        self._next_track_id = 1

    @staticmethod
    def _iou(first, second) -> float:
        ax1, ay1, ax2, ay2 = first
        bx1, by1, bx2, by2 = second

        intersection_width = max(0, min(ax2, bx2) - max(ax1, bx1))
        intersection_height = max(0, min(ay2, by2) - max(ay1, by1))
        intersection = intersection_width * intersection_height

        first_area = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        second_area = max(0, bx2 - bx1) * max(0, by2 - by1)
        union = first_area + second_area - intersection

        return intersection / union if union > 0 else 0.0

    @staticmethod
    def _center_distance(first, second) -> float:
        ax1, ay1, ax2, ay2 = first
        bx1, by1, bx2, by2 = second
        ax, ay = (ax1 + ax2) / 2, (ay1 + ay2) / 2
        bx, by = (bx1 + bx2) / 2, (by1 + by2) / 2
        return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5
