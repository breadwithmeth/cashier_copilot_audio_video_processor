from dataclasses import dataclass
from typing import Tuple


BBox = Tuple[int, int, int, int]
Polygon = list[tuple[int, int]]


@dataclass
class Detection:
    class_name: str
    confidence: float
    bbox: BBox
    roi_name: str = "scan_zone"
    track_id: int | None = None
    polygon: Polygon | None = None

    @property
    def x1(self) -> int:
        return self.bbox[0]

    @property
    def y1(self) -> int:
        return self.bbox[1]

    @property
    def x2(self) -> int:
        return self.bbox[2]

    @property
    def y2(self) -> int:
        return self.bbox[3]

    @property
    def center(self) -> tuple[int, int]:
        return (
            int((self.x1 + self.x2) / 2),
            int((self.y1 + self.y2) / 2),
        )


@dataclass
class ScanResult:
    objects: list[Detection]
    process_ms: int = 0
