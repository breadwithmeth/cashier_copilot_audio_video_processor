from dataclasses import dataclass
from typing import Tuple


BBox = Tuple[int, int, int, int]


@dataclass
class PersonDetection:
    role: str
    confidence: float
    bbox: BBox

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
class PersonResult:
    customer_detected: bool
    cashier_detected: bool
    persons: list[PersonDetection]
    customer_ms: int = 0
    cashier_ms: int = 0