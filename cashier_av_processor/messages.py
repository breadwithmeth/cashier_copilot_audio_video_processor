from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class InferenceFrame:
    camera_id: str
    timestamp_ms: int
    jpeg_bytes: bytes


@dataclass(frozen=True)
class ClipRequest:
    request_id: str
    task_id: int
    camera_id: str
    start_timestamp_ms: int
    end_timestamp_ms: int


@dataclass(frozen=True)
class ClipResponse:
    request_id: str
    task_id: int
    camera_id: str
    frames: list[tuple[int, bytes]]
    error: Optional[str] = None
