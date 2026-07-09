from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def load_dotenv(path: Path = Path(".env")) -> None:
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return int(raw)


def _float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return float(raw)


def _db_dsn_from_env() -> str:
    explicit_dsn = os.getenv("DB_DSN")
    if explicit_dsn:
        return explicit_dsn

    db_name = os.getenv("DB_NAME")
    db_user = os.getenv("DB_USER")
    db_password = os.getenv("DB_PASSWORD")
    db_host = os.getenv("DB_HOST")
    db_port = os.getenv("DB_PORT", "5432")
    db_sslmode = os.getenv("DB_SSLMODE")

    if db_name and db_user and db_password and db_host:
        dsn = (
            f"host={db_host} port={db_port} dbname={db_name} "
            f"user={db_user} password={db_password}"
        )
        if db_sslmode:
            dsn = f"{dsn} sslmode={db_sslmode}"
        return dsn

    raise ValueError("DB_DSN or DB_HOST/DB_NAME/DB_USER/DB_PASSWORD is required")


@dataclass(frozen=True)
class AppConfig:
    db_dsn: str
    camera_id: str
    rtsp_url: str
    pos_id: str
    audio_source: str
    yolo_model_path: str = "yolov11x.pt"
    yolo_device: str = "cpu"
    fps: int = 25
    inference_stride: int = 5
    buffer_seconds: int = 120
    jpeg_quality: int = 85
    inference_queue_size: int = 64
    clip_output_dir: Path = Path("clips")
    spool_dir: Path = Path("spool")
    task_poll_interval_s: float = 1.0
    clip_response_timeout_s: float = 15.0
    yolo_profile_interval_s: float = 10.0
    vad_aggressiveness: int = 3
    audio_sample_rate: int = 16000
    audio_frame_ms: int = 30
    audio_start_window_ms: int = 300
    audio_start_ratio: float = 0.9
    audio_end_silence_ms: int = 1500
    audio_max_segment_ms: int = 30000
    whisper_model_name: str = "small"
    whisper_device: str = "cpu"
    whisper_compute_type: str = "int8"
    analytics_api_base_url: str = ""
    analytics_api_key: str = ""
    analytics_stream_url: str = ""
    analytics_stream_type: str = "hls"
    analytics_register_timeout_s: float = 5.0

    @classmethod
    def from_env(cls) -> "AppConfig":
        load_dotenv()
        db_dsn = _db_dsn_from_env()

        camera_id = os.getenv("CAMERA_ID")
        if not camera_id:
            raise ValueError("CAMERA_ID is required")

        rtsp_url = os.getenv("RTSP_URL")
        if not rtsp_url:
            raise ValueError("RTSP_URL is required")

        pos_id = os.getenv("POS_ID") or camera_id
        analytics_stream_base_url = os.getenv("ANALYTICS_STREAM_BASE_URL", "http://127.0.0.1:8888")
        analytics_stream_url = os.getenv("ANALYTICS_STREAM_URL") or (
            f"{analytics_stream_base_url.rstrip('/')}/{camera_id}/index.m3u8"
        )

        return cls(
            db_dsn=db_dsn,
            camera_id=camera_id,
            rtsp_url=rtsp_url,
            pos_id=pos_id,
            audio_source=os.getenv("AUDIO_SOURCE") or rtsp_url,
            yolo_model_path=os.getenv("YOLO_MODEL_PATH", "yolov11x.pt"),
            yolo_device=os.getenv("YOLO_DEVICE", "cpu"),
            fps=_int_env("VIDEO_FPS", 25),
            inference_stride=_int_env("INFERENCE_STRIDE", 5),
            buffer_seconds=_int_env("BUFFER_SECONDS", 120),
            jpeg_quality=_int_env("JPEG_QUALITY", 85),
            inference_queue_size=_int_env("INFERENCE_QUEUE_SIZE", 64),
            clip_output_dir=Path(os.getenv("CLIP_OUTPUT_DIR", "clips")),
            spool_dir=Path(os.getenv("SPOOL_DIR", "spool")),
            task_poll_interval_s=_float_env("TASK_POLL_INTERVAL_S", 1.0),
            clip_response_timeout_s=_float_env("CLIP_RESPONSE_TIMEOUT_S", 15.0),
            yolo_profile_interval_s=_float_env("YOLO_PROFILE_INTERVAL_S", 10.0),
            vad_aggressiveness=_int_env("VAD_AGGRESSIVENESS", 3),
            audio_sample_rate=_int_env("AUDIO_SAMPLE_RATE", 16000),
            audio_frame_ms=_int_env("AUDIO_FRAME_MS", 30),
            audio_start_window_ms=_int_env("AUDIO_START_WINDOW_MS", 300),
            audio_start_ratio=_float_env("AUDIO_START_RATIO", 0.9),
            audio_end_silence_ms=_int_env("AUDIO_END_SILENCE_MS", 1500),
            audio_max_segment_ms=_int_env("AUDIO_MAX_SEGMENT_MS", 30000),
            whisper_model_name=os.getenv("WHISPER_MODEL_NAME", "small"),
            whisper_device=os.getenv("WHISPER_DEVICE", "cpu"),
            whisper_compute_type=os.getenv("WHISPER_COMPUTE_TYPE", "int8"),
            analytics_api_base_url=os.getenv("ANALYTICS_API_BASE_URL", ""),
            analytics_api_key=os.getenv("ANALYTICS_API_KEY", ""),
            analytics_stream_url=analytics_stream_url,
            analytics_stream_type=os.getenv("ANALYTICS_STREAM_TYPE", "hls"),
            analytics_register_timeout_s=_float_env("ANALYTICS_REGISTER_TIMEOUT_S", 5.0),
        )
