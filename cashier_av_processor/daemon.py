from __future__ import annotations

import logging
import multiprocessing
import signal
import sys
import time

from .audio_stt import AudioSTTWorker
from .clip_exporter import ClipExporter
from .config import AppConfig
from .detector import YoloInferenceProcess
from .video import VideoGrabberProcess

logger = logging.getLogger(__name__)


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(processName)s %(name)s: %(message)s",
    )


def build_processes(config: AppConfig, stop_event: multiprocessing.Event) -> list[multiprocessing.Process]:
    inference_queue: multiprocessing.Queue = multiprocessing.Queue(maxsize=config.inference_queue_size)
    clip_request_queue: multiprocessing.Queue = multiprocessing.Queue(maxsize=8)
    clip_response_queue: multiprocessing.Queue = multiprocessing.Queue(maxsize=8)

    return [
        VideoGrabberProcess(
            camera_id=config.camera_id,
            rtsp_url=config.rtsp_url,
            inference_queue=inference_queue,
            clip_request_queue=clip_request_queue,
            clip_response_queue=clip_response_queue,
            stop_event=stop_event,
            fps=config.fps,
            buffer_seconds=config.buffer_seconds,
            jpeg_quality=config.jpeg_quality,
            inference_stride=config.inference_stride,
        ),
        YoloInferenceProcess(
            inference_queue=inference_queue,
            db_dsn=config.db_dsn,
            stop_event=stop_event,
            model_path=config.yolo_model_path,
            spool_dir=config.spool_dir,
            profile_interval_s=config.yolo_profile_interval_s,
        ),
        ClipExporter(
            db_pool=config.db_dsn,
            video_buffers={},
            clip_request_queue=clip_request_queue,
            clip_response_queue=clip_response_queue,
            stop_event=stop_event,
            output_dir=config.clip_output_dir,
            spool_dir=config.spool_dir,
            poll_interval_s=config.task_poll_interval_s,
            response_timeout_s=config.clip_response_timeout_s,
        ),
        AudioSTTWorker(
            camera_id=config.camera_id,
            rtsp_url=config.audio_source,
            pos_id=config.pos_id,
            db_pool=config.db_dsn,
            stop_event=stop_event,
            spool_dir=config.spool_dir,
            sample_rate=config.audio_sample_rate,
            frame_ms=config.audio_frame_ms,
            start_window_ms=config.audio_start_window_ms,
            start_ratio=config.audio_start_ratio,
            end_silence_ms=config.audio_end_silence_ms,
            max_segment_ms=config.audio_max_segment_ms,
            vad_aggressiveness=config.vad_aggressiveness,
            whisper_model_name=config.whisper_model_name,
            whisper_device=config.whisper_device,
            whisper_compute_type=config.whisper_compute_type,
        ),
    ]


def run_daemon(config: AppConfig) -> int:
    stop_event = multiprocessing.Event()

    def request_shutdown(signum, _frame) -> None:
        logger.info("Received signal=%s; stopping workers", signum)
        stop_event.set()

    signal.signal(signal.SIGINT, request_shutdown)
    signal.signal(signal.SIGTERM, request_shutdown)

    config.clip_output_dir.mkdir(parents=True, exist_ok=True)
    config.spool_dir.mkdir(parents=True, exist_ok=True)

    processes = build_processes(config, stop_event)
    for process in processes:
        process.start()
        logger.info("Started %s pid=%s", process.name, process.pid)

    try:
        while not stop_event.is_set():
            for process in processes:
                if process.exitcode is not None:
                    logger.error("%s exited unexpectedly exitcode=%s", process.name, process.exitcode)
                    stop_event.set()
                    break
            time.sleep(1.0)
    finally:
        stop_event.set()
        for process in processes:
            process.join(timeout=10)
        for process in processes:
            if process.is_alive():
                logger.warning("Terminating %s pid=%s", process.name, process.pid)
                process.terminate()
        for process in processes:
            process.join(timeout=5)

    return 0


def main() -> int:
    configure_logging()
    try:
        config = AppConfig.from_env()
    except Exception as exc:
        logger.error("Invalid configuration: %s", exc)
        return 2
    return run_daemon(config)


if __name__ == "__main__":
    sys.exit(main())
