from __future__ import annotations

import logging
import multiprocessing
import subprocess
import sys
import time
from collections import deque
from pathlib import Path
from typing import Optional

from .db import DatabaseClient, DurableEventStore, SpeechTranscript
from .video import utc_timestamp_ms

logger = logging.getLogger(__name__)


class AudioSTTWorker(multiprocessing.Process):
    def __init__(
        self,
        camera_id: str,
        rtsp_url: str,
        pos_id: str,
        db_pool: DatabaseClient | str,
        stop_event: Optional[multiprocessing.Event] = None,
        spool_dir: Path | str = "spool",
        sample_rate: int = 16000,
        frame_ms: int = 30,
        start_window_ms: int = 300,
        start_ratio: float = 0.9,
        end_silence_ms: int = 1500,
        max_segment_ms: int = 30000,
        vad_aggressiveness: int = 3,
        whisper_model_name: str = "small",
        whisper_device: str = "cuda",
        whisper_compute_type: str = "float16",
    ) -> None:
        super().__init__(name=f"AudioSttProcess-{camera_id}")
        self.camera_id = camera_id
        self.rtsp_url = rtsp_url
        self.pos_id = pos_id
        self.db_pool = db_pool
        self.stop_event = stop_event
        self.spool_dir = Path(spool_dir)
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.start_window_ms = start_window_ms
        self.start_ratio = start_ratio
        self.end_silence_ms = end_silence_ms
        self.max_segment_ms = max_segment_ms
        self.vad_aggressiveness = vad_aggressiveness
        self.whisper_model_name = whisper_model_name
        self.whisper_device = whisper_device
        self.whisper_compute_type = whisper_compute_type
        self.whisper = None

    def run(self) -> None:
        import webrtcvad
        from faster_whisper import WhisperModel

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(processName)s %(name)s: %(message)s",
        )
        self.whisper = WhisperModel(
            self.whisper_model_name,
            device=self.whisper_device,
            compute_type=self.whisper_compute_type,
        )
        vad = webrtcvad.Vad(self.vad_aggressiveness)
        db = self.db_pool if isinstance(self.db_pool, DatabaseClient) else DatabaseClient(self.db_pool)
        store = DurableEventStore(db, self.spool_dir)

        try:
            while self.stop_event is None or not self.stop_event.is_set():
                proc = self._start_ffmpeg()
                try:
                    self._consume_audio(proc, vad, store)
                except Exception:
                    logger.exception("Audio capture/transcription loop failed camera=%s", self.camera_id)
                finally:
                    self._stop_ffmpeg(proc)

                if self.stop_event is not None and self.stop_event.is_set():
                    break
                logger.warning("Audio capture stopped camera=%s; reconnecting in 5s", self.camera_id)
                time.sleep(5)
        finally:
            db.close()

    def _start_ffmpeg(self) -> subprocess.Popen:
        input_args = self._ffmpeg_input_args()
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "warning",
            *input_args,
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            str(self.sample_rate),
            "-ac",
            "1",
            "-f",
            "wav",
            "pipe:1",
        ]
        logger.info("Starting ffmpeg audio capture camera=%s", self.camera_id)
        return subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=0,
        )

    def _ffmpeg_input_args(self) -> list[str]:
        if not self.rtsp_url.startswith("mic:"):
            return ["-rtsp_transport", "tcp", "-i", self.rtsp_url]

        device = self.rtsp_url.removeprefix("mic:") or "default"
        if sys.platform == "darwin":
            return ["-f", "avfoundation", "-i", device]
        if sys.platform.startswith("linux"):
            return ["-f", "pulse", "-i", device]
        return ["-i", device]

    def _consume_audio(
        self,
        proc: subprocess.Popen,
        vad,
        store: DurableEventStore,
    ) -> None:
        if proc.stdout is None:
            raise RuntimeError("ffmpeg stdout is not available")

        bytes_per_frame = int(self.sample_rate * 2 * self.frame_ms / 1000)
        start_window_frames = max(1, self.start_window_ms // self.frame_ms)
        end_silence_frames = max(1, self.end_silence_ms // self.frame_ms)
        max_segment_frames = max(1, self.max_segment_ms // self.frame_ms)

        recent_voice: deque[bool] = deque(maxlen=start_window_frames)
        pre_roll: deque[bytes] = deque(maxlen=start_window_frames)
        segment_frames: list[bytes] = []
        in_segment = False
        silence_frames = 0
        segment_start_ms = 0

        self._skip_wav_header(proc.stdout)

        while self.stop_event is None or not self.stop_event.is_set():
            frame = proc.stdout.read(bytes_per_frame)
            if len(frame) == 0:
                break
            if len(frame) < bytes_per_frame:
                continue

            voiced = vad.is_speech(frame, self.sample_rate)
            recent_voice.append(voiced)
            pre_roll.append(frame)

            if not in_segment:
                if len(recent_voice) == start_window_frames and self._voice_ratio(recent_voice) >= self.start_ratio:
                    in_segment = True
                    segment_start_ms = utc_timestamp_ms() - self.start_window_ms
                    segment_frames = list(pre_roll)
                    silence_frames = 0
                continue

            segment_frames.append(frame)
            if voiced:
                silence_frames = 0
            else:
                silence_frames += 1

            if silence_frames >= end_silence_frames or len(segment_frames) >= max_segment_frames:
                self._transcribe_segment(segment_frames, segment_start_ms, store)
                in_segment = False
                segment_frames = []
                silence_frames = 0
                recent_voice.clear()
                pre_roll.clear()

        if in_segment and segment_frames:
            self._transcribe_segment(segment_frames, segment_start_ms, store)

    @staticmethod
    def _voice_ratio(flags: deque[bool]) -> float:
        if not flags:
            return 0.0
        return sum(1 for value in flags if value) / len(flags)

    @staticmethod
    def _skip_wav_header(stdout) -> None:
        header = stdout.read(44)
        if len(header) < 44:
            raise RuntimeError("ffmpeg wav header is incomplete")

    def _transcribe_segment(
        self,
        frames: list[bytes],
        timestamp_ms: int,
        store: DurableEventStore,
    ) -> None:
        if self.whisper is None:
            raise RuntimeError("Whisper model is not initialized")

        import numpy as np

        pcm = b"".join(frames)
        duration_ms = int(len(pcm) / 2 / self.sample_rate * 1000)
        audio = np.frombuffer(pcm, np.int16).astype(np.float32) / 32768.0

        segments, info = self.whisper.transcribe(audio, language="ru", beam_size=5)
        texts = [segment.text.strip() for segment in segments if segment.text.strip()]
        full_text = " ".join(texts).strip()
        if not full_text:
            return

        confidence = getattr(info, "language_probability", None)
        transcript = SpeechTranscript(
            pos_id=self.pos_id,
            transcript=full_text,
            timestamp_ms=timestamp_ms,
            duration_ms=duration_ms,
            confidence=float(confidence) if confidence is not None else None,
            model_name="Faster-Whisper",
            weights_version=self.whisper_model_name,
        )
        store.insert_speech_transcript(transcript)
        store.replay_pending(max_events=25)
        logger.info(
            "Stored transcript camera=%s duration_ms=%s text_len=%s",
            self.camera_id,
            duration_ms,
            len(full_text),
        )

    @staticmethod
    def _stop_ffmpeg(proc: subprocess.Popen) -> None:
        if proc.poll() is not None:
            return
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
