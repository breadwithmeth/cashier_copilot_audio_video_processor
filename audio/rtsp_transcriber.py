from __future__ import annotations

import json
import re
import queue
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
import wave
from collections import deque
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import numpy as np

from config import (
    ANALYTICS_API_BASE_URL,
    ANALYTICS_API_KEY,
    ANALYTICS_AUDIO_SOURCE,
    ANALYTICS_REGISTER_CODE,
    ANALYTICS_SEND_TIMEOUT,
    ANALYTICS_STORE_CODE,
)


class RTSPVisitTranscriber:
    """Capture RTSP audio and transcribe one file per customer visit."""

    sample_rate = 16_000
    sample_width = 2
    subtitle_chunk_seconds = 6
    subtitle_overlap_seconds = 1
    minimum_audio_rms = 180

    def __init__(self, name: str, url: str, output_dir: Path, model: str,
                 language: str = "ru", prebuffer_seconds: float = 5.0,
                 backend: str = "auto", compute_type: str = "int8",
                 device: str = "auto"):
        self.name = name
        self.url = url
        self.output_dir = output_dir
        self.model_name = model
        self.language = language
        self.backend = ("mlx" if backend == "auto" and sys.platform == "darwin"
                        else "faster-whisper" if backend == "auto" else backend)
        self.compute_type = compute_type
        self.device = device
        self._prebuffer = deque(maxlen=max(1, int(prebuffer_seconds * 10)))
        self._lock = threading.Lock()
        self._session: dict | None = None
        self._jobs: queue.Queue = queue.Queue()
        self._running = True
        self._process: subprocess.Popen | None = None
        self._subtitle = ""
        self._subtitle_until = 0.0
        self._subtitle_history = deque()
        self._subtitle_job_pending = False
        self._capture_thread = threading.Thread(target=self._capture, daemon=True)
        self._recognize_thread = threading.Thread(target=self._recognize, daemon=True)
        self._capture_thread.start()
        self._recognize_thread.start()
        print(f"[{self.name}] Speech backend: {self.backend}")

    def start_visit(self, started_at: float) -> None:
        with self._lock:
            if self._session is not None:
                return
            self._subtitle = ""
            self._subtitle_until = 0.0
            self._subtitle_history.clear()
            # The detector confirms presence after a timeout. Retain only audio
            # captured since the first frame in which the customer was seen.
            buffered = [data for captured_at, data in self._prebuffer
                        if captured_at >= started_at]
            self._session = {
                "id": uuid4().hex,
                "started_at": started_at,
                "pcm": bytearray().join(buffered),
                # Start live recognition with the same prebuffer so speech made
                # during customer-presence confirmation is not lost.
                "subtitle_pcm": bytearray().join(buffered),
            }
        print(f"[{self.name}] Speech session started")

    def end_visit(self, ended_at: float) -> None:
        with self._lock:
            session, self._session = self._session, None
        if session is None:
            return
        session["ended_at"] = ended_at
        self._jobs.put(("final", session))
        print(f"[{self.name}] Speech session queued for recognition")

    def stop(self) -> None:
        self._running = False
        self.end_visit(time.time())
        process = self._process
        if process is not None and process.poll() is None:
            process.terminate()
        self._jobs.put(None)
        self._capture_thread.join(timeout=3)
        # Let a visit queued during shutdown finish saving before Python exits.
        self._recognize_thread.join(timeout=60)

    def get_subtitle(self) -> str:
        with self._lock:
            now = time.monotonic()
            while (self._subtitle_history and
                   now - self._subtitle_history[0][0] > 10):
                self._subtitle_history.popleft()
            return "\n".join(text for _, text in self._subtitle_history)

    def _capture(self) -> None:
        chunk_size = self.sample_rate * self.sample_width // 10
        while self._running:
            command = ["ffmpeg", "-hide_banner", "-loglevel", "error",
                       "-fflags", "+genpts+discardcorrupt",
                       "-rtsp_transport", "tcp", "-i", self.url,
                       "-map", "0:a:0", "-vn", "-ac", "1", "-ar",
                       str(self.sample_rate), "-af",
                       "aresample=async=1:first_pts=0,asetpts=N/SR/TB",
                       "-acodec", "pcm_s16le",
                       "-f", "s16le", "pipe:1"]
            try:
                self._process = subprocess.Popen(command, stdout=subprocess.PIPE)
                while self._running and self._process.stdout is not None:
                    data = self._process.stdout.read(chunk_size)
                    if not data:
                        break
                    now = time.time()
                    with self._lock:
                        self._prebuffer.append((now, data))
                        if self._session is not None:
                            self._session["pcm"].extend(data)
                            preview = self._session["subtitle_pcm"]
                            preview.extend(data)
                            required = (self.sample_rate * self.sample_width *
                                        self.subtitle_chunk_seconds)
                            if (len(preview) >= required and
                                    not self._subtitle_job_pending):
                                # Always recognize the newest window. Keeping a
                                # short overlap avoids cutting words at borders.
                                self._jobs.put(("subtitle", bytes(preview[-required:])))
                                self._subtitle_job_pending = True
                                overlap = (self.sample_rate * self.sample_width *
                                           self.subtitle_overlap_seconds)
                                del preview[:-overlap]
            except (OSError, subprocess.SubprocessError) as error:
                print(f"[{self.name}] RTSP audio error: {error}")
            finally:
                if self._process is not None and self._process.poll() is None:
                    self._process.terminate()
            if self._running:
                time.sleep(2)

    def _recognize(self) -> None:
        model = None
        while True:
            job = self._jobs.get()
            if job is None:
                return
            try:
                if model is None and self.backend == "faster-whisper":
                    from faster_whisper import WhisperModel
                    model = WhisperModel(self.model_name,
                                         compute_type=self.compute_type)
                elif model is None and self.backend == "sensevoice":
                    from funasr import AutoModel
                    model = AutoModel(
                        model=self.model_name,
                        hub="hf",
                        trust_remote_code=True,
                        device="cpu",
                        disable_update=True,
                    )
                elif model is None and self.backend == "gigaam":
                    import torch
                    import gigaam
                    device = self.device
                    if device == "auto":
                        device = "mps" if torch.backends.mps.is_available() else "cpu"
                    model = gigaam.load_model(
                        self.model_name,
                        device=device,
                        fp16_encoder=device != "cpu",
                    )
                    print(f"[{self.name}] GigaAM loaded on {device}")
                kind, payload = job
                if kind == "subtitle":
                    try:
                        self._recognize_subtitle(model, payload)
                    finally:
                        with self._lock:
                            self._subtitle_job_pending = False
                else:
                    self._save_session(model, payload)
            except Exception as error:
                print(f"[{self.name}] Speech recognition error: {error}")

    def _recognize_subtitle(self, model, pcm: bytes) -> None:
        samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
        rms = float(np.sqrt(np.mean(samples * samples))) if samples.size else 0.0
        if rms < self.minimum_audio_rms:
            return
        with tempfile.NamedTemporaryFile(suffix=".wav") as temporary:
            with wave.open(temporary.name, "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(self.sample_width)
                wav.setframerate(self.sample_rate)
                wav.writeframes(pcm)
            segments = self._transcribe(model, temporary.name)
            text = " ".join(s["text"] for s in segments)
        if text:
            with self._lock:
                self._subtitle = text
                self._subtitle_until = time.monotonic() + 12
                self._subtitle_history.append((time.monotonic(), text))

    def _save_session(self, model, session: dict) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        stem = datetime.fromtimestamp(session["started_at"]).strftime("%Y%m%d_%H%M%S")
        wav_path = self.output_dir / f"{self.name}_{stem}_{session['id'][:8]}.wav"
        with wave.open(str(wav_path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(self.sample_width)
            wav.setframerate(self.sample_rate)
            wav.writeframes(session["pcm"])

        transcript = self._transcribe(model, str(wav_path))
        result = {
            "camera": self.name,
            "visit_id": session["id"],
            "started_at": datetime.fromtimestamp(session["started_at"]).astimezone().isoformat(),
            "ended_at": datetime.fromtimestamp(session["ended_at"]).astimezone().isoformat(),
            "duration": round(session["ended_at"] - session["started_at"], 3),
            "timestamps_relative_to": "customer_arrived",
            "segments": transcript,
            "audio_file": wav_path.name,
        }
        json_path = wav_path.with_suffix(".json")
        json_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

        delivery = self._send_speech_event(result)
        result["analytics_delivery"] = delivery
        json_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[{self.name}] Transcript saved: {json_path}")

    def _send_speech_event(self, result: dict) -> dict:
        text = self._segments_text(result["segments"])
        if not text:
            print(f"[{self.name}] Speech event skipped: empty transcript")
            return {"status": "skipped", "reason": "empty_transcript"}
        if not ANALYTICS_API_BASE_URL or not ANALYTICS_API_KEY:
            print(f"[{self.name}] Speech event skipped: analytics API is not configured")
            return {"status": "skipped", "reason": "analytics_api_not_configured"}

        started_at_ms = int(datetime.fromisoformat(result["started_at"]).timestamp() * 1000)
        external_event_id = f"{self.name}-audio-{started_at_ms}-{result['visit_id'][:8]}"
        payload = {
            "externalEventId": external_event_id,
            "idempotencyKey": external_event_id,
            "cameraCode": self.name,
            "eventType": "SPEECH_RECOGNIZED",
            "source": self.backend,
            "occurredAt": result["started_at"],
            "startedAt": result["started_at"],
            "endedAt": result["ended_at"],
            "speakerType": "UNKNOWN",
            "language": self.language,
            "text": text,
            "audioSource": ANALYTICS_AUDIO_SOURCE,
            "correlationId": result["visit_id"],
            "payload": {
                "visitId": result["visit_id"],
                "duration": result["duration"],
                "segments": result["segments"],
                "audioFile": result["audio_file"],
                "timestampsRelativeTo": result["timestamps_relative_to"],
            },
        }
        if ANALYTICS_STORE_CODE:
            payload["storeCode"] = ANALYTICS_STORE_CODE
            payload["payload"]["storeCode"] = ANALYTICS_STORE_CODE
        if ANALYTICS_REGISTER_CODE:
            payload["registerCode"] = ANALYTICS_REGISTER_CODE

        url = f"{ANALYTICS_API_BASE_URL}/analytics/audio/events"
        request = urllib.request.Request(
            url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "x-api-key": ANALYTICS_API_KEY,
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(
                request,
                timeout=ANALYTICS_SEND_TIMEOUT,
            ) as response:
                response_body = response.read().decode("utf-8", errors="replace")
            print(f"[{self.name}] Speech event sent: {external_event_id}")
            return {
                "status": "sent",
                "endpoint": url,
                "externalEventId": external_event_id,
                "statusCode": response.status,
                "response": response_body[:1000],
            }
        except urllib.error.HTTPError as error:
            body = error.read().decode("utf-8", errors="replace")
            print(f"[{self.name}] Speech event HTTP error {error.code}: {body[:300]}")
            return {
                "status": "failed",
                "endpoint": url,
                "externalEventId": external_event_id,
                "statusCode": error.code,
                "error": body[:1000],
            }
        except (OSError, urllib.error.URLError) as error:
            print(f"[{self.name}] Speech event send error: {error}")
            return {
                "status": "failed",
                "endpoint": url,
                "externalEventId": external_event_id,
                "error": str(error),
            }

    @staticmethod
    def _segments_text(segments: list[dict]) -> str:
        return " ".join(
            str(segment.get("text", "")).strip()
            for segment in segments
            if str(segment.get("text", "")).strip()
        )

    def _transcribe(self, model, audio_path: str) -> list[dict]:
        if self.backend == "gigaam":
            return self._transcribe_gigaam(model, audio_path)

        if self.backend == "sensevoice":
            from funasr.utils.postprocess_utils import rich_transcription_postprocess
            result = model.generate(
                input=audio_path,
                cache={},
                language="auto",
                use_itn=True,
                batch_size_s=60,
            )
            if not result:
                return []
            text = rich_transcription_postprocess(
                str(result[0].get("text", ""))).strip()
            if not text:
                return []
            with wave.open(audio_path, "rb") as wav:
                duration = wav.getnframes() / wav.getframerate()
            return [{"start": 0.0, "end": round(duration, 3), "text": text}]

        if self.backend == "mlx":
            import mlx_whisper
            model_name = self.model_name
            # The similarly named `-8bit` repository uses mlx-audio-plus
            # safetensors and cannot be loaded by mlx-whisper (which expects
            # weights.npz). Keep old environment configuration working.
            if model_name == "mlx-community/whisper-large-v3-turbo-8bit":
                model_name = "mlx-community/whisper-large-v3-turbo-q4"
            if "/" not in model_name:
                model_name = f"mlx-community/whisper-{model_name}-mlx"
            result = mlx_whisper.transcribe(
                audio_path, path_or_hf_repo=model_name,
                language=self.language, task="transcribe", temperature=0.0,
                condition_on_previous_text=False, verbose=False,
                no_speech_threshold=0.6, logprob_threshold=-1.0,
                compression_ratio_threshold=2.4)
            source = result.get("segments", [])
            transcript = []
            for segment in source:
                text = self._clean_text(str(segment.get("text", "")))
                if text:
                    transcript.append({
                        "start": round(float(segment["start"]), 3),
                        "end": round(float(segment["end"]), 3),
                        "text": text,
                    })
            return transcript

        segments, _ = model.transcribe(
            audio_path, language=self.language, vad_filter=True,
            condition_on_previous_text=False)
        return [{"start": round(s.start, 3), "end": round(s.end, 3),
                 "text": s.text.strip()}
                for s in segments if s.text.strip()]

    def _transcribe_gigaam(self, model, audio_path: str) -> list[dict]:
        max_frames = self.sample_rate * 24
        chunks = []
        with wave.open(audio_path, "rb") as source:
            while True:
                pcm = source.readframes(max_frames)
                if not pcm:
                    break
                chunks.append(pcm)

        transcript = []
        offset = 0.0
        for pcm in chunks:
            duration = len(pcm) / (self.sample_rate * self.sample_width)
            with tempfile.NamedTemporaryFile(suffix=".wav") as temporary:
                with wave.open(temporary.name, "wb") as wav_file:
                    wav_file.setnchannels(1)
                    wav_file.setsampwidth(self.sample_width)
                    wav_file.setframerate(self.sample_rate)
                    wav_file.writeframes(pcm)
                result = model.transcribe(temporary.name, word_timestamps=True)

            words = result.words or []
            if words:
                for word in words:
                    text = self._clean_text(str(word.text))
                    if text:
                        transcript.append({
                            "start": round(offset + float(word.start), 3),
                            "end": round(offset + float(word.end), 3),
                            "text": text,
                        })
            else:
                text = self._clean_text(str(result.text))
                if text:
                    transcript.append({
                        "start": round(offset, 3),
                        "end": round(offset + duration, 3),
                        "text": text,
                    })
            offset += duration
        return transcript

    @staticmethod
    def _clean_text(text: str) -> str:
        text = text.strip()
        normalized = re.sub(r"[^а-яa-z0-9]+", " ", text.lower()).strip()
        hallucinations = {
            "продолжение следует",
            "спасибо за просмотр",
            "thank you for watching",
        }
        if any(normalized == phrase or normalized.startswith(phrase + " ")
               for phrase in hallucinations):
            return ""
        subtitle_credit_patterns = (
            r"\bсубтитр(?:ы|ов|ами)?\b.*\b(?:создал|создала|создавал|сделал|автор|редактор)\b",
            r"\b(?:автор|редактор)\b.*\bсубтитр(?:ы|ов|ами)?\b",
            r"\bsubtitles?\b.*\b(?:by|created|edited)\b",
        )
        if any(re.search(pattern, normalized)
               for pattern in subtitle_credit_patterns):
            return ""
        return text
