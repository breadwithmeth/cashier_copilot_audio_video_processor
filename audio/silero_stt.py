"""Silero STT backend for RTSP audio transcription."""

from __future__ import annotations

import tempfile
import wave
from pathlib import Path
from typing import Any

import numpy as np
import torch


class SileroSTTModel:
    """Wrapper for Silero STT model loaded via torch.hub."""

    def __init__(
        self,
        model_name: str = "silero_stt",
        language: str = "ru",
        device: str = "cpu",
    ) -> None:
        self.model_name = model_name
        self.language = language
        self.device = torch.device(device)

        # Load model via torch.hub
        self.model, self.decoder, self.utils = torch.hub.load(
            repo_or_dir="snakers4/silero-models",
            model=model_name,
            language=language,
            device=self.device,
            trust_repo=True,
        )
        self.model.eval()

        # Unpack utils
        (
            self.read_batch,
            self.split_into_batches,
            self.read_audio,
            self.prepare_model_input,
        ) = self.utils

    def transcribe_file(self, audio_path: str) -> list[dict]:
        """Transcribe audio file and return segments with timestamps."""
        # Read and prepare audio
        batch = self.split_into_batches([audio_path], batch_size=1)
        input_tensor = self.prepare_model_input(self.read_batch(batch[0]), device=self.device)

        # Run inference
        with torch.no_grad():
            output = self.model(input_tensor)

        # Decode
        results = []
        for example in output:
            text = self.decoder(example.cpu())
            if text.strip():
                # Silero STT doesn't provide word-level timestamps by default
                # We'll return a single segment for the whole file
                with wave.open(audio_path, "rb") as wav:
                    duration = wav.getnframes() / wav.getframerate()
                results.append({
                    "start": 0.0,
                    "end": round(duration, 3),
                    "text": text.strip(),
                })
        return results

    def transcribe_pcm(self, pcm_bytes: bytes, sample_rate: int = 16000) -> list[dict]:
        """Transcribe raw PCM bytes (16-bit mono)."""
        # Convert to float32 numpy array
        samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0

        # Save to temporary WAV file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = Path(tmp.name)
            with wave.open(str(tmp_path), "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(sample_rate)
                wav.writeframes(pcm_bytes)

        try:
            return self.transcribe_file(str(tmp_path))
        finally:
            tmp_path.unlink(missing_ok=True)


def load_silero_stt(
    model_name: str = "silero_stt",
    language: str = "ru",
    device: str = "cpu",
) -> SileroSTTModel:
    """Factory function to load Silero STT model."""
    return SileroSTTModel(model_name=model_name, language=language, device=device)