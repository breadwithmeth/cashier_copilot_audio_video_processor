from __future__ import annotations

import os
import signal
import subprocess
import tempfile
import time
import wave
from datetime import datetime
from pathlib import Path

import mlx_whisper


RTSP_URL = os.getenv(
    "RTSP_URL",
    "rtsp://100.96.0.32:8554/"
    "microphone",
)

WHISPER_MODEL = os.getenv(
    "WHISPER_MODEL",
    "mlx-community/whisper-small-mlx",
)

LANGUAGE = os.getenv("WHISPER_LANGUAGE", "ru")

SAMPLE_RATE = 16000
CHANNELS = 1
SAMPLE_WIDTH = 2  # PCM signed 16-bit
CHUNK_SECONDS = int(os.getenv("CHUNK_SECONDS", "60"))

BYTES_PER_SECOND = SAMPLE_RATE * CHANNELS * SAMPLE_WIDTH
CHUNK_SIZE_BYTES = BYTES_PER_SECOND * CHUNK_SECONDS

TRANSCRIPT_FILE = Path("transcript.txt")

running = True


def stop_program(signum: int, frame: object) -> None:
    global running
    running = False
    print("\nОстановка программы...")


def start_ffmpeg() -> subprocess.Popen[bytes]:
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "warning",

        # Подключение к RTSP через TCP
        "-rtsp_transport",
        "tcp",

        "-i",
        RTSP_URL,

        # Берем только первую аудиодорожку
        "-map",
        "0:a:0",

        # Без видео
        "-vn",

        # Преобразуем звук в PCM mono 16 kHz
        "-ac",
        str(CHANNELS),

        "-ar",
        str(SAMPLE_RATE),

        "-acodec",
        "pcm_s16le",

        # Выводим сырые аудиоданные в stdout
        "-f",
        "s16le",

        "pipe:1",
    ]

    print("Подключение к RTSP...")
    print(mask_password(RTSP_URL))

    return subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0,
    )


def mask_password(url: str) -> str:
    if "://" not in url or "@" not in url:
        return url

    protocol, rest = url.split("://", 1)
    auth, address = rest.split("@", 1)

    if ":" not in auth:
        return url

    username, _ = auth.split(":", 1)

    return f"{protocol}://{username}:***@{address}"


def save_pcm_to_wav(pcm_data: bytes, wav_path: Path) -> None:
    with wave.open(str(wav_path), "wb") as wav_file:
        wav_file.setnchannels(CHANNELS)
        wav_file.setsampwidth(SAMPLE_WIDTH)
        wav_file.setframerate(SAMPLE_RATE)
        wav_file.writeframes(pcm_data)


def transcribe_audio(wav_path: Path) -> str:
    result = mlx_whisper.transcribe(
        str(wav_path),
        path_or_hf_repo=WHISPER_MODEL,
        language=LANGUAGE,
        task="transcribe",
        temperature=0.0,
        condition_on_previous_text=False,
        verbose=False,
    )

    return str(result.get("text", "")).strip()


def save_transcript(text: str) -> None:
    timestamp = datetime.now().astimezone().isoformat(
        timespec="seconds"
    )

    line = f"[{timestamp}] {text}"

    print(line)

    with TRANSCRIPT_FILE.open("a", encoding="utf-8") as file:
        file.write(line + "\n")


def read_exactly(
    stream,
    required_bytes: int,
) -> bytes:
    buffer = bytearray()

    while running and len(buffer) < required_bytes:
        chunk = stream.read(required_bytes - len(buffer))

        if not chunk:
            break

        buffer.extend(chunk)

    return bytes(buffer)


def process_stream(process: subprocess.Popen[bytes]) -> None:
    if process.stdout is None:
        raise RuntimeError("FFmpeg stdout недоступен")

    print(
        f"Начинаю распознавание фрагментами "
        f"по {CHUNK_SECONDS} секунд"
    )
    print(f"Модель: {WHISPER_MODEL}")
    print(f"Текст сохраняется в: {TRANSCRIPT_FILE.resolve()}")
    print("Для остановки нажмите Ctrl+C\n")

    while running:
        pcm_data = read_exactly(
            process.stdout,
            CHUNK_SIZE_BYTES,
        )

        if not pcm_data:
            raise RuntimeError(
                "RTSP-поток завершился или аудиодорожка недоступна"
            )

        # Не отправляем слишком короткий последний фрагмент
        minimum_size = BYTES_PER_SECOND * 2

        if len(pcm_data) < minimum_size:
            continue

        with tempfile.NamedTemporaryFile(
            suffix=".wav",
            delete=False,
        ) as temporary_file:
            wav_path = Path(temporary_file.name)

        try:
            save_pcm_to_wav(pcm_data, wav_path)

            started_at = time.perf_counter()

            text = transcribe_audio(wav_path)

            processing_time = time.perf_counter() - started_at

            if text:
                save_transcript(text)
                print(
                    f"Распознавание заняло "
                    f"{processing_time:.2f} сек.\n"
                )
            else:
                print(
                    f"Речь не обнаружена. "
                    f"Обработка: {processing_time:.2f} сек.\n"
                )

        except Exception as error:
            print(f"Ошибка распознавания: {error}")

        finally:
            wav_path.unlink(missing_ok=True)


def stop_ffmpeg(process: subprocess.Popen[bytes] | None) -> None:
    if process is None:
        return

    if process.poll() is not None:
        return

    process.terminate()

    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=3)


def main() -> None:
    signal.signal(signal.SIGINT, stop_program)
    signal.signal(signal.SIGTERM, stop_program)

    while running:
        process: subprocess.Popen[bytes] | None = None

        try:
            process = start_ffmpeg()

            # Небольшая пауза, чтобы сразу выявить ошибку подключения
            time.sleep(2)

            if process.poll() is not None:
                error_message = ""

                if process.stderr is not None:
                    error_message = (
                        process.stderr.read()
                        .decode("utf-8", errors="replace")
                    )

                raise RuntimeError(
                    f"FFmpeg завершился:\n{error_message}"
                )

            process_stream(process)

        except FileNotFoundError:
            print(
                "FFmpeg не найден. Установите его командой:\n"
                "brew install ffmpeg"
            )
            return

        except Exception as error:
            if not running:
                break

            print(f"Ошибка потока: {error}")
            print("Повторное подключение через 5 секунд...\n")
            time.sleep(5)

        finally:
            stop_ffmpeg(process)


if __name__ == "__main__":
    main()
