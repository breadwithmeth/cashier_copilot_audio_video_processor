from pathlib import Path
import os

# ===========================
# MODELS
# ===========================

SCAN_MODEL_PATH = Path("weights/yolov8s-worldv2.pt")
SCAN_WORLD_PROMPTS = [
    "retail product",
    "product package",
    "bottle",
    "can",
    "box",
    "bag",
    "fruit",
    "vegetable",
]
POSE_MODEL_PATH = "yolo11n-pose.pt"

# ===========================
# DETECTION
# ===========================

TARGET_FPS = 15

SCAN_CONFIDENCE = 0.2
PERSON_CONFIDENCE = 0.4
POSE_KEYPOINT_CONFIDENCE = 0.3

# Full HD inference resolution (height, width).
SCAN_IMAGE_SIZE = (1088, 1920)
POSE_IMAGE_SIZE = (1088, 1920)

# Через сколько секунд считать,
# что клиент действительно стоит у кассы

CUSTOMER_TIMEOUT = 3.0

# Через сколько секунд считать,
# что кассир действительно стоит за кассой

CASHIER_TIMEOUT = 2.0

# ===========================
# SPEECH RECOGNITION
# ===========================

# Set to False to run video analytics without RTSP audio and STT.
SPEECH_RECOGNITION_ENABLED = False
WHISPER_MODEL = os.getenv(
    "WHISPER_MODEL",
    "mlx-community/whisper-large-v3-turbo-q4",
)
WHISPER_LANGUAGE = os.getenv("WHISPER_LANGUAGE", "ru")
WHISPER_BACKEND = os.getenv("WHISPER_BACKEND", "gigaam")
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
GIGAAM_MODEL = os.getenv("GIGAAM_MODEL", "v3_e2e_rnnt")
GIGAAM_DEVICE = os.getenv("GIGAAM_DEVICE", "auto")
SENSEVOICE_MODEL = os.getenv(
    "SENSEVOICE_MODEL",
    "FunAudioLLM/SenseVoiceSmall",
)
TRANSCRIPTS_DIR = Path(os.getenv("TRANSCRIPTS_DIR", "transcripts"))

# ===========================
# DATASET COLLECTION
# ===========================

DATASET_COLLECTION_ENABLED = os.getenv("DATASET_COLLECTION_ENABLED", "1") == "1"
DATASET_DIR = Path(os.getenv("DATASET_DIR", "dataset_output"))
FLORENCE_MODEL = os.getenv(
    "FLORENCE_MODEL",
    "microsoft/Florence-2-base-ft",
)
DATASET_TRACK_TIMEOUT = float(os.getenv("DATASET_TRACK_TIMEOUT", "2.0"))

# ===========================
# CAMERAS
# ===========================

STREAMS = {
    "cam10": {
        "url": "rtsp://admin:LeWfBvc4%21@127.0.0.1:8554/cam/realmonitor?channel=4&subtype=0",
        "audio_url": "rtsp://100.96.0.32:8554/microphone",

        "scan_roi": (
            (1250, 100),
            (1750, 100),
            (1600, 1100),
            (1100, 1100),
        ),

        "customer_roi": (
            1800,
            0,
            3200,
            900,
        ),

        "cashier_roi": (
            0,
            0,
            1200,
            1300,
        ),
    },
}
