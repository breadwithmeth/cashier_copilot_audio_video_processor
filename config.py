from pathlib import Path

# ===========================
# MODELS
# ===========================

SCAN_MODEL_PATH = Path("weights/best.pt")
POSE_MODEL_PATH = "yolo11n-pose.pt"

# ===========================
# DETECTION
# ===========================

TARGET_FPS = 5

SCAN_CONFIDENCE = 0.5
PERSON_CONFIDENCE = 0.4

SCAN_IMAGE_SIZE = 640
POSE_IMAGE_SIZE = 640

# Через сколько секунд считать,
# что клиент действительно стоит у кассы

CUSTOMER_TIMEOUT = 3.0

# Через сколько секунд считать,
# что кассир действительно стоит за кассой

CASHIER_TIMEOUT = 2.0

# ===========================
# CAMERAS
# ===========================

STREAMS = {
    "cam10": {
        "url": "rtsp://admin:LeWfBvc4%21@127.0.0.1:8554/cam/realmonitor?channel=10&subtype=1",

        "scan_roi": (
            200,
            100,
            470,
            580,
        ),

        "customer_roi": (
            470,
            0,
            960,
            540,
        ),

        "cashier_roi": (
            0,
            0,
            360,
            580,
        ),
    },
}