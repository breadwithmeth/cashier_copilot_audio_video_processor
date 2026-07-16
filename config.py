import sys
from pathlib import Path

# ===========================
# LOCATION / DEVICE IDENTITY
# ===========================

STORE_CODE = "tolstogo-90"
REGISTER_CODE = "register-1"
CAMERA_CODE = "cam10"
CAMERA_NAME = "Checkout camera"

# ===========================
# MODELS
# ===========================

# Scan detector backend:
# - "yolo": trained closed-set detector from SCAN_MODEL_PATH
# - "yolo_world": prompt-based YOLO-World from SCAN_MODEL_PATH
# - "owlv2": prompt-based OWLv2 from SCAN_OWLV2_MODEL
# - "omdet_turbo": prompt-based OmDet-Turbo from SCAN_OMDET_MODEL
# - "smolvlm": image-to-text VLM labeler, no real bbox output
SCAN_BACKEND = "yolo"
SCAN_DEVICE = "auto"

# SCAN_MODEL_PATH = Path("runs/dataset1_detector/dataset1_640/weights/best.pt")
# SCAN_WORLD_PROMPTS = []

# YOLO-World fallback. Uncomment both lines below to switch from the trained
# dataset1 detector back to prompt-based open-vocabulary detection.
#SCAN_MODEL_PATH = Path("weights/yolov8m-worldv2.pt")
SCAN_MODEL_PATH = Path("weights/yolov26_product_detection_v2_best.pt")
# SCAN_WORLD_PROMPTS = ["bottle", "can", "tetra_pak", "pouch", "food", "cigarettes", "receipt", "barcode_scanner", "id_card", "phone", "shopping_bag", "bank_card", "business_card", "basket"]

SCAN_WORLD_PROMPTS = [
    "bottle",
    "milk carton",
    "tin can",
    "energy drink can",
    "glass jar",
    "cardboard box",
    "plastic food tray",
    "plastic container",
    "plastic pouch",
    "vacuum sealed package",
    "paste tube",
    "plastic shopping bag",
    "paper cup",
    "fresh produce",
    "bread",
    "eggs",
    "barcode scanner",
    "payment terminal",
    "paper receipt",
    "credit card",
    "banknote",
    "passport",
    "cigarette",
]

SCAN_OWLV2_MODEL = "google/owlv2-base-patch16-ensemble"
SCAN_OWLV2_CLASSES = [
    {"label": "bottle", "prompt": "plastic bottle or glass drink bottle"},
    {"label": "milk_carton", "prompt": "milk carton or juice carton"},
    {"label": "tin_can", "prompt": "tin can or aluminum drink can"},
    {"label": "energy_drink_can", "prompt": "energy drink can or aluminum beverage can"},
    {"label": "glass_jar", "prompt": "glass jar with food or drink"},
    {"label": "cardboard_box", "prompt": "cardboard product box"},
    {"label": "plastic_food_tray", "prompt": "plastic food tray"},
    {"label": "plastic_container", "prompt": "plastic food container"},
    {"label": "plastic_pouch", "prompt": "plastic pouch package"},
    {"label": "vacuum_sealed_package", "prompt": "vacuum sealed retail package"},
    {"label": "paste_tube", "prompt": "paste tube or squeeze tube"},
    {"label": "plastic_shopping_bag", "prompt": "plastic shopping bag"},
    {"label": "paper_cup", "prompt": "paper cup or disposable cup"},
    {"label": "fresh_produce", "prompt": "fresh fruit or vegetable produce"},
    {"label": "bread", "prompt": "bread or bakery product"},
    {"label": "eggs", "prompt": "eggs or egg carton"},
    {"label": "barcode_scanner", "prompt": "barcode scanner"},
    {"label": "payment_terminal", "prompt": "payment terminal card reader"},
    {"label": "paper_receipt", "prompt": "paper receipt"},
    {"label": "credit_card", "prompt": "credit card or bank card"},
    {"label": "banknote", "prompt": "cash banknote money"},
    {"label": "passport", "prompt": "passport document"},
    {"label": "cigarette", "prompt": "cigarette pack"},
]
SCAN_OWLV2_PROMPTS = [item["prompt"] for item in SCAN_OWLV2_CLASSES]
SCAN_OMDET_MODEL = "omlab/omdet-turbo-swin-tiny-hf"
SCAN_OMDET_CLASSES = [
    {"label": "bottle", "prompt": "plastic bottle or glass drink bottle"},
    {"label": "milk_carton", "prompt": "milk carton or juice carton"},
    {"label": "tin_can", "prompt": "tin can or aluminum drink can"},
    {"label": "energy_drink_can", "prompt": "energy drink can or aluminum beverage can"},
    {"label": "glass_jar", "prompt": "glass jar with food or drink"},
    {"label": "cardboard_box", "prompt": "cardboard product box"},
    {"label": "plastic_food_tray", "prompt": "plastic food tray"},
    {"label": "plastic_container", "prompt": "plastic food container"},
    {"label": "plastic_pouch", "prompt": "plastic pouch package"},
    {"label": "vacuum_sealed_package", "prompt": "vacuum sealed retail package"},
    {"label": "paste_tube", "prompt": "paste tube or squeeze tube"},
    {"label": "plastic_shopping_bag", "prompt": "plastic shopping bag"},
    {"label": "paper_cup", "prompt": "paper cup or disposable cup"},
    {"label": "fresh_produce", "prompt": "fresh fruit or vegetable produce"},
    {"label": "bread", "prompt": "bread or bakery product"},
    {"label": "eggs", "prompt": "eggs or egg carton"},
    {"label": "barcode_scanner", "prompt": "barcode scanner"},
    {"label": "payment_terminal", "prompt": "payment terminal card reader"},
    {"label": "paper_receipt", "prompt": "paper receipt"},
    {"label": "credit_card", "prompt": "credit card or bank card"},
    {"label": "banknote", "prompt": "cash banknote money"},
    {"label": "passport", "prompt": "passport document"},
    {"label": "cigarette", "prompt": "cigarette pack"},
]
SCAN_OMDET_PROMPTS = [item["prompt"] for item in SCAN_OMDET_CLASSES]
SCAN_SMOLVLM_MODEL = "HuggingFaceTB/SmolVLM-500M-Instruct"
SCAN_RT_DETR_V2_MODEL = "nielsr/rtdetr-tray-cart-tuned-strong-20260303-204722"
SCAN_SMOLVLM_INTERVAL_SECONDS = 2.0
SCAN_SMOLVLM_PROMPT = (
    "Look at the cashier scan area image. Describe each visible retail product "
    "or merchandise item in 1-3 words (e.g., 'energy drink can', 'milk carton', "
    "'cigarette pack', 'plastic bottle', 'bread loaf', 'shopping bag'). "
    "Return a JSON array of descriptions. If none are visible, return []."
)
POSE_MODEL_PATH = "yolo11n-pose.pt"

# ===========================
# DETECTION
# ===========================

TARGET_FPS = 5
VIDEO_ANALYTICS_ENABLED = True

SCAN_CONFIDENCE = 0.2
PERSON_CONFIDENCE = 0.4
POSE_KEYPOINT_CONFIDENCE = 0.3
SCAN_CLIP_CLASSIFICATION_ENABLED = True
SCAN_CLIP_MODEL = "openai/clip-vit-base-patch32"
SCAN_CLIP_LABELS_PATH = Path("labels.txt")
SCAN_CLIP_MIN_CONFIDENCE = 0.0
SCAN_CLIP_NEGATIVE_LABELS = [
    "price tag",
    "discount sticker",
    "shelf label",
    "advertising sticker",
    "printed sign",
    "store sign",
    "poster",
    "hand",
    "fingers",
    "background",
]
SCAN_VLM_CLASSIFICATION_ENABLED = True
SCAN_VLM_CLASSIFICATION_MODEL = "Qwen/Qwen2-VL-2B-Instruct"
SCAN_VLM_CLASSIFICATION_LOCAL_FILES_ONLY = False
SCAN_VLM_CLASSIFICATION_MODE = "all"  # "fallback" or "all"
SCAN_VLM_CLASSIFICATION_MIN_CLIP_CONFIDENCE = 0.12
SCAN_VLM_CLASSIFICATION_INTERVAL_SECONDS = 2.0
SCAN_VLM_CLASSIFICATION_MAX_OBJECTS = 4
SCAN_VLM_CLASSIFICATION_PROMPT = (
    "Look at this crop from a checkout camera. Describe the main visible retail product "
    "or merchandise item in 1-3 words (e.g., 'energy drink can', 'milk carton', 'cigarette pack', "
    "'plastic bottle', 'bread loaf', 'shopping bag'). If only price tags, stickers, posters, "
    "shelf labels, hands, background, or unreadable fragments are visible, return 'background'."
)

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
AUDIO_ONLY_VISIT_SECONDS = 300
WHISPER_MODEL = (
    "large-v3-turbo"
    if sys.platform == "win32"
    else "mlx-community/whisper-large-v3-turbo-q4"
)
WHISPER_LANGUAGE = "ru"
WHISPER_BACKEND = "faster-whisper" if sys.platform == "win32" else "gigaam"
WHISPER_COMPUTE_TYPE = "int8"
GIGAAM_MODEL = "v3_e2e_rnnt"
# Keep GigaAM off MPS when video analytics also uses torch/transformers on macOS.
# Running both in parallel on MPS can abort inside Apple's Metal runtime.
GIGAAM_DEVICE = "cpu"
SENSEVOICE_MODEL = "FunAudioLLM/SenseVoiceSmall"
TRANSCRIPTS_DIR = Path("transcripts")
SERVICE_CHECKLIST_PROFILE = "ordinary_point"

ANALYTICS_API_BASE_URL = "https://bmon.gradusy24.kz/api/v1"
ANALYTICS_API_KEY = "analytics_key_NbeYMFPwY9spmfKd5h56fZsC_sF2RQ1A7-QTAnI1"
ANALYTICS_STORE_CODE = STORE_CODE
ANALYTICS_REGISTER_CODE = REGISTER_CODE
ANALYTICS_AUDIO_SOURCE = "EXTERNAL_MICROPHONE_RTSP"
ANALYTICS_SEND_TIMEOUT = 10
ANALYTICS_VIOLATION_EVENTS_PATH = "/analytics/video/events"
CUSTOMER_WAITING_VIOLATION_SECONDS = 20.0
CASHIER_ABSENT_DURING_SERVICE_SECONDS = 10.0
OBJECT_LEFT_IN_SCAN_ZONE_SECONDS = 30.0
VIOLATION_EVENT_COOLDOWN_SECONDS = 300.0
ROI_REFERENCE_UPLOAD_TIMEOUT = 15
ROI_REFERENCE_CAPTURE_TIMEOUT = 15
ANALYTICS_ROI_FETCH_ENABLED = True
ANALYTICS_ROI_FETCH_TIMEOUT = 10

# ===========================
# DATASET COLLECTION
# ===========================

DATASET_COLLECTION_ENABLED = True
DATASET_DIR = Path("dataset_output")
FLORENCE_MODEL = "microsoft/Florence-2-base-ft"
DATASET_TRACK_TIMEOUT = 2.0

# ===========================
# CAMERAS
# ===========================

STREAMS = {
    CAMERA_CODE: {
        "url": "rtsp://admin:LeWfBvc4%21@127.0.0.1:8554/cam/realmonitor?channel=10&subtype=0",
        "audio_url": "rtsp://100.96.0.32:8554/mic",
        "service_profile": SERVICE_CHECKLIST_PROFILE,

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
