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
# ACTIVE CAMERA TYPES
# ===========================

# Set to "checkout", "callcenter", or "both" to control which cameras run
ACTIVE_CAMERA_TYPE = "both"  # "checkout", "callcenter", or "both"

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

SCAN_WORLD_PROMPTS = [
    "water bottle",
    "soda bottle",
    "juice carton",
    "milk carton",
    "glass bottle",
    "wine bottle",
    "beer can",
    "tin can",
    "glass jar",
    "cardboard box",
    "cereal box",
    "pizza box",
    "egg carton",
    "plastic pouch",
    "plastic food tray",
    "vacuum sealed package",
    "takeaway container",
    "paste tube",
    "shampoo bottle",
    "detergent bottle",
    "plastic shopping bag",
    "paper bag",
    "candy wrapper",
    "shrink wrap",
    "fresh produce",
    "bread loaf",
    "meat tray",
    "snack bag",
    "coffee jar",
    "yogurt cup",
    "butter pack",
    "ice cream tub",
    "toilet paper pack",
    "diaper pack",
    "shopping basket",
    "barcode scanner",
    "payment terminal",
    "paper receipt",
    "credit card",
    "banknote",
    "passport",
    "cigarette pack",
    "tri groove bolt",
    "anti vandal bolt",
]

SCAN_OWLV2_MODEL = "google/owlv2-base-patch16-ensemble"
SCAN_OWLV2_CLASSES = [
    {"label": "water_bottle", "prompt": "plastic or glass water bottle"},
    {"label": "soda_bottle", "prompt": "soda or soft drink plastic bottle"},
    {"label": "juice_carton", "prompt": "juice carton or tetra pak"},
    {"label": "milk_carton", "prompt": "milk carton or dairy tetra pak"},
    {"label": "glass_bottle", "prompt": "glass drink bottle for beer or wine"},
    {"label": "wine_bottle", "prompt": "wine bottle"},
    {"label": "beer_can", "prompt": "aluminum beer can or drink can"},
    {"label": "tin_can", "prompt": "tin food can or aluminum food can"},
    {"label": "glass_jar", "prompt": "glass jar with food like jam or sauce"},
    {"label": "cardboard_box", "prompt": "cardboard product box or packaging"},
    {"label": "cereal_box", "prompt": "cereal box or breakfast food box"},
    {"label": "pizza_box", "prompt": "cardboard pizza box"},
    {"label": "egg_carton", "prompt": "egg carton or egg tray"},
    {"label": "plastic_pouch", "prompt": "plastic pouch package for snacks or food"},
    {"label": "plastic_food_tray", "prompt": "plastic food tray"},
    {"label": "vacuum_sealed_package", "prompt": "vacuum sealed retail package for meat or cheese"},
    {"label": "takeaway_container", "prompt": "takeaway food container or clamshell"},
    {"label": "paste_tube", "prompt": "paste tube or squeeze tube for toothpaste or cream"},
    {"label": "shampoo_bottle", "prompt": "shampoo or shower gel plastic bottle"},
    {"label": "detergent_bottle", "prompt": "detergent or cleaning liquid bottle"},
    {"label": "plastic_shopping_bag", "prompt": "plastic shopping bag or carrier bag"},
    {"label": "paper_bag", "prompt": "paper shopping bag or kraft bag"},
    {"label": "candy_wrapper", "prompt": "candy or chocolate wrapper"},
    {"label": "shrink_wrap", "prompt": "plastic shrink wrap or multipack wrapping"},
    {"label": "fresh_produce", "prompt": "fresh fruit or vegetable produce"},
    {"label": "bread_loaf", "prompt": "bread loaf or bakery product in bag"},
    {"label": "meat_tray", "prompt": "raw meat tray with plastic film"},
    {"label": "snack_bag", "prompt": "chips or snack plastic bag"},
    {"label": "coffee_jar", "prompt": "glass coffee jar or instant coffee container"},
    {"label": "yogurt_cup", "prompt": "yogurt cup or plastic dairy cup"},
    {"label": "butter_pack", "prompt": "butter package or margarine block"},
    {"label": "ice_cream_tub", "prompt": "ice cream plastic tub"},
    {"label": "toilet_paper_pack", "prompt": "toilet paper pack or paper towels"},
    {"label": "diaper_pack", "prompt": "diaper pack or baby wipes package"},
    {"label": "shopping_basket", "prompt": "plastic or metal shopping basket"},
    {"label": "barcode_scanner", "prompt": "barcode scanner or checkout scanner"},
    {"label": "payment_terminal", "prompt": "payment terminal or card reader"},
    {"label": "paper_receipt", "prompt": "paper receipt"},
    {"label": "credit_card", "prompt": "credit card or bank card"},
    {"label": "banknote", "prompt": "cash banknote or paper money"},
    {"label": "passport", "prompt": "passport or ID document"},
    {"label": "cigarette_pack", "prompt": "cigarette pack or tobacco box"},
    {"label": "tri_groove_bolt", "prompt": "tri groove bolt or anti vandal bolt"},
    {"label": "anti_vandal_bolt", "prompt": "anti vandal bolt or security bolt"},
]
SCAN_OWLV2_PROMPTS = [item["prompt"] for item in SCAN_OWLV2_CLASSES]

SCAN_OMDET_MODEL = "omlab/omdet-turbo-swin-tiny-hf"
SCAN_OMDET_CLASSES = [
    {"label": "water_bottle", "prompt": "plastic or glass water bottle"},
    {"label": "soda_bottle", "prompt": "soda or soft drink plastic bottle"},
    {"label": "juice_carton", "prompt": "juice carton or tetra pak"},
    {"label": "milk_carton", "prompt": "milk carton or dairy tetra pak"},
    {"label": "glass_bottle", "prompt": "glass drink bottle for beer or wine"},
    {"label": "wine_bottle", "prompt": "wine bottle"},
    {"label": "beer_can", "prompt": "aluminum beer can or drink can"},
    {"label": "tin_can", "prompt": "tin food can or aluminum food can"},
    {"label": "glass_jar", "prompt": "glass jar with food like jam or sauce"},
    {"label": "cardboard_box", "prompt": "cardboard product box or packaging"},
    {"label": "cereal_box", "prompt": "cereal box or breakfast food box"},
    {"label": "pizza_box", "prompt": "cardboard pizza box"},
    {"label": "egg_carton", "prompt": "egg carton or egg tray"},
    {"label": "plastic_pouch", "prompt": "plastic pouch package for snacks or food"},
    {"label": "plastic_food_tray", "prompt": "plastic food tray"},
    {"label": "vacuum_sealed_package", "prompt": "vacuum sealed retail package for meat or cheese"},
    {"label": "takeaway_container", "prompt": "takeaway food container or clamshell"},
    {"label": "paste_tube", "prompt": "paste tube or squeeze tube for toothpaste or cream"},
    {"label": "shampoo_bottle", "prompt": "shampoo or shower gel plastic bottle"},
    {"label": "detergent_bottle", "prompt": "detergent or cleaning liquid bottle"},
    {"label": "plastic_shopping_bag", "prompt": "plastic shopping bag or carrier bag"},
    {"label": "paper_bag", "prompt": "paper shopping bag or kraft bag"},
    {"label": "candy_wrapper", "prompt": "candy or chocolate wrapper"},
    {"label": "shrink_wrap", "prompt": "plastic shrink wrap or multipack wrapping"},
    {"label": "fresh_produce", "prompt": "fresh fruit or vegetable produce"},
    {"label": "bread_loaf", "prompt": "bread loaf or bakery product in bag"},
    {"label": "meat_tray", "prompt": "raw meat tray with plastic film"},
    {"label": "snack_bag", "prompt": "chips or snack plastic bag"},
    {"label": "coffee_jar", "prompt": "glass coffee jar or instant coffee container"},
    {"label": "yogurt_cup", "prompt": "yogurt cup or plastic dairy cup"},
    {"label": "butter_pack", "prompt": "butter package or margarine block"},
    {"label": "ice_cream_tub", "prompt": "ice cream plastic tub"},
    {"label": "toilet_paper_pack", "prompt": "toilet paper pack or paper towels"},
    {"label": "diaper_pack", "prompt": "diaper pack or baby wipes package"},
    {"label": "shopping_basket", "prompt": "plastic or metal shopping basket"},
    {"label": "barcode_scanner", "prompt": "barcode scanner or checkout scanner"},
    {"label": "payment_terminal", "prompt": "payment terminal or card reader"},
    {"label": "paper_receipt", "prompt": "paper receipt"},
    {"label": "credit_card", "prompt": "credit card or bank card"},
    {"label": "banknote", "prompt": "cash banknote or paper money"},
    {"label": "passport", "prompt": "passport or ID document"},
    {"label": "cigarette_pack", "prompt": "cigarette pack or tobacco box"},
    {"label": "tri_groove_bolt", "prompt": "tri groove bolt or anti vandal bolt"},
    {"label": "anti_vandal_bolt", "prompt": "anti vandal bolt or security bolt"},
]
SCAN_OMDET_PROMPTS = [item["prompt"] for item in SCAN_OMDET_CLASSES]

SCAN_SMOLVLM_MODEL = "HuggingFaceTB/SmolVLM-500M-Instruct"
SCAN_RT_DETR_V2_MODEL = "nielsr/rtdetr-tray-cart-tuned-strong-20260303-204722"
SCAN_SMOLVLM_INTERVAL_SECONDS = 2.0
SCAN_SMOLVLM_PROMPT = (
    "Look at the cashier scan area image. Describe each visible retail product "
    "or merchandise item. Use only labels from this list: "
    "water_bottle, soda_bottle, juice_carton, milk_carton, glass_bottle, wine_bottle, "
    "beer_can, tin_can, glass_jar, cardboard_box, cereal_box, pizza_box, egg_carton, "
    "plastic_pouch, plastic_food_tray, vacuum_sealed_package, takeaway_container, "
    "paste_tube, shampoo_bottle, detergent_bottle, plastic_shopping_bag, paper_bag, "
    "candy_wrapper, shrink_wrap, fresh_produce, bread_loaf, meat_tray, snack_bag, "
    "coffee_jar, yogurt_cup, butter_pack, ice_cream_tub, toilet_paper_pack, diaper_pack, "
    "shopping_basket, barcode_scanner, payment_terminal, paper_receipt, credit_card, "
    "banknote, passport, cigarette_pack, tri_groove_bolt, anti_vandal_bolt. "
    "Return a JSON array of these labels. If none are visible, return []."
)
POSE_MODEL_PATH = "yolo11n-pose.pt"

# ===========================
# CALL CENTER / DESK DETECTION
# ===========================

DESK_MODEL_PATH = "yolo11n-pose.pt"
DESK_CONFIDENCE = 0.25
DESK_IMAGE_SIZE = (1088, 1920)
DESK_KEYPOINT_CONFIDENCE = 0.3

# ===========================
# DETECTION
# ===========================

TARGET_FPS = 5
VIDEO_ANALYTICS_ENABLED = True

SCAN_CONFIDENCE = 0.2
PERSON_CONFIDENCE = 0.25
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
    "bolt head",
    "screw head",
    "metal fastener",
    "hardware",
]
SCAN_VLM_CLASSIFICATION_ENABLED = True
SCAN_VLM_CLASSIFICATION_MODEL = "Qwen/Qwen2-VL-2B-Instruct"
SCAN_VLM_CLASSIFICATION_LOCAL_FILES_ONLY = False
SCAN_VLM_CLASSIFICATION_MODE = "all"  # "fallback" or "all"
SCAN_VLM_CLASSIFICATION_MIN_CLIP_CONFIDENCE = 0.12
SCAN_VLM_CLASSIFICATION_INTERVAL_SECONDS = 2.0
SCAN_VLM_CLASSIFICATION_MAX_OBJECTS = 4
SCAN_VLM_CLASSIFICATION_PROMPT = (
    "Look at this crop from a checkout camera. Identify the main visible retail product, "
    "merchandise item, or checkout element. "
    "Return ONLY the exact label from this list, nothing else: "
    "water_bottle, soda_bottle, juice_carton, milk_carton, glass_bottle, wine_bottle, "
    "beer_can, tin_can, glass_jar, cardboard_box, cereal_box, pizza_box, egg_carton, "
    "plastic_pouch, plastic_food_tray, vacuum_sealed_package, takeaway_container, "
    "paste_tube, shampoo_bottle, detergent_bottle, plastic_shopping_bag, paper_bag, "
    "candy_wrapper, shrink_wrap, fresh_produce, bread_loaf, meat_tray, snack_bag, "
    "coffee_jar, yogurt_cup, butter_pack, ice_cream_tub, toilet_paper_pack, diaper_pack, "
    "shopping_basket, barcode_scanner, payment_terminal, paper_receipt, credit_card, "
    "banknote, passport, cigarette_pack, tri_groove_bolt, anti_vandal_bolt. "
    "If only price tags, stickers, posters, shelf labels, hands, background, or unreadable "
    "fragments are visible, return 'background'."
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
        "type": "checkout",
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
    "desk_1": {
        "type": "callcenter",
        "url": "rtsp://admin:LeWfBvc4%21@127.0.0.1:8554/cam/realmonitor?channel=1&subtype=0",
        "audio_url": "rtsp://100.96.0.32:8554/mic",
        "service_profile": SERVICE_CHECKLIST_PROFILE,

        "agent_roi": (
            100,
            200,
            500,
            700,
        ),

        "customer_roi": (
            600,
            200,
            1000,
            700,
        ),
    },
    "desk_2": {
        "type": "callcenter",
        "url": "rtsp://admin:LeWfBvc4%21@127.0.0.1:8554/cam/realmonitor?channel=2&subtype=0",
        "audio_url": "rtsp://100.96.0.32:8554/mic",
        "service_profile": SERVICE_CHECKLIST_PROFILE,

        "agent_roi": (
            100,
            200,
            500,
            600,
        ),

        "customer_roi": (
            600,
            200,
            1000,
            700,
        ),
    },
    "desk_3": {
        "type": "callcenter",
        "url": "rtsp://admin:LeWfBvc4%21@127.0.0.1:8554/cam/realmonitor?channel=3&subtype=0",
        "audio_url": "rtsp://100.96.0.32:8554/mic",
        "service_profile": SERVICE_CHECKLIST_PROFILE,

        "agent_roi": (
            100,
            200,
            500,
            600,
        ),

        "customer_roi": (
            600,
            200,
            1000,
            700,
        ),
    },
    "desk_4": {
        "type": "callcenter",
        "url": "rtsp://admin:LeWfBvc4%21@127.0.0.1:8554/cam/realmonitor?channel=4&subtype=0",
        "audio_url": "rtsp://100.96.0.32:8554/mic",
        "service_profile": SERVICE_CHECKLIST_PROFILE,

        "agent_roi": (
            100,
            200,
            500,
            600,
        ),

        "customer_roi": (
            600,
            200,
            1000,
            700,
        ),
    },
    "desk_5": {
        "type": "callcenter",
        "url": "rtsp://admin:LeWfBvc4%21@127.0.0.1:8554/cam/realmonitor?channel=5&subtype=0",
        "audio_url": "rtsp://100.96.0.32:8554/mic",
        "service_profile": SERVICE_CHECKLIST_PROFILE,

        "agent_roi": (
            100,
            200,
            500,
            600,
        ),

        "customer_roi": (
            600,
            200,
            1000,
            700,
        ),
    },
}
