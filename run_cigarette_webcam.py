from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".cache/matplotlib").resolve()))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import cv2
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the tobacco detector on a local webcam."
    )
    parser.add_argument(
        "--model",
        default="runs/cigarette_detector/tobacco_3datasets/weights/best.pt",
    )
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--conf", type=float, default=0.55)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLO(args.model)
    capture = cv2.VideoCapture(args.camera)
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open webcam index {args.camera}")

    window_name = "tobacco webcam"
    previous_time = time.monotonic()

    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                print("Webcam frame read failed")
                time.sleep(0.1)
                continue

            result = model.predict(
                source=frame,
                imgsz=args.imgsz,
                conf=args.conf,
                iou=args.iou,
                device=args.device,
                verbose=False,
            )[0]

            annotated = result.plot()
            now = time.monotonic()
            fps = 1.0 / max(now - previous_time, 1e-6)
            previous_time = now
            cv2.putText(
                annotated,
                f"FPS {fps:.1f} | conf {args.conf:.2f} | imgsz {args.imgsz}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2,
            )
            cv2.imshow(window_name, annotated)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
