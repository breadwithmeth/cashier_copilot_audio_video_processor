import time
import cv2

from config import (STREAMS, TARGET_FPS, SPEECH_RECOGNITION_ENABLED,
                    WHISPER_MODEL, WHISPER_LANGUAGE, WHISPER_BACKEND,
                    WHISPER_COMPUTE_TYPE, SENSEVOICE_MODEL, GIGAAM_MODEL,
                    GIGAAM_DEVICE, TRANSCRIPTS_DIR)
from config import (DATASET_COLLECTION_ENABLED, DATASET_DIR,
                    DATASET_TRACK_TIMEOUT)
from audio.rtsp_transcriber import RTSPVisitTranscriber
from dataset.object_collector import ObjectDatasetCollector

from camera.rtsp_reader import RTSPReader

from vision.scan_detector import ScanDetector
from vision.person_detector import PersonDetector

from logic.checkout_state import CheckoutState

from ui.overlay import Overlay


def show_object_windows(camera_name, frame, objects, open_windows):
    frame_height, frame_width = frame.shape[:2]
    now = time.monotonic()

    for index, obj in enumerate(objects):
        object_id = obj.track_id if obj.track_id is not None else f"temp_{index}"
        class_name = obj.class_name.replace("_", " ")
        window_name = f"{camera_name} | {class_name} {object_id}"
        x1, y1, x2, y2 = obj.bbox
        x1, x2 = max(0, x1), min(frame_width, x2)
        y1, y2 = max(0, y1), min(frame_height, y2)
        if x2 <= x1 or y2 <= y1:
            continue

        crop = frame[y1:y2, x1:x2].copy()
        crop_height, crop_width = crop.shape[:2]
        scale = min(400 / crop_width, 400 / crop_height, 3.0)
        if scale != 1.0:
            crop = cv2.resize(
                crop,
                (max(1, int(crop_width * scale)),
                 max(1, int(crop_height * scale))),
                interpolation=(cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA),
            )

        label = f"{class_name} | ID {object_id} | conf {obj.confidence:.2f}"
        cv2.rectangle(crop, (0, 0), (crop.shape[1], 32), (0, 0, 0), -1)
        cv2.putText(crop, label, (8, 23), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 2)
        cv2.imshow(window_name, crop)
        if window_name not in open_windows:
            cv2.moveWindow(window_name, 40 + (index % 4) * 420,
                           80 + (index // 4) * 460)
        open_windows[window_name] = now

    stale_windows = [
        window_name
        for window_name, last_seen in open_windows.items()
        if now - last_seen > 2.0
    ]
    for stale_window in stale_windows:
        try:
            cv2.destroyWindow(stale_window)
        except cv2.error:
            pass
        del open_windows[stale_window]


def create_camera(camera_name, cfg):

    reader = RTSPReader(
        name=camera_name,
        url=cfg["url"],
    )

    scan_detector = ScanDetector(
        roi=cfg["scan_roi"],
    )

    person_detector = PersonDetector(
        customer_roi=cfg["customer_roi"],
        cashier_roi=cfg["cashier_roi"],
    )

    state = CheckoutState()

    dataset_collector = None
    if DATASET_COLLECTION_ENABLED:
        dataset_collector = ObjectDatasetCollector(
            camera_name=camera_name,
            output_dir=DATASET_DIR,
            track_timeout=DATASET_TRACK_TIMEOUT,
        )

    transcriber = None
    if SPEECH_RECOGNITION_ENABLED:
        transcriber = RTSPVisitTranscriber(
            name=camera_name,
            url=cfg.get("audio_url", cfg["url"]),
            output_dir=TRANSCRIPTS_DIR,
            model=(SENSEVOICE_MODEL if WHISPER_BACKEND == "sensevoice"
                   else GIGAAM_MODEL if WHISPER_BACKEND == "gigaam"
                   else WHISPER_MODEL),
            language=WHISPER_LANGUAGE,
            backend=WHISPER_BACKEND,
            compute_type=WHISPER_COMPUTE_TYPE,
            device=GIGAAM_DEVICE,
            prebuffer_seconds=max(5.0, cfg.get("customer_timeout", 0)),
        )

    overlay = Overlay(
        cfg["scan_roi"],
        cfg["customer_roi"],
        cfg["cashier_roi"],
    )

    return {
        "reader": reader,
        "scan_detector": scan_detector,
        "person_detector": person_detector,
        "state": state,
        "overlay": overlay,
        "transcriber": transcriber,
        "object_windows": {},
        "dataset_collector": dataset_collector,
    }


def main():

    cameras = {}

    try:
        for name, cfg in STREAMS.items():

            cameras[name] = create_camera(name, cfg)

        print("System started")

        frame_interval = 1.0 / TARGET_FPS if TARGET_FPS > 0 else 0

        while True:
            loop_started = time.monotonic()

            for camera_name, camera in cameras.items():

                frame = camera["reader"].get_frame()

                if frame is None:
                    continue

                scan_result = camera["scan_detector"].detect(frame)

                if camera["dataset_collector"] is not None:
                    camera["dataset_collector"].observe(frame, scan_result.objects)

                show_object_windows(
                    camera_name,
                    frame,
                    scan_result.objects,
                    camera["object_windows"],
                )

                person_result = camera["person_detector"].detect(frame)

                camera["state"].update(
                    customer_detected=person_result.customer_detected,
                    cashier_detected=person_result.cashier_detected,
                    scan_objects=scan_result.objects,
                )

                transcriber = camera["transcriber"]
                if transcriber is not None:
                    if camera["state"].customer_arrived:
                        transcriber.start_visit(camera["state"].visit_started_at)
                    if camera["state"].customer_left:
                        transcriber.end_visit(time.time())

                image = camera["overlay"].draw(
                    frame=frame,
                    state=camera["state"],
                    scan_result=scan_result,
                    person_result=person_result,
                    subtitle=(camera["transcriber"].get_subtitle()
                              if camera["transcriber"] is not None else ""),
                )

                cv2.imshow(camera_name, image)

            key = cv2.waitKey(1)

            if key == 27 or key == ord("q"):
                break

            if key == ord("r"):
                for camera in cameras.values():
                    camera["state"].reset_product_count()

            if frame_interval > 0:
                elapsed = time.monotonic() - loop_started
                sleep_seconds = frame_interval - elapsed

                if sleep_seconds > 0:
                    time.sleep(sleep_seconds)

    finally:
        for camera in cameras.values():
            camera["reader"].stop()
            if camera["transcriber"] is not None:
                camera["transcriber"].stop()
            if camera["dataset_collector"] is not None:
                camera["dataset_collector"].stop()

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
