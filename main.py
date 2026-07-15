import os
import time
import cv2

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from config import (STREAMS, TARGET_FPS, SPEECH_RECOGNITION_ENABLED,
                    WHISPER_MODEL, WHISPER_LANGUAGE, WHISPER_BACKEND,
                    WHISPER_COMPUTE_TYPE, SENSEVOICE_MODEL, GIGAAM_MODEL,
                    GIGAAM_DEVICE, TRANSCRIPTS_DIR, VIDEO_ANALYTICS_ENABLED,
                    AUDIO_ONLY_VISIT_SECONDS)
from config import (DATASET_COLLECTION_ENABLED, DATASET_DIR,
                    DATASET_TRACK_TIMEOUT, ANALYTICS_ROI_FETCH_ENABLED)
from analytics_rois import apply_backend_rois
from audio.rtsp_transcriber import RTSPVisitTranscriber
from dataset.object_collector import ObjectDatasetCollector

from camera.rtsp_reader import RTSPReader

from vision.scan_detector import ScanDetector
from vision.person_detector import PersonDetector

from logic.checkout_state import CheckoutState

from ui.overlay import Overlay


def create_camera(camera_name, cfg):

    reader = None
    scan_detector = None
    person_detector = None
    overlay = None

    if VIDEO_ANALYTICS_ENABLED:
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

        overlay = Overlay(
            cfg["scan_roi"],
            cfg["customer_roi"],
            cfg["cashier_roi"],
        )

    state = CheckoutState()

    dataset_collector = None
    if DATASET_COLLECTION_ENABLED and VIDEO_ANALYTICS_ENABLED:
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

    return {
        "reader": reader,
        "scan_detector": scan_detector,
        "person_detector": person_detector,
        "state": state,
        "overlay": overlay,
        "transcriber": transcriber,
        "dataset_collector": dataset_collector,
        "audio_only_visit_started_at": None,
    }


def main():

    cameras = {}

    try:
        for name, cfg in STREAMS.items():
            if ANALYTICS_ROI_FETCH_ENABLED:
                cfg = apply_backend_rois(name, cfg)

            cameras[name] = create_camera(name, cfg)

        print("System started")

        frame_interval = 1.0 / TARGET_FPS if TARGET_FPS > 0 else 0

        while True:
            loop_started = time.monotonic()

            for camera_name, camera in cameras.items():
                if not VIDEO_ANALYTICS_ENABLED:
                    transcriber = camera["transcriber"]
                    if transcriber is None:
                        continue

                    now = time.time()
                    started_at = camera["audio_only_visit_started_at"]
                    if started_at is None:
                        camera["audio_only_visit_started_at"] = now
                        transcriber.start_visit(now)
                    elif now - started_at >= AUDIO_ONLY_VISIT_SECONDS:
                        transcriber.end_visit(now)
                        camera["audio_only_visit_started_at"] = now
                        transcriber.start_visit(now)
                    continue

                frame = camera["reader"].get_frame()

                if frame is None:
                    continue

                scan_result = camera["scan_detector"].detect(frame)

                if camera["dataset_collector"] is not None:
                    camera["dataset_collector"].observe(frame, scan_result.objects)

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
            if camera["reader"] is not None:
                camera["reader"].stop()
            if camera["transcriber"] is not None:
                camera["transcriber"].stop()
            if camera["dataset_collector"] is not None:
                camera["dataset_collector"].stop()

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
