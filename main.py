import argparse
import os
import time
import cv2

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from config import (STREAMS, TARGET_FPS, SPEECH_RECOGNITION_ENABLED,
                    WHISPER_MODEL, WHISPER_LANGUAGE, WHISPER_BACKEND,
                    WHISPER_COMPUTE_TYPE, SENSEVOICE_MODEL, GIGAAM_MODEL,
                    GIGAAM_DEVICE, TRANSCRIPTS_DIR, VIDEO_ANALYTICS_ENABLED,
                    AUDIO_ONLY_VISIT_SECONDS, ACTIVE_CAMERA_TYPE)
from config import (DATASET_COLLECTION_ENABLED, DATASET_DIR,
                    DATASET_TRACK_TIMEOUT, ANALYTICS_ROI_FETCH_ENABLED)
from analytics_rois import apply_backend_rois
from analytics_violations import VideoViolationMonitor
from audio.rtsp_transcriber import RTSPVisitTranscriber
from dataset.object_collector import ObjectDatasetCollector

from camera.rtsp_reader import RTSPReader
from camera.video_file_reader import VideoFileReader

from vision.scan_detector import ScanDetector
from vision.person_detector import PersonDetector
from vision.roi import clip_roi

from callcenter.desk_detector import DeskDetector
from logic.checkout_state import CheckoutState
from models.detection import ScanResult
from ui.overlay import Overlay


def create_camera(camera_name, cfg):

    reader = None
    scan_detector = None
    person_detector = None
    overlay = None

    if VIDEO_ANALYTICS_ENABLED:
        source_url = cfg["url"]
        is_local_file = not source_url.lower().startswith("rtsp://")

        if is_local_file:
            reader = VideoFileReader(
                name=camera_name,
                url=source_url,
            )
        else:
            reader = RTSPReader(
                name=camera_name,
                url=source_url,
            )

        camera_type = cfg.get("type", "checkout")

        if camera_type == "callcenter":
            scan_detector = None
            person_detector = DeskDetector(
                agent_roi=cfg["agent_roi"],
                customer_roi=cfg["customer_roi"],
            )
            overlay = Overlay(
                scan_roi=None,
                customer_roi=cfg["customer_roi"],
                cashier_roi=cfg["agent_roi"],
            )
        else:
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
    violation_monitor = VideoViolationMonitor(camera_name)

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
            service_profile=cfg.get("service_profile"),
        )

    return {
        "reader": reader,
        "scan_detector": scan_detector,
        "person_detector": person_detector,
        "state": state,
        "overlay": overlay,
        "transcriber": transcriber,
        "dataset_collector": dataset_collector,
        "violation_monitor": violation_monitor,
        "audio_only_visit_started_at": None,
        "roi_logged": False,
    }


def log_camera_rois_once(camera_name, camera, frame):
    if camera["roi_logged"]:
        return

    height, width = frame.shape[:2]
    print(f"[{camera_name}] frame size: {width}x{height}")

    if camera["scan_detector"] is not None:
        rois = {
            "scan_roi": camera["scan_detector"].roi,
            "customer_roi": camera["person_detector"].customer_roi,
            "cashier_roi": camera["person_detector"].cashier_roi,
        }
    else:
        rois = {
            "customer_roi": camera["person_detector"].customer_roi,
            "agent_roi": camera["person_detector"].agent_roi,
        }
    for name, roi in rois.items():
        print(f"[{camera_name}] {name}: configured={roi} clipped={clip_roi(roi, frame)}")

    camera["roi_logged"] = True


def main():

    parser = argparse.ArgumentParser(
        description="Cashier Copilot video processor")
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Local video file path (e.g. video.mp4) or RTSP URL. "
             "When a local file is given, it overrides the configured "
             "STREAMS with a single camera using that file.",
    )
    args = parser.parse_args()

    # Determine active streams: --source overrides config STREAMS
    if args.source:
        is_local_file = not args.source.lower().startswith("rtsp://")
        if is_local_file:
            if not os.path.isfile(args.source):
                print(f"Error: video file not found: {args.source}")
                return
        # Build a single-camera config from the first STREAMS entry
        base_cfg = next(iter(STREAMS.values()))
        streams = {
            "LOCAL_VIDEO": {
                **base_cfg,
                "url": args.source,
                "audio_url": None,
            },
        }
        # Disable backend ROI fetch for local files — use configured ROIs as-is
        use_backend_rois = False
    else:
        streams = STREAMS
        use_backend_rois = ANALYTICS_ROI_FETCH_ENABLED

    # Filter streams based on ACTIVE_CAMERA_TYPE
    if ACTIVE_CAMERA_TYPE != "both":
        filtered_streams = {}
        for name, cfg in streams.items():
            camera_type = cfg.get("type", "checkout")
            if camera_type == ACTIVE_CAMERA_TYPE:
                filtered_streams[name] = cfg
        streams = filtered_streams
        
        if not streams:
            print(f"No cameras found for type: {ACTIVE_CAMERA_TYPE}")
            return

    cameras = {}

    try:
        for name, cfg in streams.items():
            if use_backend_rois:
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

                # Skip cameras whose reader has been stopped (e.g. video EOF)
                if camera["reader"] is None:
                    continue

                frame = camera["reader"].get_frame()

                if frame is None:
                    # For local video files, None means EOF — stop the camera
                    if isinstance(camera["reader"], VideoFileReader):
                        print(f"[{camera_name}] Video file ended, stopping camera.")
                        camera["reader"].stop()
                        camera["reader"] = None
                        continue
                    # For RTSP, None is transient (reconnect handled in reader)
                    continue

                log_camera_rois_once(camera_name, camera, frame)

                if camera["scan_detector"] is not None:
                    scan_result = camera["scan_detector"].detect(frame)
                else:
                    scan_result = ScanResult()

                if camera["dataset_collector"] is not None:
                    camera["dataset_collector"].observe(frame, scan_result.objects)

                person_result = camera["person_detector"].detect(frame)

                camera["state"].update(
                    customer_detected=person_result.customer_detected,
                    cashier_detected=person_result.cashier_detected,
                    scan_objects=scan_result.objects,
                )
                camera["violation_monitor"].evaluate(camera["state"])

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

            # If all video readers are stopped (local files ended), exit
            if VIDEO_ANALYTICS_ENABLED:
                active_readers = [
                    c for c in cameras.values()
                    if c["reader"] is not None
                ]
                if not active_readers:
                    print("All video sources ended. Exiting.")
                    break

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
            if camera["violation_monitor"] is not None:
                camera["violation_monitor"].stop()

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
