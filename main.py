import cv2
import time
import threading
from ultralytics import YOLO

SCAN_MODEL_PATH = "weights/best.pt"
POSE_MODEL_PATH = "yolo11n-pose.pt"

STREAMS = {
    "cam10": "rtsp://admin:LeWfBvc4%21@127.0.0.1:8554/cam/realmonitor?channel=10&subtype=1",
}

SCAN_ROIS = {
    "cam10": (200, 100, 470, 580),
}

CUSTOMER_ROIS = {
    "cam10": (470, 0, 960, 540),
}

CASHIER_ROIS = {
    "cam10": (0, 80, 300, 580),
}

CONFIDENCE_SCAN = 0.5
CONFIDENCE_PERSON = 0.4
IMAGE_SIZE_SCAN = 640
IMAGE_SIZE_POSE = 640
TARGET_FPS = 5

latest_raw_frames = {}
latest_result_frames = {}
frame_locks = {}
running = True


def draw_roi(frame, roi, name, color):
    x1, y1, x2, y2 = roi
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, name, (x1, max(30, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)


def clip_roi(roi, frame):
    x1, y1, x2, y2 = roi
    h, w = frame.shape[:2]
    return (
        max(0, min(x1, w - 1)),
        max(0, min(y1, h - 1)),
        max(0, min(x2, w)),
        max(0, min(y2, h)),
    )


def rtsp_reader(name, url):
    global running
    cap = None

    while running:
        if cap is None or not cap.isOpened():
            print(f"[{name}] Connecting RTSP...")
            cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            time.sleep(1)
            continue

        success, frame = cap.read()
        if not success:
            print(f"[{name}] RTSP lost, reconnecting...")
            cap.release()
            cap = None
            time.sleep(2)
            continue

        with frame_locks[name]:
            latest_raw_frames[name] = frame

    if cap:
        cap.release()


def detect_person_in_roi(pose_model, frame, roi, confidence, image_size):
    x1, y1, x2, y2 = clip_roi(roi, frame)
    roi_frame = frame[y1:y2, x1:x2]

    if roi_frame.size == 0:
        return False, [], 0

    start = time.time()

    results = pose_model.predict(
        source=roi_frame,
        conf=confidence,
        imgsz=image_size,
        verbose=False
    )

    ms = int((time.time() - start) * 1000)
    persons = []

    for result in results:
        if result.boxes is None:
            continue

        for box in result.boxes:
            cls = int(box.cls.item())
            conf = float(box.conf.item())
            class_name = pose_model.names[cls]

            if class_name != "person":
                continue

            bx1, by1, bx2, by2 = box.xyxy[0].tolist()

            persons.append({
                "confidence": conf,
                "bbox": (
                    int(bx1 + x1),
                    int(by1 + y1),
                    int(bx2 + x1),
                    int(by2 + y1),
                )
            })

    return len(persons) > 0, persons, ms


def yolo_worker(name):
    global running

    print(f"[{name}] Loading scan model...")
    scan_model = YOLO(SCAN_MODEL_PATH)

    print(f"[{name}] Loading pose model...")
    pose_model = YOLO(POSE_MODEL_PATH)

    interval = 1.0 / TARGET_FPS
    last_processed_at = 0

    customer_present_since = None
    customer_is_present = False

    cashier_present_since = None
    cashier_is_present = False

    while running:
        now = time.time()
        if now - last_processed_at < interval:
            time.sleep(0.005)
            continue

        last_processed_at = now

        with frame_locks[name]:
            frame = latest_raw_frames.get(name)
            if frame is not None:
                frame = frame.copy()

        if frame is None:
            time.sleep(0.05)
            continue

        annotated = frame.copy()

        scan_roi = clip_roi(SCAN_ROIS[name], frame)
        customer_roi = clip_roi(CUSTOMER_ROIS[name], frame)
        cashier_roi = clip_roi(CASHIER_ROIS[name], frame)

        draw_roi(annotated, scan_roi, "scan_zone", (0, 255, 255))
        draw_roi(annotated, customer_roi, "customer_zone", (255, 0, 255))
        draw_roi(annotated, cashier_roi, "cashier_zone", (255, 255, 0))

        sx1, sy1, sx2, sy2 = scan_roi
        scan_frame = frame[sy1:sy2, sx1:sx2]

        scan_start = time.time()

        scan_results = scan_model.predict(
            source=scan_frame,
            conf=CONFIDENCE_SCAN,
            imgsz=IMAGE_SIZE_SCAN,
            verbose=False
        )

        scan_ms = int((time.time() - scan_start) * 1000)

        for result in scan_results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                cls = int(box.cls.item())
                conf = float(box.conf.item())
                bx1, by1, bx2, by2 = box.xyxy[0].tolist()

                full_x1 = int(bx1 + sx1)
                full_y1 = int(by1 + sy1)
                full_x2 = int(bx2 + sx1)
                full_y2 = int(by2 + sy1)

                class_name = scan_model.names[cls]

                cv2.rectangle(annotated, (full_x1, full_y1), (full_x2, full_y2), (0, 255, 0), 2)
                cv2.putText(annotated, f"{class_name} {conf:.2f}",
                            (full_x1, max(25, full_y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        customer_detected, customer_persons, customer_ms = detect_person_in_roi(
            pose_model, frame, customer_roi, CONFIDENCE_PERSON, IMAGE_SIZE_POSE
        )

        cashier_detected, cashier_persons, cashier_ms = detect_person_in_roi(
            pose_model, frame, cashier_roi, CONFIDENCE_PERSON, IMAGE_SIZE_POSE
        )

        for p in customer_persons:
            x1, y1, x2, y2 = p["bbox"]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 0, 255), 2)
            cv2.putText(annotated, f"customer {p['confidence']:.2f}",
                        (x1, max(25, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

        for p in cashier_persons:
            x1, y1, x2, y2 = p["bbox"]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 255, 0), 2)
            cv2.putText(annotated, f"cashier {p['confidence']:.2f}",
                        (x1, max(25, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        if customer_detected:
            if customer_present_since is None:
                customer_present_since = time.time()
            if time.time() - customer_present_since >= 2.0 and not customer_is_present:
                customer_is_present = True
                print(f"[{name}] CUSTOMER_PRESENT")
        else:
            customer_present_since = None
            if customer_is_present:
                customer_is_present = False
                print(f"[{name}] CUSTOMER_LEFT")

        if cashier_detected:
            if cashier_present_since is None:
                cashier_present_since = time.time()
            if time.time() - cashier_present_since >= 2.0 and not cashier_is_present:
                cashier_is_present = True
                print(f"[{name}] CASHIER_PRESENT")
        else:
            cashier_present_since = None
            if cashier_is_present:
                cashier_is_present = False
                print(f"[{name}] CASHIER_LEFT")

        customer_status = "CUSTOMER: YES" if customer_is_present else "CUSTOMER: NO"
        cashier_status = "CASHIER: YES" if cashier_is_present else "CASHIER: NO"

        cv2.putText(annotated, customer_status, (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 0, 255) if customer_is_present else (180, 180, 180), 2)

        cv2.putText(annotated, cashier_status, (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 255, 0) if cashier_is_present else (180, 180, 180), 2)

        cv2.putText(annotated,
                    f"{name} | scan {scan_ms}ms | customer {customer_ms}ms | cashier {cashier_ms}ms",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 255), 2)

        with frame_locks[name]:
            latest_result_frames[name] = annotated


def main():
    global running

    for name in STREAMS:
        frame_locks[name] = threading.Lock()
        latest_raw_frames[name] = None
        latest_result_frames[name] = None

    for name, url in STREAMS.items():
        threading.Thread(target=rtsp_reader, args=(name, url), daemon=True).start()
        threading.Thread(target=yolo_worker, args=(name,), daemon=True).start()

    try:
        while running:
            for name in STREAMS:
                with frame_locks[name]:
                    frame = latest_result_frames.get(name)
                    if frame is not None:
                        frame = frame.copy()

                if frame is not None:
                    frame = cv2.resize(frame, (960, 540))
                    cv2.imshow(name, frame)

            key = cv2.waitKey(1)

            if key == 27 or key == ord("q"):
                running = False
                break

            time.sleep(0.01)

    except KeyboardInterrupt:
        running = False

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()