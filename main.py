import time
import cv2

from config import STREAMS

from camera.rtsp_reader import RTSPReader

from vision.scan_detector import ScanDetector
from vision.person_detector import PersonDetector

from logic.checkout_state import CheckoutState

from ui.overlay import Overlay


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
    }


def main():

    cameras = {}

    for name, cfg in STREAMS.items():

        cameras[name] = create_camera(name, cfg)

    print("System started")

    while True:

        for camera_name, camera in cameras.items():

            frame = camera["reader"].get_frame()

            if frame is None:
                continue

            scan_result = camera["scan_detector"].detect(frame)

            person_result = camera["person_detector"].detect(frame)

            camera["state"].update(
                customer_detected=person_result.customer_detected,
                cashier_detected=person_result.cashier_detected,
                scan_objects=scan_result.objects,
            )

            image = camera["overlay"].draw(
                frame=frame,
                state=camera["state"],
                scan_result=scan_result,
                person_result=person_result,
            )

            cv2.imshow(camera_name, image)

        key = cv2.waitKey(1)

        if key == 27 or key == ord("q"):
            break

        time.sleep(0.001)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()