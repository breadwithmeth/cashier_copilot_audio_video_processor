import cv2

from logic.checkout_state import CheckoutStatus


class Overlay:
    def __init__(
        self,
        scan_roi,
        customer_roi,
        cashier_roi,
    ):
        self.scan_roi = scan_roi
        self.customer_roi = customer_roi
        self.cashier_roi = cashier_roi

    def draw(
        self,
        frame,
        state,
        scan_result,
        person_result,
    ):
        image = frame.copy()

        self._draw_roi(image, self.scan_roi, "scan_zone", (0, 255, 255))
        self._draw_roi(image, self.customer_roi, "customer_zone", (255, 0, 255))
        self._draw_roi(image, self.cashier_roi, "cashier_zone", (255, 255, 0))

        self._draw_scan_objects(image, scan_result.objects)
        self._draw_persons(image, person_result.persons)

        self._draw_top_info(image, state, scan_result, person_result)

        if state.status == CheckoutStatus.NO_CASHIER:
            self._draw_no_cashier_alarm(image, state)

        return image

    def _draw_roi(self, image, roi, label, color):
        x1, y1, x2, y2 = roi

        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        cv2.putText(
            image,
            label,
            (x1, max(30, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
        )

    def _draw_scan_objects(self, image, objects):
        for obj in objects:
            x1, y1, x2, y2 = obj.bbox

            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2,
            )

            cv2.putText(
                image,
                f"{obj.class_name} {obj.confidence:.2f}",
                (x1, max(25, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2,
            )

    def _draw_persons(self, image, persons):
        for person in persons:
            x1, y1, x2, y2 = person.bbox

            if person.role == "customer":
                color = (255, 0, 255)
                label = "customer"
            else:
                color = (255, 255, 0)
                label = "cashier"

            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                color,
                2,
            )

            cv2.putText(
                image,
                f"{label} {person.confidence:.2f}",
                (x1, max(25, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                color,
                2,
            )

    def _draw_top_info(self, image, state, scan_result, person_result):
        customer_text = "CUSTOMER: YES" if state.customer_is_present else "CUSTOMER: NO"
        cashier_text = "CASHIER: YES" if state.cashier_is_present else "CASHIER: NO"

        status_text = f"STATE: {state.status}"

        cv2.putText(
            image,
            status_text,
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            image,
            customer_text,
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 0, 255) if state.customer_is_present else (180, 180, 180),
            2,
        )

        cv2.putText(
            image,
            cashier_text,
            (20, 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0) if state.cashier_is_present else (180, 180, 180),
            2,
        )

        cv2.putText(
            image,
            f"scan {scan_result.process_ms}ms | customer {person_result.customer_ms}ms | cashier {person_result.cashier_ms}ms",
            (20, 140),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2,
        )

    def _draw_no_cashier_alarm(self, image, state):
        h, w = image.shape[:2]

        overlay = image.copy()

        cv2.rectangle(
            overlay,
            (0, 0),
            (w, 130),
            (0, 0, 255),
            -1,
        )

        cv2.addWeighted(
            overlay,
            0.45,
            image,
            0.55,
            0,
            image,
        )

        wait_seconds = state.get_customer_wait_seconds()

        cv2.putText(
            image,
            "НЕТ КАССИРА ЗА КАССОЙ",
            (30, 65),
            cv2.FONT_HERSHEY_DUPLEX,
            1.4,
            (255, 255, 255),
            3,
        )

        cv2.putText(
            image,
            f"Клиент ожидает: {wait_seconds:.1f} сек",
            (30, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
        )