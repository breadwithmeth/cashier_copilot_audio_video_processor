import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from logic.checkout_state import CheckoutStatus
from vision.roi import is_rectangle_roi, roi_bounds


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
        subtitle="",
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

        if subtitle:
            image = self._draw_subtitle(image, subtitle)

        return image

    def _draw_subtitle(self, image, text):
        h, w = image.shape[:2]
        font_size = max(22, min(36, w // 55))
        font_paths = (
            "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        )
        font = None
        for path in font_paths:
            try:
                font = ImageFont.truetype(path, font_size)
                break
            except OSError:
                continue
        font = font or ImageFont.load_default()
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        canvas = Image.fromarray(rgb)
        draw = ImageDraw.Draw(canvas, "RGBA")
        panel_width = max(360, int(w * 0.38))
        x1, x2 = w - panel_width - 20, w - 20
        padding = 20
        max_text_width = panel_width - padding * 2

        lines = []
        for paragraph in text.splitlines():
            current = ""
            for word in paragraph.split():
                candidate = f"{current} {word}".strip()
                if draw.textlength(candidate, font=font) <= max_text_width:
                    current = candidate
                else:
                    if current:
                        lines.append(current)
                    current = word
            if current:
                lines.append(current)

        line_height = font_size + 10
        max_lines = max(1, (h - 80) // line_height)
        lines = lines[-max_lines:]
        panel_height = len(lines) * line_height + padding * 2
        y1 = max(20, h - panel_height - 20)
        draw.rounded_rectangle((x1, y1, x2, h - 20), radius=14,
                               fill=(0, 0, 0, 190))
        draw.text((x1 + padding, y1 + padding), "\n".join(lines),
                  font=font, fill=(255, 255, 255, 255),
                  spacing=10)
        return cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR)

    def _draw_roi(self, image, roi, label, color):
        x1, y1, x2, y2 = roi_bounds(roi)

        if is_rectangle_roi(roi):
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        else:
            points = np.array(roi, dtype=np.int32)
            cv2.polylines(image, [points], isClosed=True, color=color, thickness=2)

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
            class_name = obj.class_name.replace("_", " ")
            polygon = getattr(obj, "polygon", None)

            if polygon and len(polygon) >= 3:
                points = np.array(polygon, dtype=np.int32)
                cv2.polylines(
                    image,
                    [points],
                    isClosed=True,
                    color=(0, 255, 0),
                    thickness=2,
                )
            else:
                cv2.rectangle(
                    image,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2,
                )

            track_label = (
                f"ID {obj.track_id} " if obj.track_id is not None else ""
            )
            cv2.putText(
                image,
                f"{track_label}{class_name} {obj.confidence:.2f}",
                (x1, max(25, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2,
            )

    def _draw_hands(self, image, person):
        colors = {"left": (0, 165, 255), "right": (255, 100, 0)}
        for hand in person.hands or []:
            color = colors[hand.side]
            points = (hand.shoulder, hand.elbow, hand.wrist)
            for first, second in zip(points, points[1:]):
                if first is not None and second is not None:
                    cv2.line(image, first, second, color, 4)
            for point in points:
                if point is not None:
                    cv2.circle(image, point, 6, color, -1)
            if hand.wrist is not None:
                wx, wy = hand.wrist
                cv2.putText(
                    image,
                    f"{hand.side}: {hand.position}",
                    (wx + 8, max(25, wy - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
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

            self._draw_hands(image, person)

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
            f"scan {scan_result.process_ms}ms | person {person_result.process_ms}ms",
            (20, 140),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2,
        )

        counter = state.product_counter
        count_lines = [
            f"OBJECTS: {counter.total} | visible: {counter.visible_count}",
        ]

        for line_index, line in enumerate(count_lines):
            cv2.putText(
                image,
                line,
                (20, 175 + line_index * 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
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
