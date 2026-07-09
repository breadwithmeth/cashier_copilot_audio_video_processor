import time
from enum import Enum

from config import CUSTOMER_TIMEOUT, CASHIER_TIMEOUT


class CheckoutStatus(str, Enum):
    IDLE = "IDLE"
    CUSTOMER_WAITING = "CUSTOMER_WAITING"
    SERVICE_STARTED = "SERVICE_STARTED"
    NO_CASHIER = "NO_CASHIER"


class CheckoutState:
    def __init__(self):
        self.status = CheckoutStatus.IDLE

        self.customer_detected = False
        self.cashier_detected = False

        self.customer_present_since = None
        self.cashier_present_since = None

        self.customer_is_present = False
        self.cashier_is_present = False

        self.no_cashier_alarm = False

        self.scan_objects = []

        self.last_event = None
        self.last_event_time = None

    def update(
        self,
        customer_detected: bool,
        cashier_detected: bool,
        scan_objects: list,
    ):
        now = time.time()

        self.customer_detected = customer_detected
        self.cashier_detected = cashier_detected
        self.scan_objects = scan_objects

        self._update_customer_state(now)
        self._update_cashier_state(now)
        self._update_status(now)

    def _update_customer_state(self, now: float):
        if self.customer_detected:
            if self.customer_present_since is None:
                self.customer_present_since = now

            if now - self.customer_present_since >= CUSTOMER_TIMEOUT:
                if not self.customer_is_present:
                    self._set_event("CUSTOMER_PRESENT")

                self.customer_is_present = True

        else:
            self.customer_present_since = None

            if self.customer_is_present:
                self._set_event("CUSTOMER_LEFT")

            self.customer_is_present = False

    def _update_cashier_state(self, now: float):
        if self.cashier_detected:
            if self.cashier_present_since is None:
                self.cashier_present_since = now

            if now - self.cashier_present_since >= CASHIER_TIMEOUT:
                if not self.cashier_is_present:
                    self._set_event("CASHIER_PRESENT")

                self.cashier_is_present = True

        else:
            self.cashier_present_since = None

            if self.cashier_is_present:
                self._set_event("CASHIER_LEFT")

            self.cashier_is_present = False

    def _update_status(self, now: float):
        previous_status = self.status

        customer_waiting_long = (
            self.customer_present_since is not None
            and now - self.customer_present_since >= CUSTOMER_TIMEOUT
        )

        if customer_waiting_long and not self.cashier_is_present:
            self.status = CheckoutStatus.NO_CASHIER
            self.no_cashier_alarm = True

        elif self.customer_is_present and self.cashier_is_present:
            self.status = CheckoutStatus.SERVICE_STARTED
            self.no_cashier_alarm = False

        elif self.customer_is_present:
            self.status = CheckoutStatus.CUSTOMER_WAITING
            self.no_cashier_alarm = False

        else:
            self.status = CheckoutStatus.IDLE
            self.no_cashier_alarm = False

        if self.status != previous_status:
            self._set_event(f"STATE_CHANGED:{previous_status}->{self.status}")

    def _set_event(self, event: str):
        self.last_event = event
        self.last_event_time = time.time()
        print(event)

    def get_customer_wait_seconds(self) -> float:
        if self.customer_present_since is None:
            return 0.0

        return time.time() - self.customer_present_since

    def get_cashier_absent_seconds(self) -> float:
        if self.cashier_detected:
            return 0.0

        return time.time() - self.last_event_time if self.last_event_time else 0.0