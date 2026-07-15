from logic.checkout_state import CheckoutStatus
from analytics_violations import VideoViolationMonitor


class _State:
    customer_is_present = True
    cashier_is_present = False
    customer_present_since = 100.0
    visit_started_at = 100.0
    customer_left = False
    cashier_absent_since = None
    scan_objects = []
    status = CheckoutStatus.NO_CASHIER

    def __init__(self, wait_seconds):
        self.wait_seconds = wait_seconds

    def get_customer_wait_seconds(self):
        return self.wait_seconds

    def get_cashier_absent_seconds(self):
        return 0.0


class _CollectingMonitor(VideoViolationMonitor):
    def __init__(self):
        self.sent = []
        super().__init__(
            "cam10",
            waiting_threshold_seconds=20.0,
            cooldown_seconds=300.0,
        )

    def _send_event(self, event):
        self.sent.append(event)
        return {"status": "sent"}


def test_customer_waiting_violation_is_sent_after_threshold(monkeypatch):
    clock = [119.0]
    monkeypatch.setattr("analytics_violations.time.time", lambda: clock[0])
    monitor = _CollectingMonitor()

    monitor.evaluate(_State(wait_seconds=19.9))
    assert monitor.sent == []

    clock[0] = 120.0
    monitor.evaluate(_State(wait_seconds=20.0))
    monitor.stop()

    assert len(monitor.sent) == 1
    assert monitor.sent[0]["code"] == "CUSTOMER_WAITING_TOO_LONG"


def test_customer_waiting_violation_respects_cooldown(monkeypatch):
    clock = [120.0]
    monkeypatch.setattr("analytics_violations.time.time", lambda: clock[0])
    monitor = _CollectingMonitor()

    monitor.evaluate(_State(wait_seconds=20.0))
    clock[0] = 121.0
    monitor.evaluate(_State(wait_seconds=21.0))
    monitor.stop()

    assert len(monitor.sent) == 1


def test_customer_waiting_violation_resets_for_new_visit(monkeypatch):
    clock = [120.0]
    monkeypatch.setattr("analytics_violations.time.time", lambda: clock[0])
    monitor = _CollectingMonitor()

    monitor.evaluate(_State(wait_seconds=20.0))

    next_state = _State(wait_seconds=20.0)
    next_state.customer_present_since = 500.0
    next_state.visit_started_at = 500.0
    clock[0] = 520.0
    monitor.evaluate(next_state)
    monitor.stop()

    assert len(monitor.sent) == 2


def test_cashier_absent_during_service_requires_cashier_seen_in_same_visit(monkeypatch):
    clock = [130.0]
    monkeypatch.setattr("analytics_violations.time.time", lambda: clock[0])
    monitor = _CollectingMonitor()

    state = _State(wait_seconds=30.0)
    state.cashier_absent_since = 115.0
    state.get_cashier_absent_seconds = lambda: 15.0
    monitor.evaluate(state)
    assert [event["code"] for event in monitor.sent] == ["CUSTOMER_WAITING_TOO_LONG"]

    cashier_present = _State(wait_seconds=31.0)
    cashier_present.cashier_is_present = True
    cashier_present.status = CheckoutStatus.SERVICE_STARTED
    monitor.evaluate(cashier_present)

    clock[0] = 145.0
    state.cashier_absent_since = 135.0
    state.get_cashier_absent_seconds = lambda: 10.0
    monitor.evaluate(state)
    monitor.stop()

    assert "CASHIER_ABSENT_DURING_SERVICE" in [event["code"] for event in monitor.sent]
