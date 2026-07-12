from logic.checkout_state import CheckoutState


def test_customer_visit_emits_one_start_and_end(monkeypatch):
    clock = [10.0]
    monkeypatch.setattr("logic.checkout_state.time.time", lambda: clock[0])
    state = CheckoutState()

    clock[0] = 14.0
    state.update(True, False, [])
    assert not state.customer_arrived

    state.update(True, False, [])
    assert state.customer_arrived
    assert state.visit_started_at == 10.0

    clock[0] = 15.0
    state.update(False, False, [])
    assert state.customer_left
