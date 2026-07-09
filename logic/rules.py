from logic.checkout_state import CheckoutStatus


class RuleEngine:
    def __init__(self):
        self.active_events = []

    def evaluate(self, state):
        self.active_events = []

        if state.status == CheckoutStatus.NO_CASHIER:
            self.active_events.append({
                "type": "NO_CASHIER",
                "message": "Нет кассира за кассой",
                "level": "critical",
            })

        if state.customer_is_present and not state.cashier_is_present:
            self.active_events.append({
                "type": "CUSTOMER_WAITING",
                "message": "Клиент ожидает кассира",
                "level": "warning",
            })

        if len(state.scan_objects) > 0:
            self.active_events.append({
                "type": "OBJECT_IN_SCAN_ZONE",
                "message": "Объект в зоне сканирования",
                "level": "info",
            })

        return self.active_events