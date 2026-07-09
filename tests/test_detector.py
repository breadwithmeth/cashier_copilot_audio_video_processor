import sys
from unittest.mock import MagicMock

# Mock ultralytics
mock_ultralytics = MagicMock()
mock_yolo = MagicMock()
mock_ultralytics.YOLO = mock_yolo
sys.modules['ultralytics'] = mock_ultralytics

# Mock shapely
mock_shapely = MagicMock()
mock_shapely_geometry = MagicMock()

class MockPolygon:
    def __init__(self, coords):
        self.coords = coords
    def intersects(self, point):
        # By default intersect, but we can override or keep it simple
        return True

class MockPoint:
    def __init__(self, x, y):
        self.x = x
        self.y = y

mock_shapely_geometry.Polygon = MockPolygon
mock_shapely_geometry.Point = MockPoint

sys.modules['shapely'] = mock_shapely
sys.modules['shapely.geometry'] = mock_shapely_geometry

import unittest
from unittest.mock import patch
from pathlib import Path
import json

from cashier_av_processor.detector import YoloDetector, ALLOWED_EVENT_TYPES
from cashier_av_processor.db import DatabaseClient

class TestDetector(unittest.TestCase):
    def setUp(self):
        self.mock_db = MagicMock(spec=DatabaseClient)
        # Mock load_active_cameras return value
        # id, roi_config, pos_id
        self.mock_db.fetchall.return_value = [
            ("cam-01", '{"scanner_zone": [[0,0], [10,0], [10,10], [0,10]], "cash_drawer_zone": [[11,0], [20,0], [20,10], [11,10]]}', "pos-01")
        ]
        
        # Instantiate detector
        self.detector = YoloDetector(
            model_path="mock_yolo.pt",
            db_pool=self.mock_db,
            spool_dir="test_spool",
        )

    def test_normalize_token(self):
        self.assertEqual(self.detector._normalize_token("  Scanner Zone  "), "scanner_zone")
        self.assertEqual(self.detector._normalize_token("Item-Return"), "item_return")

    def test_class_classifiers(self):
        self.assertTrue(self.detector._is_customer("customer"))
        self.assertTrue(self.detector._is_customer("person"))
        self.assertFalse(self.detector._is_customer("hand"))

        self.assertTrue(self.detector._is_cashier("cashier"))
        self.assertTrue(self.detector._is_cashier("employee"))
        self.assertFalse(self.detector._is_cashier("item"))

        self.assertTrue(self.detector._is_item("item"))
        self.assertTrue(self.detector._is_item("product"))

        self.assertTrue(self.detector._is_phone("phone"))
        self.assertTrue(self.detector._is_phone("qr_code"))

        self.assertTrue(self.detector._is_document("passport"))
        self.assertTrue(self.detector._is_document("id_card"))

    def test_event_type_for(self):
        # 1. hand in cash_drawer_zone -> hand_to_drawer
        self.assertEqual(
            self.detector._event_type_for("hand", "cash_drawer_zone"),
            "hand_to_drawer"
        )
        # 2. item in bagging_zone -> item_in_bag
        self.assertEqual(
            self.detector._event_type_for("item", "bagging_zone"),
            "item_in_bag"
        )
        # 3. phone in scanner_zone -> phone_scanned_by_cashier
        self.assertEqual(
            self.detector._event_type_for("phone", "scanner_zone"),
            "phone_scanned_by_cashier"
        )
        # 4. document in customer_zone -> document_presented
        self.assertEqual(
            self.detector._event_type_for("document", "customer_zone"),
            "document_presented"
        )
        # 5. hand in scanner_zone -> hand_to_scanner
        self.assertEqual(
            self.detector._event_type_for("hand", "scanner_zone"),
            "hand_to_scanner"
        )
        # 6. item in scanner_zone -> item_return
        self.assertEqual(
            self.detector._event_type_for("item", "scanner_zone"),
            "item_return"
        )
        # 7. Unmatched
        self.assertIsNone(self.detector._event_type_for("unknown", "unknown"))

    def test_should_emit_cooldowns(self):
        camera_id = "cam-01"
        event_type = "item_in_bag"
        object_key = "track-1"
        
        # 1. First emission: True
        self.assertTrue(self.detector._should_emit(camera_id, event_type, object_key, 1000))
        # 2. Same object, same event, immediately after: False
        self.assertFalse(self.detector._should_emit(camera_id, event_type, object_key, 2000))
        # 3. Different object: True
        self.assertTrue(self.detector._should_emit(camera_id, event_type, "track-2", 2000))
        # 4. Same object after cooldown (5000ms for item_in_bag): True
        self.assertTrue(self.detector._should_emit(camera_id, event_type, object_key, 7000))

    def test_emit_presence_state_events(self):
        camera_id = "cam-01"
        self.detector.camera_state[camera_id] = {"customer_present": False, "cashier_present": False}
        
        # Mock insert_cv_event
        self.detector.store.insert_cv_event = MagicMock()

        # Customer becomes present
        self.detector._emit_presence_state_events(
            camera_id=camera_id,
            timestamp_ms=1000,
            inference_time_ms=10.0,
            customer_detected=True,
            cashier_detected=False,
            has_customer_zone=True,
            has_cashier_zone=True,
        )
        self.detector.store.insert_cv_event.assert_called_once()
        event = self.detector.store.insert_cv_event.call_args[0][0]
        self.assertEqual(event.event_type, "customer_present")
        self.assertTrue(self.detector.camera_state[camera_id]["customer_present"])
        
        # Reset mock
        self.detector.store.insert_cv_event.reset_mock()

        # Cashier becomes present
        self.detector._emit_presence_state_events(
            camera_id=camera_id,
            timestamp_ms=2000,
            inference_time_ms=10.0,
            customer_detected=True,
            cashier_detected=True,
            has_customer_zone=True,
            has_cashier_zone=True,
        )
        self.detector.store.insert_cv_event.assert_called_once()
        event = self.detector.store.insert_cv_event.call_args[0][0]
        self.assertEqual(event.event_type, "cashier_present")
        self.assertTrue(self.detector.camera_state[camera_id]["cashier_present"])

        # Reset mock
        self.detector.store.insert_cv_event.reset_mock()

        # Cashier still present -> triggers heartbeat `cashier_present`
        self.detector._emit_presence_state_events(
            camera_id=camera_id,
            timestamp_ms=3000, # inside cooldown (20s)
            inference_time_ms=10.0,
            customer_detected=True,
            cashier_detected=True,
            has_customer_zone=True,
            has_cashier_zone=True,
        )
        # Should be ignored because of cooldown (heartbeat checks _should_emit)
        self.detector.store.insert_cv_event.assert_not_called()

        # After cooldown
        self.detector._emit_presence_state_events(
            camera_id=camera_id,
            timestamp_ms=25000, # 23s later
            inference_time_ms=10.0,
            customer_detected=True,
            cashier_detected=True,
            has_customer_zone=True,
            has_cashier_zone=True,
        )
        self.detector.store.insert_cv_event.assert_called_once()
        event = self.detector.store.insert_cv_event.call_args[0][0]
        self.assertEqual(event.event_type, "cashier_present")
