import unittest

from logic.product_counter import ProductCounter
from models.detection import Detection


def detection(class_name, bbox, track_id=None):
    return Detection(
        class_name=class_name,
        confidence=0.9,
        bbox=bbox,
        track_id=track_id,
    )


class ProductCounterTest(unittest.TestCase):
    def test_same_product_is_counted_once_across_frames(self):
        counter = ProductCounter()

        self.assertEqual(counter.update([detection("bottle", (10, 10, 50, 80))]), 1)
        self.assertEqual(counter.update([detection("bottle", (12, 11, 52, 81))]), 0)
        self.assertEqual(counter.total, 1)

    def test_two_products_are_counted(self):
        counter = ProductCounter()

        counter.update([
            detection("bottle", (10, 10, 50, 80)),
            detection("bottle", (100, 10, 140, 80)),
        ])

        self.assertEqual(counter.total, 2)
        self.assertEqual(counter.class_counts, {"bottle": 2})

    def test_moving_product_with_changed_class_is_not_counted_twice(self):
        counter = ProductCounter()

        counter.update([detection("bottle", (10, 10, 50, 80))])
        counter.update([detection("unknown", (70, 10, 110, 80))])

        self.assertEqual(counter.total, 1)

    def test_product_can_be_counted_again_after_leaving(self):
        counter = ProductCounter(max_missed_frames=1)
        product = detection("bottle", (10, 10, 50, 80))

        counter.update([product])
        counter.update([])
        counter.update([])
        counter.update([product])

        self.assertEqual(counter.total, 2)

    def test_reset_clears_count_and_tracks(self):
        counter = ProductCounter()
        counter.update([detection("bottle", (10, 10, 50, 80))])

        counter.reset()

        self.assertEqual(counter.total, 0)
        self.assertEqual(counter.class_counts, {})

    def test_tracker_id_keeps_fast_moving_object_stable(self):
        counter = ProductCounter(max_center_distance=10)
        counter.update([detection("object", (0, 0, 20, 20), track_id=7)])
        newly_counted = counter.update([
            detection("object", (500, 500, 520, 520), track_id=7)
        ])
        self.assertEqual(newly_counted, 0)
        self.assertEqual(counter.total, 1)

    def test_different_tracker_ids_are_different_objects(self):
        counter = ProductCounter()
        counter.update([detection("object", (0, 0, 20, 20), track_id=7)])
        counter.update([detection("object", (1, 1, 21, 21), track_id=8)])
        self.assertEqual(counter.total, 2)

    def test_new_tracker_id_after_detection_gap_is_not_recounted(self):
        counter = ProductCounter()
        counter.update([detection("object", (10, 10, 50, 50), track_id=7)])
        counter.update([])
        newly_counted = counter.update([
            detection("object", (12, 11, 52, 51), track_id=19)
        ])
        self.assertEqual(newly_counted, 0)
        self.assertEqual(counter.total, 1)

    def test_ignored_class_is_not_counted(self):
        counter = ProductCounter(ignored_classes={"barcode_scanner"})

        newly_counted = counter.update([
            detection("barcode_scanner", (0, 0, 100, 100)),
            detection("plastic_bottle", (120, 0, 180, 100)),
        ])

        self.assertEqual(newly_counted, 1)
        self.assertEqual(counter.total, 1)
        self.assertEqual(counter.visible_count, 1)
        self.assertEqual(counter.class_counts, {"plastic_bottle": 1})


if __name__ == "__main__":
    unittest.main()
