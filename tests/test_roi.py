import unittest

import numpy as np

from vision.roi import bbox_center_in_roi, crop_roi, roi_bounds


class RoiTest(unittest.TestCase):
    def test_rectangle_roi_bounds_are_unchanged(self):
        self.assertEqual(roi_bounds((10, 20, 30, 40)), (10, 20, 30, 40))

    def test_polygon_roi_bounds_are_calculated_from_points(self):
        roi = ((10, 40), (30, 20), (50, 60), (20, 70))

        self.assertEqual(roi_bounds(roi), (10, 20, 50, 70))

    def test_polygon_crop_masks_pixels_outside_polygon(self):
        frame = np.full((10, 10, 3), 255, dtype=np.uint8)
        roi = ((2, 2), (7, 2), (2, 7))

        cropped, clipped_roi = crop_roi(frame, roi)

        self.assertEqual(clipped_roi, (2, 2, 7, 7))
        self.assertTrue(np.array_equal(cropped[1, 1], [255, 255, 255]))
        self.assertTrue(np.array_equal(cropped[4, 4], [0, 0, 0]))

    def test_bbox_center_is_checked_against_polygon(self):
        roi = ((0, 0), (10, 0), (0, 10))

        self.assertTrue(bbox_center_in_roi((1, 1, 3, 3), roi))
        self.assertFalse(bbox_center_in_roi((8, 8, 10, 10), roi))


if __name__ == "__main__":
    unittest.main()
