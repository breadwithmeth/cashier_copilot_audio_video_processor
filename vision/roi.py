import cv2
import numpy as np


def is_rectangle_roi(roi):
    return len(roi) == 4 and all(isinstance(value, (int, float)) for value in roi)


def roi_bounds(roi):
    if is_rectangle_roi(roi):
        x1, y1, x2, y2 = roi
        return int(x1), int(y1), int(x2), int(y2)

    if len(roi) < 3:
        raise ValueError("Polygon ROI must contain at least three points")

    xs = [point[0] for point in roi]
    ys = [point[1] for point in roi]
    return int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))


def clip_roi(roi, frame):
    x1, y1, x2, y2 = roi_bounds(roi)
    h, w = frame.shape[:2]

    return (
        max(0, min(x1, w - 1)),
        max(0, min(y1, h - 1)),
        max(0, min(x2, w)),
        max(0, min(y2, h)),
    )


def crop_roi(frame, roi):
    x1, y1, x2, y2 = clip_roi(roi, frame)
    roi_frame = frame[y1:y2, x1:x2]

    if is_rectangle_roi(roi) or roi_frame.size == 0:
        return roi_frame, (x1, y1, x2, y2)

    local_points = np.array(
        [[int(px - x1), int(py - y1)] for px, py in roi],
        dtype=np.int32,
    )
    mask = np.zeros(roi_frame.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [local_points], 255)

    return cv2.bitwise_and(roi_frame, roi_frame, mask=mask), (x1, y1, x2, y2)


def bbox_center_in_roi(bbox, roi, offset_x=0, offset_y=0):
    if is_rectangle_roi(roi):
        return True

    x1, y1, x2, y2 = bbox
    center = (
        float((x1 + x2) / 2 + offset_x),
        float((y1 + y2) / 2 + offset_y),
    )
    polygon = np.array(roi, dtype=np.float32)

    return cv2.pointPolygonTest(polygon, center, False) >= 0


def offset_bbox(bbox, offset_x, offset_y):
    x1, y1, x2, y2 = bbox

    return (
        int(x1 + offset_x),
        int(y1 + offset_y),
        int(x2 + offset_x),
        int(y2 + offset_y),
    )
