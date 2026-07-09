def clip_roi(roi, frame):
    x1, y1, x2, y2 = roi
    h, w = frame.shape[:2]

    return (
        max(0, min(x1, w - 1)),
        max(0, min(y1, h - 1)),
        max(0, min(x2, w)),
        max(0, min(y2, h)),
    )


def crop_roi(frame, roi):
    x1, y1, x2, y2 = clip_roi(roi, frame)
    return frame[y1:y2, x1:x2], (x1, y1, x2, y2)


def offset_bbox(bbox, offset_x, offset_y):
    x1, y1, x2, y2 = bbox

    return (
        int(x1 + offset_x),
        int(y1 + offset_y),
        int(x2 + offset_x),
        int(y2 + offset_y),
    )