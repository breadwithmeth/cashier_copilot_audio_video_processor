from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Sequence

from config import (
    ANALYTICS_API_BASE_URL,
    ANALYTICS_API_KEY,
    ANALYTICS_ROI_FETCH_TIMEOUT,
)
from vision.roi import roi_bounds


ROI_KEY_MAP = {
    "scanRoi": "scan_roi",
    "customerRoi": "customer_roi",
    "cashierRoi": "cashier_roi",
}


def fetch_camera_rois(camera_code: str) -> dict | None:
    url = (
        f"{ANALYTICS_API_BASE_URL}/analytics/cameras/"
        f"{urllib.parse.quote(camera_code)}/rois"
    )
    request = urllib.request.Request(
        url,
        headers={"x-api-key": ANALYTICS_API_KEY},
        method="GET",
    )

    try:
        with urllib.request.urlopen(
            request,
            timeout=ANALYTICS_ROI_FETCH_TIMEOUT,
        ) as response:
            body = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as error:
        error_body = error.read().decode("utf-8", errors="replace")
        print(
            f"[{camera_code}] ROI fetch HTTP error {error.code}: "
            f"{error_body[:300]}"
        )
        return None
    except (OSError, urllib.error.URLError) as error:
        print(f"[{camera_code}] ROI fetch error: {error}")
        return None

    return json.loads(body)


def apply_backend_rois(camera_code: str, cfg: dict) -> dict:
    payload = fetch_camera_rois(camera_code)
    if not payload:
        return cfg

    rois = payload.get("rois") or {}
    reference_image = payload.get("referenceImage") or {}
    image_size = _reference_image_size(reference_image)
    print(
        f"[{camera_code}] Backend ROI payload:\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )
    print(
        f"[{camera_code}] Backend ROI raw values:\n"
        f"{json.dumps(rois, ensure_ascii=False, indent=2)}"
    )
    updated = dict(cfg)
    applied = []

    for backend_key, config_key in ROI_KEY_MAP.items():
        raw_roi = rois.get(backend_key)
        roi = _parse_roi(raw_roi, image_size=image_size)
        print(
            f"[{camera_code}] ROI {backend_key}: "
            f"raw={json.dumps(raw_roi, ensure_ascii=False)} parsed={roi}"
        )
        if roi is None:
            continue

        if config_key in {"customer_roi", "cashier_roi"}:
            roi = roi_bounds(roi)
            print(f"[{camera_code}] ROI {backend_key}: rectangle={roi}")

        updated[config_key] = roi
        applied.append(config_key)

    if applied:
        print(f"[{camera_code}] Loaded ROI from backend: {', '.join(applied)}")
    else:
        print(f"[{camera_code}] Backend ROI response has no usable ROI values")

    return updated


def _reference_image_size(reference_image):
    width = reference_image.get("width")
    height = reference_image.get("height")
    if not width or not height:
        return None

    return int(width), int(height)


def _parse_roi(value, image_size=None):
    if value in (None, [], {}):
        return None

    if isinstance(value, dict):
        for key in ("points", "polygon", "vertices"):
            roi = _parse_points(value.get(key), image_size=image_size)
            if roi is not None:
                return roi

        if all(key in value for key in ("x1", "y1", "x2", "y2")):
            return (
                int(value["x1"]),
                int(value["y1"]),
                int(value["x2"]),
                int(value["y2"]),
            )

        if all(key in value for key in ("x", "y", "width", "height")):
            x = int(value["x"])
            y = int(value["y"])
            return (
                x,
                y,
                x + int(value["width"]),
                y + int(value["height"]),
            )

        return None

    if _is_roi_record_list(value):
        return _parse_roi(value[0], image_size=image_size)

    return _parse_points(value, image_size=image_size)


def _parse_points(value, image_size=None):
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return None

    if len(value) == 4 and all(isinstance(item, (int, float)) for item in value):
        x1, y1, x2, y2 = value
        return int(x1), int(y1), int(x2), int(y2)

    points = []
    for item in value:
        point = _parse_point(item, image_size=image_size)
        if point is None:
            return None
        points.append(point)

    return points if len(points) >= 3 else None


def _parse_point(value, image_size=None):
    if isinstance(value, dict) and "x" in value and "y" in value:
        return _scale_point(float(value["x"]), float(value["y"]), image_size)

    if (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes))
        and len(value) >= 2
    ):
        return _scale_point(float(value[0]), float(value[1]), image_size)

    return None


def _is_roi_record_list(value):
    return (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes))
        and len(value) == 1
        and isinstance(value[0], dict)
        and any(key in value[0] for key in ("points", "polygon", "vertices"))
    )


def _scale_point(x, y, image_size):
    if image_size is not None and 0 <= x <= 1 and 0 <= y <= 1:
        width, height = image_size
        return int(round(x * width)), int(round(y * height))

    return int(round(x)), int(round(y))
