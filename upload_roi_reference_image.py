from __future__ import annotations

import argparse
import json
import mimetypes
import time
import uuid
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import cv2

from config import (
    ANALYTICS_API_BASE_URL,
    ANALYTICS_API_KEY,
    CAMERA_CODE,
    ROI_REFERENCE_CAPTURE_TIMEOUT,
    ROI_REFERENCE_UPLOAD_TIMEOUT,
    STREAMS,
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _image_dimensions(image_bytes: bytes) -> tuple[int, int]:
    import numpy as np

    array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Cannot decode image bytes")
    height, width = image.shape[:2]
    return width, height


def _capture_camera_jpeg(camera_code: str, timeout: float) -> tuple[bytes, int, int]:
    cfg = STREAMS[camera_code]
    capture = cv2.VideoCapture(cfg["url"], cv2.CAP_FFMPEG)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    deadline = time.monotonic() + timeout

    try:
        while time.monotonic() < deadline:
            ok, frame = capture.read()
            if not ok or frame is None:
                time.sleep(0.1)
                continue

            height, width = frame.shape[:2]
            encoded, buffer = cv2.imencode(
                ".jpg",
                frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), 92],
            )
            if not encoded:
                raise RuntimeError("Failed to encode camera frame as JPEG")
            return buffer.tobytes(), width, height
    finally:
        capture.release()

    raise TimeoutError(f"No frame received from {camera_code} in {timeout}s")


def _multipart_body(
    fields: dict[str, str],
    file_field: str,
    file_name: str,
    file_bytes: bytes,
    mime_type: str,
) -> tuple[bytes, str]:
    boundary = f"----cashier-copilot-{uuid.uuid4().hex}"
    chunks: list[bytes] = []

    for name, value in fields.items():
        chunks.extend(
            [
                f"--{boundary}\r\n".encode("utf-8"),
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode(
                    "utf-8"
                ),
                str(value).encode("utf-8"),
                b"\r\n",
            ]
        )

    chunks.extend(
        [
            f"--{boundary}\r\n".encode("utf-8"),
            (
                f'Content-Disposition: form-data; name="{file_field}"; '
                f'filename="{file_name}"\r\n'
            ).encode("utf-8"),
            f"Content-Type: {mime_type}\r\n\r\n".encode("utf-8"),
            file_bytes,
            b"\r\n",
            f"--{boundary}--\r\n".encode("utf-8"),
        ]
    )

    return b"".join(chunks), boundary


def upload_roi_reference_image(
    camera_code: str,
    image_bytes: bytes,
    width: int,
    height: int,
    mime_type: str,
    captured_at: str,
) -> dict:
    url = (
        f"{ANALYTICS_API_BASE_URL}/analytics/cameras/"
        f"{camera_code}/roi-reference-image"
    )
    extension = mimetypes.guess_extension(mime_type) or ".jpg"
    body, boundary = _multipart_body(
        fields={
            "width": str(width),
            "height": str(height),
            "capturedAt": captured_at,
        },
        file_field="file",
        file_name=f"roi-reference{extension}",
        file_bytes=image_bytes,
        mime_type=mime_type,
    )
    request = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Content-Length": str(len(body)),
            "x-api-key": ANALYTICS_API_KEY,
        },
        method="POST",
    )

    with urllib.request.urlopen(
        request,
        timeout=ROI_REFERENCE_UPLOAD_TIMEOUT,
    ) as response:
        response_body = response.read().decode("utf-8", errors="replace")

    return json.loads(response_body)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upload a camera ROI reference image to analytics backend."
    )
    parser.add_argument("--camera", default=CAMERA_CODE, choices=sorted(STREAMS))
    parser.add_argument("--image", type=Path)
    parser.add_argument("--captured-at", default=_utc_now_iso())
    args = parser.parse_args()

    if args.image is not None:
        image_bytes = args.image.read_bytes()
        width, height = _image_dimensions(image_bytes)
        mime_type = mimetypes.guess_type(args.image.name)[0] or "image/jpeg"
    else:
        image_bytes, width, height = _capture_camera_jpeg(
            args.camera,
            ROI_REFERENCE_CAPTURE_TIMEOUT,
        )
        mime_type = "image/jpeg"

    result = upload_roi_reference_image(
        camera_code=args.camera,
        image_bytes=image_bytes,
        width=width,
        height=height,
        mime_type=mime_type,
        captured_at=args.captured_at,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
