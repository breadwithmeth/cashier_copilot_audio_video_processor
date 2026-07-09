from __future__ import annotations

import json
import logging
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from .config import AppConfig

logger = logging.getLogger(__name__)


def register_analytics_stream(config: AppConfig, status: str = "online") -> bool:
    """Send current analytics stream URL to the Go backend."""
    if not config.analytics_api_base_url or not config.analytics_api_key:
        logger.info("Analytics stream registration skipped: backend URL or API key is not configured")
        return False

    if not config.analytics_stream_url:
        logger.info("Analytics stream registration skipped: stream URL is not configured")
        return False

    camera_id = quote(config.camera_id, safe="")
    endpoint = (
        f"{config.analytics_api_base_url.rstrip('/')}"
        f"/api/v1/analytics/cameras/{camera_id}/stream"
    )
    payload = {
        "analytics_stream_url": config.analytics_stream_url,
        "analytics_stream_type": config.analytics_stream_type,
        "analytics_stream_status": status,
    }
    request = Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "X-API-Key": config.analytics_api_key,
        },
        method="POST",
    )

    try:
        with urlopen(request, timeout=config.analytics_register_timeout_s) as response:
            response.read()
        logger.info(
            "Registered analytics stream camera=%s status=%s url=%s",
            config.camera_id,
            status,
            config.analytics_stream_url,
        )
        return True
    except HTTPError as exc:
        logger.warning(
            "Analytics stream registration failed camera=%s status=%s http_status=%s",
            config.camera_id,
            status,
            exc.code,
        )
    except URLError as exc:
        logger.warning(
            "Analytics stream registration failed camera=%s status=%s reason=%s",
            config.camera_id,
            status,
            exc.reason,
        )
    except Exception:
        logger.exception("Analytics stream registration failed camera=%s status=%s", config.camera_id, status)

    return False
