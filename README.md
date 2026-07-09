# Cashier Copilot Audio/Video Processor

Multiprocess daemon for RTSP video analytics, violation clip export, and Russian speech-to-text processing.

## Status

This repository contains the audio/video processor service only. It expects PostgreSQL, camera records, RTSP access, FFmpeg, OpenCV, YOLO weights, and Faster-Whisper runtime dependencies to be available in the target environment.

Database credentials are stored in local `.env`. That file is intentionally ignored by git and must not be copied into documentation, commits, logs, or tickets. Use `.env.example` as the public template.

## Architecture

The daemon starts four isolated Python processes through `multiprocessing`:

1. `VideoGrabberProcess`
   Captures RTSP video, stores a configurable RAM ring buffer, JPEG-compresses every frame, and sends every fifth frame to the inference queue.

2. `YoloInferenceProcess`
   Reads frames from `inference_queue`, runs YOLO on the configured device, loads ROI polygons from PostgreSQL, checks object locations with Shapely, and inserts rows into `cv_events`.

3. `ClipExporterProcess`
   Polls `tasks` for pending `video_export` jobs, requests the relevant buffered frames from the video process through IPC queues, and writes MP4 files.

4. `AudioSttProcess`
   Captures audio with FFmpeg, applies WebRTC VAD, sends speech segments to Faster-Whisper, and inserts rows into `speech_transcripts`.

The clip worker cannot directly read the `deque` owned by the video process because process memory is isolated. Clip extraction is implemented with `ClipRequest` and `ClipResponse` messages over `multiprocessing.Queue`.

## Process Map

```text
RTSP video
  -> VideoGrabberProcess
      -> RAM deque[(timestamp_ms, jpeg_bytes)]
      -> inference_queue every 5th frame
      -> clip request/response queues

inference_queue
  -> YoloInferenceProcess
      -> YOLOv11 CPU/GPU
      -> Shapely ROI checks
      -> PostgreSQL cv_events

PostgreSQL tasks
  -> ClipExporterProcess
      -> ClipRequest to VideoGrabberProcess
      -> MP4 in CLIP_OUTPUT_DIR
      -> tasks.result_path

RTSP audio or local microphone
  -> AudioSttProcess
      -> FFmpeg PCM 16 kHz mono
      -> webrtcvad
      -> Faster-Whisper
      -> PostgreSQL speech_transcripts
```

## Files

```text
cashier_av_processor/
  audio_stt.py       FFmpeg capture, VAD, Faster-Whisper worker
  clip_exporter.py   video_export polling and MP4 export
  config.py          environment and .env loading
  daemon.py          process orchestration and shutdown
  db.py              SQL, connection pool, durable spool
  detector.py        YOLO and Shapely ROI analysis
  messages.py        IPC dataclasses
  video.py           RTSP capture and JPEG ring buffer
```

## Requirements

Python:

- Python 3.10+
- `opencv-python`
- `numpy`
- `psycopg2-binary`
- `ultralytics`
- `shapely`
- `faster-whisper`
- `webrtcvad`

System:

- FFmpeg available in `PATH`
- CPU runtime by default, or NVIDIA GPU/CUDA if `YOLO_DEVICE=cuda` and `WHISPER_DEVICE=cuda`
- Network access to PostgreSQL and camera RTSP endpoints
- Write access to `CLIP_OUTPUT_DIR` and `SPOOL_DIR`

Install:

```bash
pip install -e .
```

## Configuration

The service reads environment variables and also auto-loads a local `.env` file from the current working directory.

Use `.env.example` as a template:

```bash
cp .env.example .env
```

Required values:

```dotenv
DB_HOST=<postgres-host>
DB_PORT=5432
DB_SSLMODE=disable
DB_NAME=<database-name>
DB_USER=<database-user>
DB_PASSWORD=<database-password>

CAMERA_ID=<camera-id>
RTSP_URL=rtsp://<user>:<password>@<host>:<port>/<path>
POS_ID=<pos-id>
```

Optional values:

```dotenv
YOLO_MODEL_PATH=weights/best.pt
YOLO_DEVICE=cpu
CLIP_OUTPUT_DIR=clips
SPOOL_DIR=spool
VIDEO_FPS=25
INFERENCE_STRIDE=25
BUFFER_SECONDS=120
JPEG_QUALITY=85
INFERENCE_QUEUE_SIZE=64
TASK_POLL_INTERVAL_S=1.0
CLIP_RESPONSE_TIMEOUT_S=15.0
YOLO_PROFILE_INTERVAL_S=10.0
WHISPER_MODEL_NAME=small
WHISPER_DEVICE=cpu
WHISPER_COMPUTE_TYPE=int8
ANALYTICS_API_BASE_URL=http://127.0.0.1:8085
ANALYTICS_API_KEY=<analytics-api-key>
ANALYTICS_STREAM_BASE_URL=http://127.0.0.1:8888
ANALYTICS_STREAM_URL=
ANALYTICS_STREAM_TYPE=hls
ANALYTICS_REGISTER_TIMEOUT_S=5
```

`DB_DSN` is also supported instead of separate `DB_HOST`, `DB_PORT`, `DB_SSLMODE`, `DB_NAME`, `DB_USER`, and `DB_PASSWORD`.

If `ANALYTICS_STREAM_URL` is empty, it is derived from `ANALYTICS_STREAM_BASE_URL` and `CAMERA_ID`:

```text
<ANALYTICS_STREAM_BASE_URL>/<CAMERA_ID>/index.m3u8
```

Audio source:

- Empty `AUDIO_SOURCE` means the worker uses `RTSP_URL`.
- `AUDIO_SOURCE=rtsp://...` uses a separate RTSP audio stream.
- `AUDIO_SOURCE=mic::0` uses macOS AVFoundation device `:0`.
- `AUDIO_SOURCE=mic:default` uses Linux PulseAudio default input.

## Run

```bash
python -m cashier_av_processor
```

or:

```bash
cashier-av-daemon
```

Stop with `Ctrl+C` or `SIGTERM`. The parent process sets a shared stop event, waits for workers, and terminates workers that do not exit in time.

## Backend Stream Registration

The daemon can notify the Go backend where the analytics overlay stream is available. On startup it sends `online`; during normal shutdown it sends `offline`.

Endpoint:

```text
POST /api/v1/analytics/cameras/<camera_id>/stream
```

Headers:

```text
X-API-Key: <analytics-api-key>
Content-Type: application/json
```

Payload:

```json
{
  "analytics_stream_url": "http://127.0.0.1:8888/<camera_id>/index.m3u8",
  "analytics_stream_type": "hls",
  "analytics_stream_status": "online"
}
```

Manual check:

```bash
STREAM_URL="${ANALYTICS_STREAM_URL:-$ANALYTICS_STREAM_BASE_URL/$CAMERA_ID/index.m3u8}"
curl -X POST "$ANALYTICS_API_BASE_URL/api/v1/analytics/cameras/$CAMERA_ID/stream" \
  -H "X-API-Key: $ANALYTICS_API_KEY" \
  -H "Content-Type: application/json" \
  -d "{\"analytics_stream_url\":\"$STREAM_URL\",\"analytics_stream_type\":\"$ANALYTICS_STREAM_TYPE\",\"analytics_stream_status\":\"online\"}"
```

The stream registration request is best-effort: failure to reach the backend is logged but does not stop video capture or analytics workers.

## Database Schema

The current database contains these public tables:

- `cameras`
- `cv_events`
- `pos_events`
- `speech_transcripts`
- `tasks`
- `upsell_rules`
- `violations`

No custom enum types or views were present when the schema was inspected.

## Backend Event Contract

The Go Rule Engine currently expects only these `cv_events.event_type` values:

| Event Type | Meaning |
| --- | --- |
| `item_in_bag` | Item moved into bagging or packing zone. |
| `hand_to_drawer` | Cashier hand near cash drawer. |
| `phone_scanned_by_cashier` | Cashier scans their own phone or QR. |
| `document_presented` | Customer document or passport visible. |
| `item_return` | Item moved back toward cashier/scanner area. |
| `hand_to_scanner` | Hand returned toward scanner/cashier area. |
| `customer_present` | Customer detected in service zone. |
| `customer_left` | Customer leaves service zone. |
| `no_cashier` | Cashier absent from workplace. |
| `cashier_present` | Cashier present at workplace. |

The detector must not insert arbitrary event names such as `<class>_in_<zone>` unless the backend contract is changed.

Recommended `bbox_jsonb` shape:

```json
{
  "bbox": [120, 80, 260, 220],
  "class_name": "item",
  "track_id": "track-123",
  "roi": "bag_zone",
  "frame_id": 45678,
  "extra": {
    "direction": "scanner_to_bag"
  }
}
```

Presence events such as `customer_present`, `customer_left`, `cashier_present`, and `no_cashier` are emitted on state changes. `cashier_present` and `no_cashier` may also be emitted as low-rate heartbeat events.

### cameras

| Column | Type | Null | Default |
| --- | --- | --- | --- |
| `id` | `varchar` | no | |
| `ip_address` | `varchar` | no | |
| `username` | `varchar` | no | |
| `password` | `varchar` | no | |
| `pos_id` | `varchar` | no | |
| `status` | `varchar` | no | `'inactive'` |
| `roi_config` | `jsonb` | no | `{}` |
| `created_at` | `timestamptz` | yes | `CURRENT_TIMESTAMP` |

The daemon loads active cameras with:

```sql
SELECT id, roi_config, pos_id FROM cameras WHERE status = 'active';
```

Expected `roi_config` shape:

```json
{
  "scanner_zone": [[0, 0], [100, 0], [100, 100], [0, 100]],
  "cash_drawer_zone": [[120, 0], [220, 0], [220, 100], [120, 100]],
  "packing_zone": [[0, 120], [100, 120], [100, 220], [0, 220]],
  "customer_zone": [[120, 120], [220, 120], [220, 220], [120, 220]]
}
```

### cv_events

| Column | Type | Null |
| --- | --- | --- |
| `id` | `bigint` | no |
| `camera_id` | `varchar` | no |
| `event_type` | `varchar` | no |
| `timestamp_ms` | `bigint` | no |
| `confidence` | `double precision` | no |
| `model_name` | `varchar` | no |
| `weights_version` | `varchar` | no |
| `inference_time_ms` | `integer` | no |
| `bbox_jsonb` | `jsonb` | no |
| `snapshot_path` | `varchar` | no |

Insert used by the daemon:

```sql
INSERT INTO cv_events (
    camera_id,
    event_type,
    timestamp_ms,
    confidence,
    model_name,
    weights_version,
    inference_time_ms,
    bbox_jsonb,
    snapshot_path
)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s);
```

`snapshot_path` is `NOT NULL` in the database, so the daemon writes an empty string when no snapshot file is available.

### speech_transcripts

| Column | Type | Null |
| --- | --- | --- |
| `id` | `bigint` | no |
| `pos_id` | `varchar` | no |
| `transcript` | `text` | no |
| `timestamp_ms` | `bigint` | no |
| `duration_ms` | `integer` | no |
| `confidence` | `double precision` | no |
| `model_name` | `varchar` | no |
| `weights_version` | `varchar` | no |

Insert used by the daemon:

```sql
INSERT INTO speech_transcripts (
    pos_id,
    transcript,
    timestamp_ms,
    duration_ms,
    confidence,
    model_name,
    weights_version
)
VALUES (%s, %s, %s, %s, %s, %s, %s);
```

`confidence` is `NOT NULL`, so missing model confidence is stored as `0.0`.

### tasks

| Column | Type | Null | Default |
| --- | --- | --- | --- |
| `id` | `bigint` | no | |
| `task_type` | `varchar` | no | |
| `camera_id` | `varchar` | no | |
| `violation_id` | `bigint` | yes | |
| `payload` | `jsonb` | no | `{}` |
| `status` | `varchar` | no | `'pending'` |
| `result_path` | `varchar` | yes | |
| `error_message` | `text` | yes | |
| `created_at` | `timestamptz` | yes | `CURRENT_TIMESTAMP` |
| `updated_at` | `timestamptz` | yes | `CURRENT_TIMESTAMP` |
| `processed_at` | `timestamptz` | yes | |

Task polling and claiming:

```sql
WITH next_task AS (
    SELECT id
    FROM tasks
    WHERE status = 'pending' AND task_type = 'video_export'
    ORDER BY created_at ASC
    FOR UPDATE SKIP LOCKED
    LIMIT 1
)
UPDATE tasks
SET status = 'processing',
    updated_at = CURRENT_TIMESTAMP
WHERE id IN (SELECT id FROM next_task)
RETURNING id, task_type, camera_id, violation_id, payload;
```

Expected `video_export` payload:

```json
{
  "start_timestamp_ms": 1780000000000,
  "end_timestamp_ms": 1780000005000
}
```

Supported aliases are also accepted:

- start: `start_timestamp_ms`, `start_ts`, `start_ms`, `from_timestamp_ms`, `from_ms`
- end: `end_timestamp_ms`, `end_ts`, `end_ms`, `to_timestamp_ms`, `to_ms`

Completion:

```sql
UPDATE tasks
SET status = 'completed',
    result_path = %s,
    error_message = NULL,
    updated_at = CURRENT_TIMESTAMP
WHERE id = %s;
```

Failure:

```sql
UPDATE tasks
SET status = 'failed',
    error_message = %s,
    updated_at = CURRENT_TIMESTAMP
WHERE id = %s;
```

The video server must not update `processed_at`; that field is reserved for Go backend acknowledgement.

### Other Tables

`pos_events` stores POS timeline events and has an index on `(timestamp_ms, pos_id)`.

`violations` links generated violations to optional `cv_events`, `pos_events`, and `speech_transcripts` rows.

`upsell_rules` stores keyword-based upsell suggestions.

## Indexes and Constraints

Primary keys:

- `cameras_pkey`
- `cv_events_pkey`
- `pos_events_pkey`
- `speech_transcripts_pkey`
- `tasks_pkey`
- `upsell_rules_pkey`
- `violations_pkey`

Operational indexes:

- `idx_cv_events_time_cam` on `cv_events(timestamp_ms, camera_id)`
- `idx_pos_events_time_pos` on `pos_events(timestamp_ms, pos_id)`
- `idx_speech_time_pos` on `speech_transcripts(timestamp_ms, pos_id)`
- `idx_tasks_status` on `tasks(status)`
- `idx_violations_time` on `violations(timestamp_ms)`

Foreign keys:

- `tasks.violation_id -> violations.id`
- `violations.cv_event_id -> cv_events.id`
- `violations.pos_event_id -> pos_events.id`
- `violations.speech_transcript_id -> speech_transcripts.id`

## Video Buffer

The video process keeps frames as JPEG bytes instead of raw NumPy arrays:

```python
cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
```

Default sizing:

- `VIDEO_FPS=25`
- `BUFFER_SECONDS=120`
- `deque(maxlen=3000)`
- `INFERENCE_STRIDE=5`, so YOLO receives about 5 FPS

If frame spacing exceeds the target interval, the process logs a capture lag warning and continues.

## YOLO and ROI Logic

The detector loads:

```python
YOLO(YOLO_MODEL_PATH).to(YOLO_DEVICE)
```

The local default is CPU:

```dotenv
YOLO_DEVICE=cpu
WHISPER_DEVICE=cpu
WHISPER_COMPUTE_TYPE=int8
```

`yolov11x.pt` is heavy for CPU inference. If real-time performance is not acceptable on the target machine, use a smaller compatible model through `YOLO_MODEL_PATH`.

For every detection box `[x1, y1, x2, y2]`, the daemon uses the bottom-center point:

```python
Point((x1 + x2) / 2.0, y2)
```

This point is checked against all configured ROI polygons. A `hand` class inside `cash_drawer_zone` becomes `hand_to_drawer`; an item in `bag_zone`, `bagging_zone`, or `packing_zone` becomes `item_in_bag`. Detections that do not map to the backend event contract are ignored.

Every 10 seconds by default, the YOLO worker logs average inference time. CUDA memory usage is logged only when `YOLO_DEVICE` starts with `cuda`.

## Clip Export

`ClipExporterProcess` writes files to `CLIP_OUTPUT_DIR` with this format:

```text
violation_<violation_id>_task_<task_id>_<camera_id>_<start_timestamp_ms>_<end_timestamp_ms>.mp4
```

Frames are decoded from JPEG bytes with OpenCV and written using:

```python
cv2.VideoWriter_fourcc(*"mp4v")
```

If requested frames are no longer in the ring buffer, the task is marked `failed`.

## Speech Processing

FFmpeg command shape:

```bash
ffmpeg -rtsp_transport tcp -i <source> -vn -acodec pcm_s16le -ar 16000 -ac 1 -f wav pipe:1
```

VAD defaults:

- `webrtcvad.Vad(3)`
- 30 ms frames
- speech starts when 90% of the last 300 ms are voiced
- segment closes after 1.5 seconds of silence
- max segment length is 30 seconds

Whisper defaults:

```python
WhisperModel("small", device="cpu", compute_type="int8")
```

Transcription language is fixed to Russian: `language="ru"`.

## Failure Handling

PostgreSQL insert failures do not crash workers. Failed `cv_events` and `speech_transcripts` inserts are appended to JSONL files under `SPOOL_DIR`.

The spool file name includes the worker PID:

```text
pending_events_<pid>.jsonl
```

Workers replay a limited number of spooled events after successful writes.

RTSP video read failures close the current OpenCV capture, sleep five seconds, and reconnect.

Audio capture failures terminate the FFmpeg subprocess, sleep five seconds, and reconnect.

## Smoke Checks

Syntax check:

```bash
python3 -m py_compile cashier_av_processor/*.py
```

Configuration check:

```bash
python -m cashier_av_processor
```

Without `CAMERA_ID` and `RTSP_URL`, the service should stop with a configuration error. After camera configuration is filled, the daemon will start workers and attempt RTSP, PostgreSQL, CUDA, and FFmpeg runtime work.

Database connectivity check with `psql`:

```bash
psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c "select current_database(), current_user, current_schema();"
```

## Security Notes

- Keep `.env` local and ignored by git.
- Do not put real database passwords or RTSP passwords in Markdown files.
- Avoid printing `DB_DSN`, `.env`, camera passwords, or RTSP URLs in logs.
- Rotate credentials if they were copied into chat, tickets, screenshots, commits, or logs.
