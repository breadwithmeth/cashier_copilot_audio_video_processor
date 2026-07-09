import sys
from unittest.mock import MagicMock

# Mock psycopg2 and its submodules before importing db
mock_psycopg2 = MagicMock()
mock_psycopg2_pool = MagicMock()
mock_psycopg2_extras = MagicMock()
mock_psycopg2_extras.Json = lambda x: x

sys.modules['psycopg2'] = mock_psycopg2
sys.modules['psycopg2.pool'] = mock_psycopg2_pool
sys.modules['psycopg2.extras'] = mock_psycopg2_extras

import unittest
from unittest.mock import patch
import json
from pathlib import Path
import os
import shutil

from cashier_av_processor.db import (
    DatabaseClient,
    DurableEventStore,
    CvEvent,
    SpeechTranscript,
    VideoExportTask,
)

class TestDatabaseAndSpool(unittest.TestCase):
    def setUp(self):
        self.spool_dir = Path("test_spool")
        self.spool_dir.mkdir(parents=True, exist_ok=True)
        self.mock_db = MagicMock(spec=DatabaseClient)

    def tearDown(self):
        if self.spool_dir.exists():
            shutil.rmtree(self.spool_dir)

    def test_coerce_payload(self):
        # 1. Dict input
        self.assertEqual(DurableEventStore._coerce_payload({"a": 1}), {"a": 1})
        # 2. String input
        self.assertEqual(DurableEventStore._coerce_payload('{"a": 2}'), {"a": 2})
        # 3. None input
        self.assertEqual(DurableEventStore._coerce_payload(None), {})
        # 4. Other types (e.g. tuple list)
        self.assertEqual(DurableEventStore._coerce_payload([("a", 3)]), {"a": 3})

    def test_extract_timestamp_ms(self):
        keys = ("start_timestamp_ms", "start_ts")
        # Direct key
        payload = {"start_timestamp_ms": 1000}
        self.assertEqual(DurableEventStore._extract_timestamp_ms(payload, keys), 1000)
        
        # Alias key
        payload = {"start_ts": 2000}
        self.assertEqual(DurableEventStore._extract_timestamp_ms(payload, keys), 2000)

        # Embedded clip key
        payload = {"clip": {"start_timestamp_ms": 3000}}
        self.assertEqual(DurableEventStore._extract_timestamp_ms(payload, keys), 3000)

        # Not found
        payload = {"something_else": 4000}
        self.assertIsNone(DurableEventStore._extract_timestamp_ms(payload, keys))

    def test_poll_clip_task(self):
        store = DurableEventStore(self.mock_db, self.spool_dir)
        
        # Case 1: No task returned from database
        self.mock_db.fetchone.return_value = None
        self.assertIsNone(store.poll_clip_task())

        # Case 2: Valid task returned
        # RETURNING id, task_type, camera_id, violation_id, payload
        self.mock_db.fetchone.return_value = (
            123,
            "video_export",
            "cam-01",
            456,
            '{"start_timestamp_ms": 1000, "end_timestamp_ms": 2000}',
        )
        task = store.poll_clip_task()
        self.assertIsNotNone(task)
        self.assertEqual(task.id, 123)
        self.assertEqual(task.task_type, "video_export")
        self.assertEqual(task.camera_id, "cam-01")
        self.assertEqual(task.violation_id, 456)
        self.assertEqual(task.start_timestamp_ms, 1000)
        self.assertEqual(task.end_timestamp_ms, 2000)

    def test_insert_cv_event_success(self):
        store = DurableEventStore(self.mock_db, self.spool_dir)
        event = CvEvent(
            camera_id="cam-01",
            event_type="item_in_bag",
            timestamp_ms=1000,
            confidence=0.9,
            model_name="yolo",
            weights_version="v1",
            inference_time_ms=15.5,
            bbox_jsonb={"box": [1,2,3,4]},
            snapshot_path="snap.jpg",
        )
        store.insert_cv_event(event)
        self.mock_db.execute.assert_called_once()
        # Verify no files were created in spool dir
        self.assertEqual(len(list(self.spool_dir.glob("*.jsonl"))), 0)

    def test_insert_cv_event_failure_spools(self):
        self.mock_db.execute.side_effect = Exception("DB error")
        store = DurableEventStore(self.mock_db, self.spool_dir)
        event = CvEvent(
            camera_id="cam-01",
            event_type="item_in_bag",
            timestamp_ms=1000,
            confidence=0.9,
            model_name="yolo",
            weights_version="v1",
            inference_time_ms=15.5,
            bbox_jsonb={"box": [1,2,3,4]},
            snapshot_path="snap.jpg",
        )
        store.insert_cv_event(event)
        
        # Check that file was created
        spool_files = list(self.spool_dir.glob("*.jsonl"))
        self.assertEqual(len(spool_files), 1)
        
        # Read the spooled content
        content = spool_files[0].read_text(encoding="utf-8")
        data = json.loads(content.splitlines()[0])
        self.assertEqual(data["kind"], "cv_event")
        self.assertEqual(data["payload"]["camera_id"], "cam-01")
        self.assertEqual(data["payload"]["event_type"], "item_in_bag")

    def test_replay_pending(self):
        store = DurableEventStore(self.mock_db, self.spool_dir)
        
        event_payload = {
            "camera_id": "cam-01",
            "event_type": "item_in_bag",
            "timestamp_ms": 1000,
            "confidence": 0.9,
            "model_name": "yolo",
            "weights_version": "v1",
            "inference_time_ms": 15.5,
            "bbox_jsonb": {"box": [1,2,3,4]},
            "snapshot_path": "snap.jpg",
        }
        transcript_payload = {
            "pos_id": "pos-01",
            "transcript": "hello",
            "timestamp_ms": 2000,
            "duration_ms": 500,
            "confidence": 0.95,
            "model_name": "whisper",
            "weights_version": "small",
        }
        
        # Write to spool file manually
        with store.spool_path.open("w", encoding="utf-8") as fh:
            fh.write(json.dumps({"kind": "cv_event", "payload": event_payload}) + "\n")
            fh.write(json.dumps({"kind": "speech_transcript", "payload": transcript_payload}) + "\n")

        # Replay when database client succeeds
        store.replay_pending()
        self.assertEqual(self.mock_db.execute.call_count, 2)
        # Spool file should be deleted after successful replay
        self.assertFalse(store.spool_path.exists())

    def test_replay_pending_partial_failure(self):
        store = DurableEventStore(self.mock_db, self.spool_dir)
        
        event_payload = {
            "camera_id": "cam-01",
            "event_type": "item_in_bag",
            "timestamp_ms": 1000,
            "confidence": 0.9,
            "model_name": "yolo",
            "weights_version": "v1",
            "inference_time_ms": 15.5,
            "bbox_jsonb": {"box": [1,2,3,4]},
            "snapshot_path": "snap.jpg",
        }
        transcript_payload = {
            "pos_id": "pos-01",
            "transcript": "hello",
            "timestamp_ms": 2000,
            "duration_ms": 500,
            "confidence": 0.95,
            "model_name": "whisper",
            "weights_version": "small",
        }

        # Write to spool file manually
        with store.spool_path.open("w", encoding="utf-8") as fh:
            fh.write(json.dumps({"kind": "cv_event", "payload": event_payload}) + "\n")
            fh.write(json.dumps({"kind": "speech_transcript", "payload": transcript_payload}) + "\n")

        # Let the first execute succeed, and the second raise an error
        self.mock_db.execute.side_effect = [None, Exception("DB error again")]
        
        store.replay_pending()
        
        # One call succeeded, second failed. Spool file should still exist and contain the failed event
        self.assertTrue(store.spool_path.exists())
        lines = store.spool_path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 1)
        data = json.loads(lines[0])
        self.assertEqual(data["kind"], "speech_transcript")
