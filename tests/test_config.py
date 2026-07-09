import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
import os

from cashier_av_processor.config import load_dotenv, AppConfig, _int_env, _float_env, _db_dsn_from_env

class TestConfig(unittest.TestCase):
    def setUp(self):
        self.env_patcher = patch.dict(os.environ, {}, clear=True)
        self.env_patcher.start()

    def tearDown(self):
        self.env_patcher.stop()

    @patch("cashier_av_processor.config.Path.exists")
    @patch("cashier_av_processor.config.Path.read_text")
    def test_load_dotenv(self, mock_read_text, mock_exists):
        mock_exists.return_value = True
        mock_read_text.return_value = (
            "# This is a comment\n"
            "DB_HOST=localhost\n"
            "DB_PORT=5432\n"
            "  SPACED_KEY  =  spaced_val \n"
            "QUOTED_VAL=\"quotes\"\n"
            "SINGLE_QUOTED_VAL='single'\n"
            "INVALID_LINE\n"
        )
        load_dotenv(Path(".env"))
        
        self.assertEqual(os.environ.get("DB_HOST"), "localhost")
        self.assertEqual(os.environ.get("DB_PORT"), "5432")
        self.assertEqual(os.environ.get("SPACED_KEY"), "spaced_val")
        self.assertEqual(os.environ.get("QUOTED_VAL"), "quotes")
        self.assertEqual(os.environ.get("SINGLE_QUOTED_VAL"), "single")

    def test_int_env(self):
        os.environ["INT_VAR"] = "42"
        self.assertEqual(_int_env("INT_VAR", 10), 42)
        self.assertEqual(_int_env("NON_EXISTENT", 10), 10)

    def test_float_env(self):
        os.environ["FLOAT_VAR"] = "3.14"
        self.assertEqual(_float_env("FLOAT_VAR", 1.0), 3.14)
        self.assertEqual(_float_env("NON_EXISTENT", 1.0), 1.0)

    def test_db_dsn_from_env(self):
        # 1. Explicit DSN
        os.environ["DB_DSN"] = "postgresql://user:pass@host:5432/dbname"
        self.assertEqual(_db_dsn_from_env(), "postgresql://user:pass@host:5432/dbname")

        # 2. Re-evaluate with parts
        del os.environ["DB_DSN"]
        os.environ["DB_HOST"] = "localhost"
        os.environ["DB_PORT"] = "5432"
        os.environ["DB_NAME"] = "testdb"
        os.environ["DB_USER"] = "postgres"
        os.environ["DB_PASSWORD"] = "password"
        os.environ["DB_SSLMODE"] = "require"
        
        expected_dsn = "host=localhost port=5432 dbname=testdb user=postgres password=password sslmode=require"
        self.assertEqual(_db_dsn_from_env(), expected_dsn)

        # 3. Missing fields
        del os.environ["DB_PASSWORD"]
        with self.assertRaises(ValueError):
            _db_dsn_from_env()

    @patch("cashier_av_processor.config.load_dotenv")
    def test_app_config_from_env_missing_required(self, mock_load_dotenv):
        # Missing CAMERA_ID and RTSP_URL
        with self.assertRaises(ValueError):
            AppConfig.from_env()

    @patch("cashier_av_processor.config.load_dotenv")
    def test_app_config_from_env_valid(self, mock_load_dotenv):
        os.environ["DB_DSN"] = "postgresql://mock"
        os.environ["CAMERA_ID"] = "cam-01"
        os.environ["RTSP_URL"] = "rtsp://test"
        
        config = AppConfig.from_env()
        self.assertEqual(config.camera_id, "cam-01")
        self.assertEqual(config.rtsp_url, "rtsp://test")
        self.assertEqual(config.pos_id, "cam-01")  # fallback to camera_id
        self.assertEqual(config.audio_source, "rtsp://test")  # fallback to rtsp_url
        self.assertEqual(config.fps, 25)  # default
