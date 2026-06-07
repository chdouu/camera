import csv
import os
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import speed_plate_camera as camera


class CoreIntegrationTests(unittest.TestCase):
    def test_flat_config_parser_handles_example_values(self):
        data = camera.parse_flat_config(
            """
            camera_source: picamera2
            plate_enabled: true
            fps: 30
            plate_scan_interval: 0.5
            telegram_token: ""
            """
        )

        self.assertEqual(data["camera_source"], "picamera2")
        self.assertIs(data["plate_enabled"], True)
        self.assertEqual(data["fps"], 30)
        self.assertEqual(data["plate_scan_interval"], 0.5)
        self.assertEqual(data["telegram_token"], "")

    def test_cli_help_mentions_capturetest(self):
        self.assertIn("capturetest", camera.RUN_MODES)
        self.assertIn("platetest", camera.RUN_MODES)

    def test_config_relative_paths_resolve_from_config_directory(self):
        previous_cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as tmpdir, tempfile.TemporaryDirectory() as otherdir:
            config_dir = Path(tmpdir)
            (config_dir / "models").mkdir()
            (config_dir / "models" / "model.onnx").write_bytes(b"model")
            config_file = config_dir / "config.yaml"
            config_file.write_text(
                "plate_model: models/model.onnx\n"
                "tesseract_cmd: bin/tesseract\n",
                encoding="utf-8",
            )

            os.chdir(otherdir)
            try:
                cfg = camera.Config.load(config_file)
                self.assertEqual(cfg.plate_model, (config_dir / "models" / "model.onnx").resolve())
                self.assertEqual(cfg.tesseract_cmd, (config_dir / "bin" / "tesseract").resolve())
                self.assertEqual(cfg.logs_dir, config_dir / "logs")
                self.assertEqual(cfg.preview_file, config_dir / "preview.jpg")
            finally:
                os.chdir(previous_cwd)

    def test_tesseract_bare_command_is_not_resolved_as_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir)
            config_file = config_dir / "config.yaml"
            config_file.write_text("tesseract_cmd: tesseract\n", encoding="utf-8")

            cfg = camera.Config.load(config_file)

            self.assertEqual(cfg.tesseract_cmd, "tesseract")

    def test_speed_summary_and_plate_helpers(self):
        speeds = np.array([10.0, 20.0, 22.0, 21.0, 11.0])
        areas = np.array([1000.0, 2500.0, 2600.0, 2550.0, 1200.0])

        mean_speed, avg_area, sd_speed, sd_area, confidence = camera.summarize_event(speeds, areas)

        self.assertAlmostEqual(mean_speed, 21.0)
        self.assertAlmostEqual(avg_area, 2550.0)
        self.assertGreater(confidence, 0)
        self.assertGreater(sd_speed, 0)
        self.assertGreater(sd_area, 0)
        self.assertEqual(camera.newest_event_plate([{"plate": ""}, {"plate": "CJ 12 XYZ"}]), "CJ 12 XYZ")
        self.assertTrue(camera.PlateFormatRO.is_valid("CJ12XYZ"))
        self.assertTrue(camera.PlateFormatTW.is_valid("ABC1234"))
        self.assertTrue(camera.PlateFormatTW.is_valid("1234AB"))
        self.assertTrue(camera.PlateFormatTW.is_valid("ABC123"))
        self.assertTrue(camera.PlateFormatTW.is_valid("AB1234"))
        self.assertEqual(camera.PlateFormatTW.normalize("ABC1234"), "ABC-1234")
        self.assertEqual(camera.PlateFormatTW.normalize("1234AB"), "1234-AB")

    def test_plate_formatter_uses_taiwan_by_default(self):
        cfg = camera.Config.load("config.yaml.example")

        self.assertEqual(cfg.plate_country, "TW")
        self.assertIs(camera.get_plate_formatter(cfg.plate_country), camera.PlateFormatTW)

    def test_plate_recognizer_uses_taiwan_formatter_for_ocr_text(self):
        recognizer = camera.PlateRecognizer.__new__(camera.PlateRecognizer)
        recognizer.plate_formatter = camera.PlateFormatTW

        self.assertEqual(recognizer.plate_formatter.normalize("ABC1234"), "ABC-1234")
        self.assertTrue(recognizer.plate_formatter.is_valid("ABC1234"))

    def test_only_tracking_state_updates_speed(self):
        self.assertTrue(camera.should_update_tracking(camera.TRACKING))
        self.assertFalse(camera.should_update_tracking(camera.WAITING))
        self.assertFalse(camera.should_update_tracking(camera.SAVING))

    def test_onnx_backend_check_runs_blank_inference(self):
        self.assertTrue(camera.has_onnx_backend(Path("models/yolov8n-license_plate.onnx")))

    def test_opencv_dnn_inference_output_can_be_postprocessed(self):
        cfg = camera.Config()
        cfg.plate_model = Path("models/yolov8n-license_plate.onnx")
        cfg.plate_enabled = True
        cfg.tesseract_cmd = ""

        recognizer = camera.PlateRecognizer.__new__(camera.PlateRecognizer)
        recognizer.enabled = True
        recognizer.backend = "opencv-dnn"
        recognizer.net = camera.cv2.dnn.readNetFromONNX(str(cfg.plate_model))
        recognizer.output_names = recognizer.net.getUnconnectedOutLayersNames()
        recognizer.image_size = 512
        recognizer.confidence = 0.25
        recognizer.iou = 0.7

        outputs = recognizer._run_inference(np.zeros((512, 512, 3), dtype=np.uint8))
        detections = recognizer._postprocess(outputs)

        self.assertIsInstance(outputs, list)
        self.assertEqual(outputs[0].shape[0], 1)
        self.assertIsInstance(detections, list)

    def test_plate_recognizer_processes_blank_frame_without_camera(self):
        cfg = camera.Config.load("config.yaml.example")
        cfg.plate_scan_interval = 0
        cfg.plate_ocr_interval = 0

        recognizer = camera.PlateRecognizer(cfg)
        result = recognizer.process_frame(np.zeros((512, 512, 3), dtype=np.uint8))

        self.assertTrue(recognizer.enabled)
        self.assertIsNone(result)
        self.assertEqual(recognizer.recent_plate(), "")

    def test_camera_dependency_check_for_opencv(self):
        self.assertTrue(camera.camera_dependency_available("opencv"))

    def test_picamera2_config_retries_without_framerate_for_uvc_camera(self):
        class FakePicamera2:
            def __init__(self):
                self.create_calls = []
                self.configure_calls = []

            def create_video_configuration(self, **kwargs):
                self.create_calls.append(kwargs)
                return kwargs

            def configure(self, config):
                self.configure_calls.append(config)
                if config.get("controls"):
                    raise RuntimeError("Control FrameDurationLimits is not advertised by libcamera")

        cfg = camera.Config()
        fake_camera = FakePicamera2()

        config = camera.configure_picamera2_video(fake_camera, cfg)

        self.assertEqual(len(fake_camera.create_calls), 2)
        self.assertEqual(fake_camera.create_calls[0]["controls"], {"FrameRate": cfg.fps})
        self.assertNotIn("controls", fake_camera.create_calls[1])
        self.assertNotIn("controls", config)
        self.assertEqual(config["main"], {"format": "RGB888", "size": (cfg.image_width, cfg.image_height)})

    def test_tesseract_check_accepts_configured_executable(self):
        self.assertTrue(camera.tesseract_available(sys.executable))

    def test_systemd_template_replacement_contract(self):
        template = Path("speed-plate-camera.service").read_text(encoding="utf-8")
        rendered = (
            template
            .replace("@SERVICE_USER@", "camerauser")
            .replace("@INSTALL_DIR@", "/opt/speed-plate")
        )

        self.assertNotIn("@SERVICE_USER@", rendered)
        self.assertNotIn("@INSTALL_DIR@", rendered)
        self.assertIn("User=camerauser", rendered)
        self.assertIn("WorkingDirectory=/opt/speed-plate", rendered)
        self.assertIn(
            "ExecStart=/opt/speed-plate/venv/bin/python /opt/speed-plate/speed_plate_camera.py --config /opt/speed-plate/config.yaml",
            rendered,
        )

    def test_makefile_install_uses_install_dir_for_runtime_paths(self):
        makefile = Path("Makefile").read_text(encoding="utf-8")

        self.assertIn('test -x "$(INSTALL_DIR)/venv/bin/python"', makefile)
        self.assertIn('"$(INSTALL_DIR)/venv/bin/pip" install -r "$(INSTALL_DIR)/requirements.txt"', makefile)
        self.assertIn('test -f "$(INSTALL_DIR)/config.yaml"', makefile)
        self.assertIn('mkdir -p "$(INSTALL_DIR)/logs"', makefile)
        self.assertIn('"$(INSTALL_DIR)/venv" "$(INSTALL_DIR)/logs" "$(INSTALL_DIR)/config.yaml"', makefile)
        self.assertIn('"$(INSTALL_DIR)/speed-plate-camera.service" > /etc/systemd/system/speed-plate-camera.service', makefile)
        self.assertIn('"$(INSTALL_DIR)/speed_plate_camera.py" preview --config "$(INSTALL_DIR)/config.yaml"', makefile)
        self.assertIn('"$(INSTALL_DIR)/speed_plate_camera.py" selftest --config "$(INSTALL_DIR)/config.yaml"', makefile)
        self.assertIn('"$(INSTALL_DIR)/speed_plate_camera.py" capturetest --config "$(INSTALL_DIR)/config.yaml"', makefile)
        self.assertIn('"$(INSTALL_DIR)/speed_plate_camera.py" platetest --config "$(INSTALL_DIR)/config.yaml"', makefile)
        self.assertIn('unittest discover -s "$(INSTALL_DIR)/tests"', makefile)
        self.assertIn('rm -rf "$(INSTALL_DIR)/logs"', makefile)
        self.assertIn('rm -rf "$(INSTALL_DIR)/venv"', makefile)
        self.assertIn('rm -f "$(INSTALL_DIR)/preview.jpg"', makefile)

    def test_recorder_writes_speed_event_with_plate_column(self):
        cfg = camera.Config()
        cfg.min_speed = 1
        cfg.min_speed_alert = 999
        cfg.min_area = 1
        cfg.min_confidence = 1
        cfg.min_confidence_alert = 999
        cfg.telegram_token = ""
        cfg.telegram_chat_id = ""

        previous_cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as tmpdir:
            os.chdir(tmpdir)
            try:
                Path("logs").mkdir()
                recorder = camera.Recorder(cfg)
                recorded = recorder.record(
                    confidence=95,
                    image=None,
                    timestamp=datetime(2026, 5, 27, tzinfo=timezone.utc),
                    mean_speed=42,
                    avg_area=2500,
                    sd_speed=2,
                    sd_area=100,
                    speeds=np.array([40.0, 42.0, 44.0]),
                    secs=1.25,
                    direction=camera.LEFT_TO_RIGHT,
                    events=[],
                    plate="CJ 12 XYZ",
                )

                self.assertTrue(recorded)
                with open("logs/recorded_speed.csv", newline="", encoding="utf-8") as csv_file:
                    rows = list(csv.reader(csv_file))
                self.assertEqual(rows[0], camera.Recorder.record_headers.split(","))
                self.assertEqual(rows[1][-2:], ["LTR", "CJ 12 XYZ"])
            finally:
                os.chdir(previous_cwd)


if __name__ == "__main__":
    unittest.main()
