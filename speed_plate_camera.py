# speed_plate_camera.py
"""
Integrated Raspberry Pi speed camera and license plate recognition.

Usage:
    speed_plate_camera.py [preview|display|selftest|capturetest|platetest] [--config=<file>] [--image=<file>]

Options:
    -h --help       Show this screen.
    --config=<file> YAML config file [default: config.yaml].
"""

from datetime import datetime, timezone
from pathlib import Path
from multiprocessing import Process
import argparse
import glob
import json
import logging
import math
import re
import shutil
import subprocess
import time

import cv2
import numpy as np

try:
    import telegram
except ImportError:
    telegram = None


MIN_SAVE_BUFFER = 2
THRESHOLD = 25
BLURSIZE = (15, 15)
FT_PER_SECOND_TO_MPS = 0.3048
MPH_TO_MPS = 0.44704

WAITING = 0
TRACKING = 1
SAVING = 2
UNKNOWN = 0
LEFT_TO_RIGHT = 1
RIGHT_TO_LEFT = 2
RUN_MODES = ("preview", "display", "selftest", "capturetest", "platetest")


class Config:
    upper_left_x = 0
    upper_left_y = 0
    lower_right_x = 1024
    lower_right_y = 576
    l2r_distance = 65
    r2l_distance = 80
    fov = 62.2
    fps = 30
    image_width = 1024
    image_height = 576
    image_min_area = 500
    camera_source = "picamera2"
    opencv_camera_index = 0
    camera_vflip = False
    camera_hflip = False
    min_distance = 0.4
    min_speed = 4.5
    min_speed_alert = 13.4
    min_area = 2000
    min_confidence = 70
    min_confidence_alert = 90
    telegram_token = ""
    telegram_chat_id = ""
    telegram_frequency = 6
    plate_enabled = True
    plate_model = "models/yolov8n-license_plate.onnx"
    plate_country = "TW"
    plate_confidence = 0.25
    plate_iou = 0.7
    plate_scan_interval = 0.5
    plate_ocr_interval = 3.0
    plate_recent_seconds = 8.0
    tesseract_cmd = ""

    @staticmethod
    def load(config_file):
        cfg = Config()
        cfg.config_file = Path(config_file).resolve()
        cfg.base_dir = cfg.config_file.parent
        with open(config_file, "r", encoding="utf-8") as stream:
            data = load_yaml_like_config(stream)
            for key, value in data.items():
                if hasattr(cfg, key):
                    setattr(cfg, key, value)

        if cfg.upper_left_x > cfg.lower_right_x:
            cfg.upper_left_x, cfg.lower_right_x = cfg.lower_right_x, cfg.upper_left_x
        if cfg.upper_left_y > cfg.lower_right_y:
            cfg.upper_left_y, cfg.lower_right_y = cfg.lower_right_y, cfg.upper_left_y

        cfg.upper_left = (cfg.upper_left_x, cfg.upper_left_y)
        cfg.lower_right = (cfg.lower_right_x, cfg.lower_right_y)
        cfg.monitored_width = cfg.lower_right_x - cfg.upper_left_x
        cfg.monitored_height = cfg.lower_right_y - cfg.upper_left_y
        cfg.resolution = [cfg.image_width, cfg.image_height]
        cfg.plate_model = resolve_config_path(cfg.base_dir, cfg.plate_model)
        if cfg.tesseract_cmd:
            cfg.tesseract_cmd = resolve_executable_config_value(cfg.base_dir, cfg.tesseract_cmd)
        cfg.logs_dir = cfg.base_dir / "logs"
        cfg.preview_file = cfg.base_dir / "preview.jpg"
        return cfg


def resolve_config_path(base_dir, path_value):
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def resolve_executable_config_value(base_dir, value):
    value = str(value)
    if "/" not in value and "\\" not in value:
        return value
    return resolve_config_path(base_dir, value)


def load_yaml_like_config(stream):
    content = stream.read()
    try:
        import yaml
        return yaml.safe_load(content) or {}
    except ImportError:
        return parse_flat_config(content)


def parse_flat_config(content):
    data = {}
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        data[key.strip()] = parse_config_value(value.strip())
    return data


def parse_config_value(value):
    if "#" in value:
        value = value.split("#", 1)[0].strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        return value[1:-1]
    lower_value = value.lower()
    if lower_value == "true":
        return True
    if lower_value == "false":
        return False
    if lower_value in ("", "null", "none"):
        return ""
    try:
        if any(char in value for char in (".", "e", "E")):
            return float(value)
        return int(value)
    except ValueError:
        return value


class PlateFormatRO:
    prefixes = {
        "AB", "AR", "AG", "BC", "BH", "BN", "BR", "BT", "BV", "BZ",
        "CS", "CL", "CJ", "CT", "CV", "DB", "DJ", "GL", "GR", "GJ",
        "HR", "HD", "IL", "IS", "IF", "MM", "MH", "MS", "NT", "OT",
        "PH", "SM", "SJ", "SB", "SV", "TR", "TM", "TL", "VS", "VL",
        "VN", "B",
    }

    @staticmethod
    def normalize(plate):
        plate = plate.strip().upper().replace("-", "").replace(" ", "")
        if len(plate) == 7:
            if plate[0] == "B" and plate[1:4].isdigit() and plate[4:].isalpha():
                return "{} {} {}".format(plate[0], plate[1:4], plate[4:])
            if plate[:2] in PlateFormatRO.prefixes and plate[2:4].isdigit() and plate[4:].isalpha():
                return "{} {} {}".format(plate[:2], plate[2:4], plate[4:])
        return plate

    @staticmethod
    def is_valid(plate):
        plate = PlateFormatRO.normalize(" ".join(plate.strip().upper().split()))
        parts = plate.split(" ")
        if len(parts) != 3:
            return False
        prefix, digits, suffix = parts
        if prefix == "B":
            return len(digits) == 3 and digits.isdigit() and len(suffix) == 3 and suffix.isalpha()
        return (
            prefix in PlateFormatRO.prefixes
            and len(digits) == 2
            and digits.isdigit()
            and len(suffix) == 3
            and suffix.isalpha()
        )


class PlateFormatTW:
    patterns = (
        re.compile(r"^[A-Z]{3}\d{4}$"),
        re.compile(r"^\d{4}[A-Z]{2}$"),
        re.compile(r"^[A-Z]{3}\d{3}$"),
        re.compile(r"^[A-Z]{2}\d{4}$"),
        re.compile(r"^[A-Z]{2}\d{3}$"),
        re.compile(r"^\d{3}[A-Z]{2}$"),
    )

    @staticmethod
    def normalize(plate):
        plate = plate.strip().upper().replace("-", "").replace(" ", "")
        if re.match(r"^[A-Z]{3}\d{4}$", plate):
            return "{}-{}".format(plate[:3], plate[3:])
        if re.match(r"^\d{4}[A-Z]{2}$", plate):
            return "{}-{}".format(plate[:4], plate[4:])
        if re.match(r"^[A-Z]{3}\d{3}$", plate):
            return "{}-{}".format(plate[:3], plate[3:])
        if re.match(r"^[A-Z]{2}\d{4}$", plate):
            return "{}-{}".format(plate[:2], plate[2:])
        if re.match(r"^[A-Z]{2}\d{3}$", plate):
            return "{}-{}".format(plate[:2], plate[2:])
        if re.match(r"^\d{3}[A-Z]{2}$", plate):
            return "{}-{}".format(plate[:3], plate[3:])
        return plate

    @staticmethod
    def is_valid(plate):
        plate = plate.strip().upper().replace("-", "").replace(" ", "")
        return any(pattern.match(plate) for pattern in PlateFormatTW.patterns)


def get_plate_formatter(country):
    country = str(country or "TW").strip().upper()
    if country in ("TW", "TWN", "TAIWAN"):
        return PlateFormatTW
    if country in ("RO", "ROU", "ROMANIA"):
        return PlateFormatRO
    raise ValueError("unsupported plate_country: {}".format(country))


class PlateRecognizer:
    image_size = 512
    tesseract_config = "--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

    def __init__(self, cfg):
        self.enabled = bool(cfg.plate_enabled)
        self.confidence = float(cfg.plate_confidence)
        self.iou = float(cfg.plate_iou)
        self.scan_interval = float(cfg.plate_scan_interval)
        self.ocr_interval = float(cfg.plate_ocr_interval)
        self.recent_seconds = float(cfg.plate_recent_seconds)
        self.last_scan = 0.0
        self.last_ocr = 0.0
        self.last_plate = None
        self.last_plate_time = 0.0
        self.seen_boxes = {}
        self.clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8))
        try:
            self.plate_formatter = get_plate_formatter(cfg.plate_country)
        except ValueError as exc:
            logging.warning("Plate recognition disabled, %s", exc)
            self.enabled = False
            return

        if not self.enabled:
            return

        try:
            import pytesseract
        except ImportError as exc:
            logging.warning("Plate recognition disabled, missing dependency: %s", exc)
            self.enabled = False
            return

        if cfg.tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = str(cfg.tesseract_cmd)

        model_path = Path(cfg.plate_model)
        if not model_path.is_file():
            logging.warning("Plate recognition disabled, model not found: %s", model_path)
            self.enabled = False
            return

        self.pytesseract = pytesseract
        self.backend = None
        self.session = None
        self.net = None
        self.input_name = None
        self.output_names = None

        try:
            import onnxruntime as ort
            self.session = ort.InferenceSession(str(model_path))
            self.input_name = self.session.get_inputs()[0].name
            self.output_names = [out.name for out in self.session.get_outputs()]
            self.backend = "onnxruntime"
        except ImportError:
            logging.warning("ONNX Runtime not installed, trying OpenCV DNN backend")
        except Exception as exc:
            logging.warning("ONNX Runtime failed to load model, trying OpenCV DNN backend: %s", exc)

        if self.backend is None:
            try:
                self.net = cv2.dnn.readNetFromONNX(str(model_path))
                self.output_names = self.net.getUnconnectedOutLayersNames()
                self.backend = "opencv-dnn"
            except Exception as exc:
                logging.warning("Plate recognition disabled, no ONNX backend could load %s: %s", model_path, exc)
                self.enabled = False
                return

        logging.info("Plate recognition enabled with %s using %s", model_path, self.backend)

    def recent_plate(self):
        if self.last_plate and time.time() - self.last_plate_time <= self.recent_seconds:
            return self.last_plate
        return ""

    def process_frame(self, frame):
        if not self.enabled:
            return None

        now = time.time()
        if now - self.last_scan < self.scan_interval:
            return None
        self.last_scan = now

        outputs = self._run_inference(frame)
        detections = self._postprocess(outputs)
        height, width = frame.shape[:2]
        x_scale = width / self.image_size
        y_scale = height / self.image_size

        for detection in detections:
            box, crop = self._extract_plate_box(frame, detection, x_scale, y_scale)
            if crop is None or crop.size == 0:
                continue
            if box in self.seen_boxes and now - self.seen_boxes[box] < self.recent_seconds:
                continue
            if now - self.last_ocr < self.ocr_interval:
                continue

            plate = self._extract_valid_plate(crop)
            self.last_ocr = now
            if plate:
                self.last_plate = plate
                self.last_plate_time = now
                self.seen_boxes[box] = now
                logging.info("Plate detected: %s", plate)
                return plate
        return None

    def _run_inference(self, frame):
        resized = cv2.resize(frame, (self.image_size, self.image_size))
        if self.backend == "onnxruntime":
            input_tensor = resized.astype(np.float32) / 255.0
            input_tensor = np.transpose(input_tensor, (2, 0, 1))
            input_tensor = np.expand_dims(input_tensor, axis=0)
            return self.session.run(self.output_names, {self.input_name: input_tensor})

        blob = cv2.dnn.blobFromImage(resized, 1.0 / 255.0, (self.image_size, self.image_size), swapRB=False, crop=False)
        self.net.setInput(blob)
        outputs = self.net.forward(self.output_names)
        return normalize_dnn_outputs(outputs)

    def _postprocess(self, outputs):
        detections = []
        output = normalize_dnn_outputs(outputs)[0]
        if output.ndim == 2:
            output = np.expand_dims(output, axis=0)

        for detection in output:
            boxes = detection[:4, :]
            scores = detection[4:, :]
            if scores.size == 0:
                continue
            class_ids = np.argmax(scores, axis=0)
            confs = scores[class_ids, np.arange(scores.shape[1])]
            indices = np.where(confs >= self.confidence)[0]
            suppressed = np.zeros(len(indices))

            for i, idx in enumerate(indices):
                if suppressed[i]:
                    continue
                box = boxes[:, idx]
                class_id = class_ids[idx]
                for j, idx2 in enumerate(indices):
                    if idx2 < idx or class_ids[idx2] != class_id:
                        continue
                    if box_iou(box, boxes[:, idx2]) >= self.iou:
                        suppressed[j] = True
                detections.append({"bbox": box, "confidence": confs[idx], "class_id": class_id})
                suppressed[i] = True
        return detections

    def _extract_plate_box(self, frame, detection, x_scale, y_scale):
        x, y, w, h = detection["bbox"]
        x1 = max(0, int((x - w / 2) * x_scale))
        y1 = max(0, int((y - h / 2) * y_scale))
        x2 = min(frame.shape[1], int((x + w / 2) * x_scale))
        y2 = min(frame.shape[0], int((y + h / 2) * y_scale))
        if x2 - x1 < 60 or y2 - y1 < 20:
            return None, None
        return (x1, y1, x2, y2), frame[y1:y2, x1:x2]

    def _preprocess_plate(self, plate_crop):
        gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
        gray = self.clahe.apply(gray)
        blur = cv2.bilateralFilter(gray, 11, 16, 16)
        thresh = cv2.adaptiveThreshold(
            blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 13, 2
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
        return cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    def _extract_valid_plate(self, plate_crop):
        raw_text = self.pytesseract.image_to_string(
            self._preprocess_plate(plate_crop), config=self.tesseract_config
        )
        raw_text = raw_text.strip().replace("\n", " ").replace("\f", "")
        raw_text = "".join(c for c in raw_text if c.isalnum() or c.isspace())
        if self.plate_formatter.is_valid(raw_text):
            return self.plate_formatter.normalize(raw_text)
        return None


class Recorder:
    record_filename = "logs/recorded_speed.csv"
    record_headers = "timestamp,speed,speed_deviation,area,area_deviation,frames,seconds,direction,plate"

    def __init__(self, cfg):
        self.min_speed = cfg.min_speed
        self.min_speed_alert = cfg.min_speed_alert
        self.min_area = cfg.min_area
        self.min_confidence = cfg.min_confidence
        self.min_confidence_alert = cfg.min_confidence_alert
        self.telegram_token = cfg.telegram_token
        self.telegram_chat_id = cfg.telegram_chat_id
        self.logs_dir = getattr(cfg, "logs_dir", Path("logs"))
        self.record_filename = self.logs_dir / "recorded_speed.csv"
        self.bot = None

        if self.telegram_token and self.telegram_chat_id and telegram:
            self.bot = telegram.Bot(self.telegram_token)
        elif self.telegram_token and self.telegram_chat_id:
            logging.warning("Telegram disabled, python-telegram-bot is not installed")

        self.logs_dir.mkdir(parents=True, exist_ok=True)
        if not self.record_filename.is_file():
            self.write_csv(self.record_headers)

    def write_csv(self, message):
        with open(self.record_filename, "a", encoding="utf-8") as csv_file:
            csv_file.write(message + "\n")

    def send_message(self, text):
        if self.bot:
            self.bot.send_message(chat_id=self.telegram_chat_id, text=text)

    def send_image(self, filename, text):
        if self.bot:
            self.bot.send_photo(chat_id=self.telegram_chat_id, photo=open(filename, "rb"), caption=text)

    def send_gif(self, filename, text):
        if self.bot:
            self.bot.send_animation(chat_id=self.telegram_chat_id, animation=open(filename, "rb"), caption=text)

    def send_animation(self, timestamp, events, confidence, mps, plate):
        folder = self.logs_dir / "{}-{:.1f}mps-{:.0f}".format(
            timestamp.strftime("%Y-%m-%d_%H:%M:%S.%f"), mps, confidence
        )
        gif_file = folder.with_suffix(".gif")
        json_file = folder.with_suffix(".json")
        folder.mkdir(parents=True, exist_ok=True)

        data = []
        for event in events:
            image = annotate_image(
                event["image"],
                event["ts"],
                mps=event_speed_mps(event),
                confidence=confidence,
                x=event["x"],
                y=event["y"],
                w=event["w"],
                h=event["h"],
                plate=plate,
            )
            cv2.imwrite(str(folder / "{}.jpg".format(event["ts"])), image)
            del event["image"]
            event["ts"] = event["ts"].timestamp()
            event["plate"] = plate
            data.append(event)

        with open(json_file, "w", encoding="utf-8") as outfile:
            json.dump(data, outfile)

        jpg_files = sorted(glob.glob(str(folder / "*.jpg")))
        process = subprocess.Popen(["/usr/bin/convert", "-delay", "10", *jpg_files, str(gif_file)])
        process.wait()
        shutil.rmtree(folder, ignore_errors=True)

        caption = "{:.1f} m/s @ {:.0f}%".format(mps, confidence)
        if plate:
            caption = "{} plate {}".format(caption, plate)
        self.send_gif(filename=str(gif_file), text=caption)

    def record(self, confidence, image, timestamp, mean_speed, avg_area, sd_speed, sd_area, speeds, secs, direction, events, plate):
        if confidence < self.min_confidence or mean_speed < self.min_speed or avg_area < self.min_area:
            return False

        self.write_csv("{},{:.1f},{:.1f},{:.0f},{:.0f},{:d},{:.2f},{:s},{}".format(
            timestamp.timestamp(),
            mean_speed,
            sd_speed,
            avg_area,
            sd_area,
            len(speeds),
            secs,
            str_direction(direction),
            plate or "",
        ))

        if confidence >= self.min_confidence_alert and mean_speed >= self.min_speed_alert:
            process = Process(target=self.send_animation, args=(timestamp, events, confidence, mean_speed, plate))
            process.start()

        return True


class PiCameraFrames:
    def __init__(self, cfg):
        from picamera import PiCamera
        from picamera.array import PiRGBArray

        self.camera = PiCamera(resolution=cfg.resolution, framerate=cfg.fps, sensor_mode=5)
        self.camera.vflip = cfg.camera_vflip
        self.camera.hflip = cfg.camera_hflip
        self.capture = PiRGBArray(self.camera, size=self.camera.resolution)
        time.sleep(2)

    def frames(self):
        for frame in self.camera.capture_continuous(self.capture, format="bgr", use_video_port=True):
            yield frame.array
            self.capture.truncate(0)

    def close(self):
        self.camera.close()


def create_picamera2_video_config(camera, cfg, include_framerate=True):
    kwargs = {
        "main": {"format": "RGB888", "size": (cfg.image_width, cfg.image_height)},
    }
    if include_framerate:
        kwargs["controls"] = {"FrameRate": cfg.fps}
    return camera.create_video_configuration(**kwargs)


def configure_picamera2_video(camera, cfg):
    config = create_picamera2_video_config(camera, cfg, include_framerate=True)
    try:
        camera.configure(config)
        return config
    except RuntimeError as exc:
        message = str(exc).lower()
        control_error = "not advertised by libcamera" in message or "framedurationlimits" in message
        if not control_error:
            raise
        logging.warning("Picamera2 frame-rate control unavailable; retrying without controls: %s", exc)

    config = create_picamera2_video_config(camera, cfg, include_framerate=False)
    camera.configure(config)
    return config


class PiCamera2Frames:
    def __init__(self, cfg):
        from picamera2 import Picamera2

        self.camera = Picamera2()
        configure_picamera2_video(self.camera, cfg)
        self.vflip = cfg.camera_vflip
        self.hflip = cfg.camera_hflip
        self.camera.start()
        time.sleep(2)

    def frames(self):
        while True:
            frame = self.camera.capture_array()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            if self.vflip:
                frame = cv2.flip(frame, 0)
            if self.hflip:
                frame = cv2.flip(frame, 1)
            yield frame

    def close(self):
        self.camera.close()


class OpenCvFrames:
    def __init__(self, cfg):
        self.capture = cv2.VideoCapture(cfg.opencv_camera_index)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.image_width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.image_height)
        self.capture.set(cv2.CAP_PROP_FPS, cfg.fps)
        if not self.capture.isOpened():
            raise RuntimeError("Unable to open OpenCV camera index {}".format(cfg.opencv_camera_index))

    def frames(self):
        while True:
            ok, frame = self.capture.read()
            if ok:
                yield frame
            else:
                time.sleep(0.1)

    def close(self):
        self.capture.release()


def box_iou(box1, box2):
    box1_w, box1_h = box1[2] / 2.0, box1[3] / 2.0
    box2_w, box2_h = box2[2] / 2.0, box2[3] / 2.0
    b1_x1, b1_y1 = box1[0] - box1_w, box1[1] - box1_h
    b1_x2, b1_y2 = box1[0] + box1_w, box1[1] + box1_h
    b2_x1, b2_y1 = box2[0] - box2_w, box2[1] - box2_h
    b2_x2, b2_y2 = box2[0] + box2_w, box2[1] + box2_h
    x1, y1 = max(b1_x1, b2_x1), max(b1_y1, b2_y1)
    x2, y2 = min(b1_x2, b2_x2), min(b1_y2, b2_y2)
    intersect = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    area2 = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    union = area1 + area2 - intersect
    return intersect / union if union > 0 else 0


def normalize_dnn_outputs(outputs):
    if isinstance(outputs, (list, tuple)):
        return list(outputs)
    return [outputs]


def get_speed(pixels, ftperpixel, secs):
    if secs > 0.0:
        return ((pixels * ftperpixel) / secs) * FT_PER_SECOND_TO_MPS
    return 0.0


def event_speed_mps(event):
    if "mps" in event:
        return event["mps"]
    return event.get("mph", 0) * MPH_TO_MPS


def get_pixel_width(fov, distance, image_width):
    frame_width_ft = 2 * (math.tan(math.radians(fov * 0.5)) * distance)
    return frame_width_ft / float(image_width)


def str_direction(direction):
    if direction == LEFT_TO_RIGHT:
        return "LTR"
    if direction == RIGHT_TO_LEFT:
        return "RTL"
    return "???"


def secs_diff(end_time, beg_time):
    return (end_time - beg_time).total_seconds()


def detect_motion(image, min_area):
    image = cv2.dilate(image, None, iterations=2)
    contours = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = contours[0] if len(contours) == 2 else contours[1]

    motion_found = False
    biggest_area = 0
    x = y = w = h = 0

    for contour in cnts:
        x1, y1, w1, h1 = cv2.boundingRect(contour)
        found_area = w1 * h1
        if found_area > min_area and found_area > biggest_area:
            biggest_area = found_area
            motion_found = True
            x, y, w, h = x1, y1, w1, h1

    return motion_found, x, y, w, h, biggest_area


def should_update_tracking(state):
    return state == TRACKING


def annotate_image(image, timestamp, mps=0, confidence=0, h=0, w=0, x=0, y=0, plate=""):
    color_green = (0, 255, 0)
    color_red = (0, 0, 255)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.putText(image, timestamp.strftime("%d %B %Y %H:%M:%S.%f"),
                (10, image.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color_red, 2)

    if mps > 0:
        msg = "{:.1f} m/s".format(mps)
        size, _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 2, 3)
        cntr_x = int((cfg.image_width - size[0]) / 2)
        cv2.putText(image, msg, (cntr_x, int(cfg.image_height * 0.2)), cv2.FONT_HERSHEY_SIMPLEX, 2.0, color_red, 3)

    if confidence > 0:
        msg = "{:.0f}%".format(confidence)
        size, _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 2, 3)
        cntr_x = int((cfg.image_width - size[0]) / 4) * 3
        cv2.putText(image, msg, (cntr_x, int(cfg.image_height * 0.2)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color_red, 3)

    if plate:
        cv2.putText(image, plate, (10, int(cfg.image_height * 0.2)), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color_red, 3)

    cv2.line(image, (cfg.upper_left_x, cfg.upper_left_y), (cfg.upper_left_x, cfg.lower_right_y), color_green, 4)
    cv2.line(image, (cfg.lower_right_x, cfg.upper_left_y), (cfg.lower_right_x, cfg.lower_right_y), color_green, 4)

    if h > 0 and w > 0:
        cv2.rectangle(
            image,
            (cfg.upper_left_x + x, cfg.upper_left_y + y),
            (cfg.upper_left_x + x + w, cfg.upper_left_y + y + h),
            color_green,
            2,
        )
    return image


def show_detection_frame(image, timestamp, mps=0, h=0, w=0, x=0, y=0, plate=""):
    preview_image = annotate_image(image, timestamp, mps=mps, h=h, w=w, x=x, y=y, plate=plate)
    cv2.imshow("Speed Plate Camera", preview_image)
    key = cv2.waitKey(1) & 0xFF
    return key not in (ord("q"), 27)


def setup_logging(base_dir):
    logs_dir = Path(base_dir) / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        handlers=[logging.FileHandler(logs_dir / "service.log"), logging.StreamHandler()],
    )


def parse_command_line():
    parser = argparse.ArgumentParser(description="Integrated Raspberry Pi speed and plate camera")
    parser.add_argument("mode", nargs="?", choices=RUN_MODES, help="run mode")
    parser.add_argument("--config", default="config.yaml", help="YAML config file")
    parser.add_argument("--image", help="image file for platetest mode")
    args = parser.parse_args()
    config_file = Path(args.config)
    if not config_file.is_file():
        raise FileNotFoundError("config file does not exist: {}".format(config_file))
    if args.image and args.mode != "platetest":
        raise ValueError("--image is only supported with platetest mode")
    return args.mode, config_file, Path(args.image) if args.image else None


def setup_frame_source(cfg):
    logging.info("Booting up %s camera", cfg.camera_source)
    if cfg.camera_source == "picamera2":
        return PiCamera2Frames(cfg)
    if cfg.camera_source == "opencv":
        return OpenCvFrames(cfg)
    return PiCameraFrames(cfg)


def run_selftest(config_file):
    setup_logging(Path(config_file).resolve().parent)
    logging.info("Running selftest")
    failures = []
    try:
        cfg = Config.load(config_file)
    except ImportError as exc:
        logging.error("SELFTEST FAIL: required config dependency is missing: %s", exc)
        return 1
    except Exception as exc:
        logging.error("SELFTEST FAIL: could not load config %s: %s", config_file, exc)
        return 1

    if cfg.camera_source not in ("picamera2", "picamera", "opencv"):
        failures.append("camera_source must be picamera2, picamera, or opencv")
    elif not camera_dependency_available(cfg.camera_source):
        failures.append("camera dependency is not importable for camera_source={}".format(cfg.camera_source))

    if cfg.lower_right_x <= cfg.upper_left_x or cfg.lower_right_y <= cfg.upper_left_y:
        failures.append("monitoring area must have positive width and height")

    model_path = Path(cfg.plate_model)
    if cfg.plate_enabled and not model_path.is_file():
        failures.append("plate model not found: {}".format(model_path))

    dependency_checks = [
        ("cv2", "OpenCV"),
        ("numpy", "NumPy"),
    ]
    if cfg.plate_enabled:
        dependency_checks.extend([
            ("pytesseract", "pytesseract"),
        ])
    for module_name, label in dependency_checks:
        try:
            __import__(module_name)
        except ImportError:
            failures.append("{} is not importable".format(label))

    try:
        plate_formatter = get_plate_formatter(cfg.plate_country)
        validator_sample = "ABC1234" if plate_formatter is PlateFormatTW else "CJ12XYZ"
        if not plate_formatter.is_valid(validator_sample):
            failures.append("plate format validator failed for {}".format(validator_sample))
    except ValueError as exc:
        failures.append(str(exc))

    if cfg.plate_enabled and model_path.is_file() and not has_onnx_backend(model_path):
        failures.append("no ONNX backend could load {}".format(model_path))

    if cfg.plate_enabled and not tesseract_available(cfg.tesseract_cmd):
        failures.append("tesseract executable not found; install tesseract-ocr or set tesseract_cmd")

    ft_per_pixel = get_pixel_width(cfg.fov, cfg.l2r_distance, cfg.image_width)
    speed = get_speed(100, ft_per_pixel, 1.0)
    if speed <= 0:
        failures.append("speed calculation returned a non-positive value")

    if failures:
        for failure in failures:
            logging.error("SELFTEST FAIL: %s", failure)
        return 1

    logging.info("SELFTEST OK: config, model path, dependencies, plate format, and speed math passed")
    return 0


def run_capturetest(config_file):
    setup_logging(Path(config_file).resolve().parent)
    logging.info("Running capturetest")
    cfg = Config.load(config_file)
    frame_source = None
    try:
        frame_source = setup_frame_source(cfg)
        for image in frame_source.frames():
            timestamp = datetime.now(timezone.utc)
            preview_image = annotate_image(image, timestamp)
            cv2.imwrite(str(cfg.preview_file), preview_image)
            logging.info("CAPTURETEST OK: wrote %s shape=%s", cfg.preview_file, image.shape)
            return 0
    except Exception as exc:
        logging.error("CAPTURETEST FAIL: %s", describe_camera_error(exc))
        return 1
    finally:
        if frame_source is not None:
            frame_source.close()

    logging.error("CAPTURETEST FAIL: camera produced no frames")
    return 1


def run_platetest(config_file, image_file=None):
    setup_logging(Path(config_file).resolve().parent)
    logging.info("Running platetest")
    cfg = Config.load(config_file)
    recognizer = PlateRecognizer(cfg)
    if not recognizer.enabled:
        logging.error("PLATETEST FAIL: plate recognition is disabled or unavailable")
        return 1

    image = None
    if image_file:
        image = cv2.imread(str(image_file))
        if image is None:
            logging.error("PLATETEST FAIL: could not read image %s", image_file)
            return 1
        logging.info("PLATETEST: loaded image %s shape=%s", image_file, image.shape)
    else:
        frame_source = None
        try:
            frame_source = setup_frame_source(cfg)
            for frame in frame_source.frames():
                image = frame
                logging.info("PLATETEST: captured frame shape=%s", image.shape)
                break
        except Exception as exc:
            logging.error("PLATETEST FAIL: %s", describe_camera_error(exc))
            return 1
        finally:
            if frame_source is not None:
                frame_source.close()

    if image is None:
        logging.error("PLATETEST FAIL: no image available")
        return 1

    recognizer.scan_interval = 0
    recognizer.ocr_interval = 0
    plate = recognizer.process_frame(image) or recognizer.recent_plate()
    preview_image = annotate_image(image, datetime.now(timezone.utc), plate=plate or "")
    cv2.imwrite(str(cfg.preview_file), preview_image)

    if plate:
        logging.info("PLATETEST OK: detected plate=%s wrote %s", plate, cfg.preview_file)
        return 0

    logging.warning("PLATETEST OK: no valid plate detected, wrote %s", cfg.preview_file)
    return 2


def describe_camera_error(exc):
    message = str(exc)
    lower_message = message.lower()
    if "resource busy" in lower_message or "pipeline handler in use" in lower_message:
        return "{}; camera is busy. Stop the service/display process before running this test.".format(message)
    return message


def camera_dependency_available(camera_source):
    if camera_source == "opencv":
        return True

    module_name = "picamera2" if camera_source == "picamera2" else "picamera"
    try:
        __import__(module_name)
        logging.info("SELFTEST: camera dependency available: %s", module_name)
        return True
    except ImportError as exc:
        logging.info("SELFTEST: camera dependency unavailable: %s", exc)
        return False


def tesseract_available(tesseract_cmd):
    if tesseract_cmd:
        if "/" not in str(tesseract_cmd) and "\\" not in str(tesseract_cmd):
            available = shutil.which(str(tesseract_cmd)) is not None
            logging.info("SELFTEST: configured tesseract_cmd %s on PATH available=%s", tesseract_cmd, available)
            return available

        path = Path(tesseract_cmd)
        available = path.is_file()
        logging.info("SELFTEST: configured tesseract_cmd path %s available=%s", path, available)
        return available

    available = shutil.which("tesseract") is not None
    logging.info("SELFTEST: tesseract on PATH available=%s", available)
    return available


def has_onnx_backend(model_path):
    try:
        import onnxruntime as ort
        session = ort.InferenceSession(str(model_path))
        input_meta = session.get_inputs()[0]
        output_names = [out.name for out in session.get_outputs()]
        input_shape = tuple(1 if not isinstance(dim, int) or dim <= 0 else dim for dim in input_meta.shape)
        input_tensor = np.zeros(input_shape, dtype=np.float32)
        outputs = normalize_dnn_outputs(session.run(output_names, {input_meta.name: input_tensor}))
        if not outputs or not hasattr(outputs[0], "shape"):
            raise RuntimeError("onnxruntime returned no ndarray outputs")
        logging.info("SELFTEST: ONNX backend available: onnxruntime, output_shape=%s", outputs[0].shape)
        return True
    except Exception as exc:
        logging.info("SELFTEST: ONNX Runtime unavailable: %s", exc)

    try:
        net = cv2.dnn.readNetFromONNX(str(model_path))
        blob = cv2.dnn.blobFromImage(
            np.zeros((PlateRecognizer.image_size, PlateRecognizer.image_size, 3), dtype=np.uint8),
            1.0 / 255.0,
            (PlateRecognizer.image_size, PlateRecognizer.image_size),
            swapRB=False,
            crop=False,
        )
        net.setInput(blob)
        output_names = net.getUnconnectedOutLayersNames()
        outputs = normalize_dnn_outputs(net.forward(output_names))
        if not outputs or not hasattr(outputs[0], "shape"):
            raise RuntimeError("opencv-dnn returned no ndarray outputs")
        logging.info("SELFTEST: ONNX backend available: opencv-dnn, output_shape=%s", outputs[0].shape)
        return True
    except Exception as exc:
        logging.info("SELFTEST: OpenCV DNN ONNX backend unavailable: %s", exc)
        return False


def main():
    global cfg
    mode, config_file, image_file = parse_command_line()
    if mode == "selftest":
        raise SystemExit(run_selftest(config_file))
    if mode == "capturetest":
        raise SystemExit(run_capturetest(config_file))
    if mode == "platetest":
        raise SystemExit(run_platetest(config_file, image_file))

    setup_logging(config_file.resolve().parent)
    logging.info("Initializing")
    cfg = Config.load(config_file)

    frame_source = setup_frame_source(cfg)
    recorder = Recorder(cfg)
    plate_recognizer = PlateRecognizer(cfg)

    logging.info("Monitoring: (%d,%d) to (%d,%d) = %dx%d space",
                 cfg.upper_left_x, cfg.upper_left_y, cfg.lower_right_x, cfg.lower_right_y,
                 cfg.monitored_width, cfg.monitored_height)

    l2r_ft_per_pixel = get_pixel_width(cfg.fov, cfg.l2r_distance, cfg.image_width)
    r2l_ft_per_pixel = get_pixel_width(cfg.fov, cfg.r2l_distance, cfg.image_width)
    logging.info("L2R: %.0fft from camera == %.2f per pixel", cfg.l2r_distance, l2r_ft_per_pixel)
    logging.info("R2L: %.0fft from camera == %.2f per pixel", cfg.r2l_distance, r2l_ft_per_pixel)

    state = WAITING
    direction = UNKNOWN
    initial_x = initial_w = last_x = last_w = 0
    areas = np.array([])
    speeds = np.array([])
    initial_time = datetime.now(timezone.utc)
    cap_time = datetime.now(timezone.utc)
    fps_time = datetime.now(timezone.utc)
    stats_time = datetime.now(timezone.utc)
    stats_l2r = np.array([])
    stats_r2l = np.array([])
    events = []
    base_image = None
    fps_frames = 0
    has_started = False

    try:
        for image in frame_source.frames():
            timestamp = datetime.now(timezone.utc)
            plate_recognizer.process_frame(image)
            current_plate = plate_recognizer.recent_plate()

            if not has_started:
                preview_image = annotate_image(image, timestamp, plate=current_plate)
                cv2.imwrite(str(cfg.preview_file), preview_image)
                recorder.send_image(filename=str(cfg.preview_file), text="Current View")
                has_started = True
                if mode == "preview":
                    return

            fps_frames += 1
            if fps_frames > 1000:
                elapsed = secs_diff(timestamp, fps_time)
                logging.info("Current FPS @ %.0f", fps_frames / elapsed)
                fps_time = timestamp
                fps_frames = 0

            if secs_diff(timestamp, stats_time) > cfg.telegram_frequency * 60 * 60:
                send_periodic_stats(recorder, stats_l2r, stats_r2l)
                stats_l2r = np.array([])
                stats_r2l = np.array([])
                stats_time = timestamp

            gray = image[cfg.upper_left_y:cfg.lower_right_y, cfg.upper_left_x:cfg.lower_right_x]
            gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, BLURSIZE, 0)

            if base_image is None:
                base_image = gray.copy().astype("float")
                if mode == "display" and not show_detection_frame(image, timestamp, plate=current_plate):
                    return
                continue

            frame_delta = cv2.absdiff(gray, cv2.convertScaleAbs(base_image))
            thresh = cv2.threshold(frame_delta, THRESHOLD, 255, cv2.THRESH_BINARY)[1]
            motion_found, x, y, w, h, biggest_area = detect_motion(thresh, cfg.image_min_area)
            display_mps = 0

            if motion_found:
                if state == WAITING:
                    state = TRACKING
                    direction = UNKNOWN
                    initial_x = x
                    initial_w = w
                    last_x = x
                    last_w = w
                    initial_time = timestamp
                    areas = np.array([])
                    speeds = np.array([])
                    events = []
                    car_gap = secs_diff(initial_time, cap_time)
                    logging.info("Tracking")
                    logging.info("Initial Data: x=%.0f w=%.0f area=%.0f gap=%s", initial_x, initial_w, biggest_area, car_gap)
                    logging.info(" x-?     Secs      m/s  x-pos width area dir plate")
                    if car_gap < cfg.min_distance:
                        state = WAITING
                        base_image = None
                        logging.info("Car too close, skipping")
                        continue
                else:
                    secs = secs_diff(timestamp, initial_time)
                    if secs >= 5:
                        state = WAITING
                        direction = UNKNOWN
                        base_image = None
                        logging.info("Resetting")
                        continue

                    if not should_update_tracking(state):
                        continue

                    abs_chg = 0
                    mps = 0
                    distance = 0
                    if x >= last_x:
                        direction = LEFT_TO_RIGHT
                        distance = cfg.l2r_distance
                        abs_chg = (x + w) - (initial_x + initial_w)
                        mps = get_speed(abs_chg, l2r_ft_per_pixel, secs)
                    else:
                        direction = RIGHT_TO_LEFT
                        distance = cfg.r2l_distance
                        abs_chg = initial_x - x
                        mps = get_speed(abs_chg, r2l_ft_per_pixel, secs)

                    speeds = np.append(speeds, mps)
                    areas = np.append(areas, biggest_area)
                    events.append({
                        "image": image.copy(),
                        "ts": timestamp,
                        "x": x,
                        "y": y,
                        "w": w,
                        "h": h,
                        "mps": mps,
                        "fov": cfg.fov,
                        "image_width": cfg.image_width,
                        "distance": distance,
                        "secs": secs,
                        "delta": abs_chg,
                        "area": biggest_area,
                        "dir": str_direction(direction),
                        "plate": current_plate,
                    })

                    if mps <= 0:
                        logging.info("negative speed - stopping tracking")
                        if direction == LEFT_TO_RIGHT:
                            direction = RIGHT_TO_LEFT
                            x = 1
                        else:
                            direction = LEFT_TO_RIGHT
                            x = cfg.monitored_width + MIN_SAVE_BUFFER

                    logging.info("%4d  %7.2f  %7.1f   %4d  %4d %4d %s %s",
                                 abs_chg, secs, mps, x, w, biggest_area, str_direction(direction), current_plate)
                    display_mps = mps

                    if ((x <= MIN_SAVE_BUFFER) and direction == RIGHT_TO_LEFT) or (
                        (x + w >= cfg.monitored_width - MIN_SAVE_BUFFER) and direction == LEFT_TO_RIGHT
                    ):
                        mean_speed, avg_area, sd_speed, sd_area, confidence = summarize_event(speeds, areas)
                        logging.info("Determined area:   avg=%4.0f deviation=%4.0f frames=%d", avg_area, sd_area, len(areas))
                        logging.info("Determined speed: mean=%4.1f deviation=%4.1f frames=%d", mean_speed, sd_speed, len(speeds))
                        logging.info("Overall Confidence Level %.0f%%", confidence)

                        final_plate = newest_event_plate(events) or plate_recognizer.recent_plate()
                        recorded = recorder.record(
                            image=image,
                            timestamp=timestamp,
                            confidence=confidence,
                            mean_speed=mean_speed,
                            avg_area=avg_area,
                            sd_speed=sd_speed,
                            sd_area=sd_area,
                            speeds=speeds,
                            secs=secs,
                            direction=direction,
                            events=events,
                            plate=final_plate,
                        )
                        if recorded:
                            logging.info("Event recorded%s", " plate={}".format(final_plate) if final_plate else "")
                            if direction == LEFT_TO_RIGHT:
                                stats_l2r = np.append(stats_l2r, mean_speed)
                            elif direction == RIGHT_TO_LEFT:
                                stats_r2l = np.append(stats_r2l, mean_speed)
                        else:
                            logging.info("Event not recorded: Speed, Area, or Confidence too low")

                        state = SAVING
                        cap_time = timestamp
                    last_x = x
                    last_w = w
            else:
                if state != WAITING:
                    state = WAITING
                    direction = UNKNOWN
                    logging.info("Resetting")

            if state == WAITING:
                last_x = 0
                last_w = 0
                cv2.accumulateWeighted(gray, base_image, 0.25)

            if mode == "display" and not show_detection_frame(
                image, timestamp, mps=display_mps, h=h, w=w, x=x, y=y, plate=current_plate
            ):
                return
    finally:
        frame_source.close()
        cv2.destroyAllWindows()


def summarize_event(speeds, areas):
    if len(speeds) > 3:
        mean_speed = np.mean(speeds[1:-1])
        avg_area = np.average(areas[1:-1])
        sd_speed = np.std(speeds[:-1])
        sd_area = np.std(areas[1:-1])
        confidence = ((mean_speed - sd_speed) / mean_speed) * 100 if mean_speed else 0
    elif len(speeds) > 1:
        mean_speed = speeds[-1]
        avg_area = areas[-1]
        sd_speed = 99
        sd_area = 99999
        confidence = 0
    else:
        mean_speed = avg_area = sd_speed = sd_area = confidence = 0
    return mean_speed, avg_area, sd_speed, sd_area, confidence


def newest_event_plate(events):
    for event in reversed(events):
        if event.get("plate"):
            return event["plate"]
    return ""


def send_periodic_stats(recorder, stats_l2r, stats_r2l):
    total = len(stats_l2r) + len(stats_r2l)
    if total <= 0:
        return
    l2r_perc = len(stats_l2r) / total * 100
    r2l_perc = len(stats_r2l) / total * 100
    l2r_mean = np.mean(stats_l2r) if len(stats_l2r) > 0 else 0
    r2l_mean = np.mean(stats_r2l) if len(stats_r2l) > 0 else 0
    recorder.send_message(
        "{:.0f} cars in the past period\nL2R {:.0f}% at {:.1f} m/s\nR2L {:.0f}% at {:.1f} m/s".format(
            total, l2r_perc, l2r_mean, r2l_perc, r2l_mean
        )
    )


if __name__ == "__main__":
    main()
