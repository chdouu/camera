# Integrated Speed and Plate Camera

This folder combines the existing Raspberry Pi speed camera and license plate recognition projects into one runtime.

The important change is that only one camera loop is used. Every frame comes from the Raspberry Pi camera or USB camera, then:

1. The speed-camera motion tracker estimates direction and MPH.
2. The plate recognizer periodically runs the ONNX license plate model and Tesseract OCR on the same frame.
3. Recorded events are written to `logs/recorded_speed.csv` with a `plate` column.
4. Alert GIF/JSON files include the latest plate seen near the speed event.

## Files

- `speed_plate_camera.py` - integrated runtime.
- `config.yaml.example` - template used to create `config.yaml`; tune `config.yaml` for your camera position.
- `speed-plate-camera.service` - systemd service template for Raspberry Pi.
- `Makefile` - install, service, preview, log helper commands.
- `requirements.txt` - Python packages that are not installed by apt.

## Raspberry Pi Setup

Copy this folder to the Pi, for example:

```bash
/home/pi/integrated-speed-plate-camera
```

The ONNX plate model is included in this integrated folder. The default config points to:

```bash
models/yolov8n-license_plate.onnx
```

Install dependencies:

```bash
sudo make install
```

`Makefile` installs OpenCV and NumPy from apt because that is usually more reliable on Raspberry Pi than building `opencv-python` from pip.
It also creates a `venv` with `--system-site-packages`, so the service can use apt-installed Pi camera/OpenCV packages and pip-installed ONNX/OCR packages together.
The Pi camera packages differ between Raspberry Pi OS releases, so the Makefile tries both `python3-picamera2` and legacy `python3-picamera`.
During `sudo make install`, the systemd service is generated from the current project path and the sudo user. Override with `sudo make install INSTALL_DIR=/path/to/project SERVICE_USER=youruser` if needed.
The install step creates `config.yaml` from `config.yaml.example` if it does not already exist. It also gives `venv/`, `logs/`, and `config.yaml` ownership to `SERVICE_USER`, so the service can run and write logs without root-owned runtime files getting in the way.

Edit `config.yaml` for the monitored area, road distances, and camera orientation. Leave Telegram fields empty if you do not use Telegram.
Relative paths in `config.yaml`, such as `plate_model` and path-like `tesseract_cmd` values, are resolved from the folder containing `config.yaml`. A bare `tesseract_cmd: tesseract` is treated as a command on `PATH`.

Run a no-camera dependency/config check:

```bash
make selftest
```

`selftest` checks the selected camera Python module, the plate model, at least one ONNX inference backend, `pytesseract`, and the `tesseract` executable.

The plate recognizer uses ONNX Runtime when available and falls back to OpenCV DNN. ONNX Runtime is usually faster, but OpenCV DNN avoids a hard dependency on ONNX Runtime wheels for older Raspberry Pi images.

Run the no-camera integration tests:

```bash
make test
```

The Makefile uses `./venv/bin/python` after install and falls back to `python3` before install. Override with `make test PYTHON=/path/to/python` if needed.

Run a preview:

```bash
make preview
```

`preview` writes one annotated frame to `preview.jpg` and exits; it does not open a live detection window.
To view the live detection feed on a desktop session, run:

```bash
make display
```

Press `q` or `Esc` to close the detection window. This requires a graphical desktop; it will not work from a headless SSH/systemd session.

Run a one-frame camera capture test:

```bash
make capturetest
```

Run a one-frame plate recognition test from the configured camera:

```bash
make platetest
```

Run plate recognition against an existing image:

```bash
./venv/bin/python speed_plate_camera.py platetest --config config.yaml --image /path/to/plate.jpg
```

`platetest` writes the annotated result to `preview.jpg`. It exits with code `0` when a valid plate is detected, `2` when the image/camera worked but no valid plate was detected, and `1` for setup or dependency failures.

Start the service:

```bash
sudo make restart
make tail
```

## Camera Source

For current Raspberry Pi OS camera stack:

```yaml
camera_source: picamera2
```

For the legacy Raspberry Pi Camera Module stack:

```yaml
camera_source: picamera
```

For a USB camera or local testing:

```yaml
camera_source: opencv
opencv_camera_index: 0
```

USB UVC cameras can also appear through Picamera2/libcamera. If the camera does not advertise frame-rate controls, this program retries Picamera2 setup without forcing FPS; if that still fails, use `camera_source: opencv` for the USB camera.

## Output

Speed records are written to:

```bash
logs/recorded_speed.csv
```

CSV columns:

```text
timestamp,speed,speed_deviation,area,area_deviation,frames,seconds,direction,plate
```

If no plate is confidently recognized near an event, the `plate` field is empty and the speed event is still recorded.

## Notes

- The default plate validator is Taiwan (`plate_country: TW`). It supports common OCR-normalized forms such as `ABC1234`, `1234AB`, `ABC123`, and `AB1234`. The original Romanian validator is still available with `plate_country: RO`.
- Taiwan support here covers OCR text validation and normalization. The included ONNX model is still the original license-plate detector; if it misses local plate shapes or camera angles, collect Taiwan samples and retrain or replace `models/yolov8n-license_plate.onnx`.
- Tesseract must be installed on the Pi: `sudo apt-get install tesseract-ocr`.
- ONNX inference can be heavy on older Raspberry Pi boards. Install `onnxruntime` in the venv if a compatible wheel exists and you need more speed; otherwise the OpenCV DNN fallback will be used. Increase `plate_scan_interval` or `plate_ocr_interval` if FPS is too low.
