HOST?=pi@127.0.0.1
TARGET_PATH?=integrated-speed-plate-camera
INSTALL_DIR?=$(CURDIR)
SERVICE_USER?=$(if $(SUDO_USER),$(SUDO_USER),$(shell id -un))
PYTHON?=$(shell test -x "$(INSTALL_DIR)/venv/bin/python" && echo "$(INSTALL_DIR)/venv/bin/python" || echo python3)

install:
	apt-get update
	apt-get install -y python3-opencv python3-numpy python3-yaml python3-pip python3-venv imagemagick tesseract-ocr
	-apt-get install -y python3-picamera2
	-apt-get install -y python3-picamera
	python3 -m venv --system-site-packages "$(INSTALL_DIR)/venv"
	"$(INSTALL_DIR)/venv/bin/pip" install --upgrade pip
	"$(INSTALL_DIR)/venv/bin/pip" install -r "$(INSTALL_DIR)/requirements.txt"
	test -f "$(INSTALL_DIR)/config.yaml" || cp "$(INSTALL_DIR)/config.yaml.example" "$(INSTALL_DIR)/config.yaml"
	mkdir -p "$(INSTALL_DIR)/logs"
	chown -R $(SERVICE_USER) "$(INSTALL_DIR)/venv" "$(INSTALL_DIR)/logs" "$(INSTALL_DIR)/config.yaml"
	sed -e 's|@SERVICE_USER@|$(SERVICE_USER)|g' -e 's|@INSTALL_DIR@|$(INSTALL_DIR)|g' "$(INSTALL_DIR)/speed-plate-camera.service" > /etc/systemd/system/speed-plate-camera.service
	systemctl daemon-reload
	systemctl enable speed-plate-camera.service

restart:
	systemctl restart speed-plate-camera.service

stop:
	systemctl stop speed-plate-camera.service

preview:
	$(PYTHON) "$(INSTALL_DIR)/speed_plate_camera.py" preview --config "$(INSTALL_DIR)/config.yaml"

display:
	$(PYTHON) "$(INSTALL_DIR)/speed_plate_camera.py" display --config "$(INSTALL_DIR)/config.yaml"

selftest:
	$(PYTHON) "$(INSTALL_DIR)/speed_plate_camera.py" selftest --config "$(INSTALL_DIR)/config.yaml"

capturetest:
	$(PYTHON) "$(INSTALL_DIR)/speed_plate_camera.py" capturetest --config "$(INSTALL_DIR)/config.yaml"

test:
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -m unittest discover -s "$(INSTALL_DIR)/tests"

sync:
	rsync --exclude '.*' --exclude 'logs' --exclude 'venv' -azr --progress . ${HOST}:${TARGET_PATH}

sync-logs:
	rsync -azr --progress ${HOST}:${TARGET_PATH}/logs/* ./logs/

clean:
	rm -rf "$(INSTALL_DIR)/logs"
	rm -rf "$(INSTALL_DIR)/venv"
	rm -f "$(INSTALL_DIR)/preview.jpg"

tail:
	journalctl -f -u speed-plate-camera.service

connect:
	ssh -t ${HOST} "cd ${TARGET_PATH}; bash"
