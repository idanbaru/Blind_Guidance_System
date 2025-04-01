#!/bin/bash

# Install Depth AI without dependencies (assuming cv2 and torch are already installed with cuda support)
sudo -H python3 -m pip install --upgrade pip
sudo -H python3 -m pip install depthai --no-deps

# Verify installation
python3 -c "import depthai; print(depthai.__version__)"
python3 -c "import cv2; print(cv2.cuda.getCudaEnabledDeviceCount())"
python3 -c "import torch; print(torch.cuda.get_device_name(0))"
