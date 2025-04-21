#!/bin/bash

# Install Depth AI without dependencies (assuming cv2 and torch are already installed with cuda support)
sudo -H python3 -m pip install --upgrade pip
sudo -H python3 -m pip install depthai --no-deps

# Hotfix for the 'Insufficient permissions to communicate' issue
echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="03e7", MODE="0666"' | sudo tee /etc/udev/rules.d/80-movidius.rules
sudo udevadm control --reload-rules && sudo udevadm trigger


# (OPTIONAL): Verify installation (do this only when the script runs directly)
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "Verifying installation..."

    # Verification checks:
    python3 -c "import depthai; print(depthai.__version__);"
    python3 -c "import cv2; print(cv2.cuda.getCudaEnabledDeviceCount());"
    python3 -c "import torch; print(torch.cuda.get_device_name(0));"
    python3 -c "import depthai; print(depthai.Device.getAllAvailableDevices());"
        
    echo "Verification complete."
fi
