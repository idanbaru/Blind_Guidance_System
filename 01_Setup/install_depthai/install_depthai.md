## Install DepthAI on Jetson Nano
python3 -m pip install --upgrade pip
python3 -m pip install depthai --no-deps
python3 -m pip install --user jupyter

python3 -c "import depthai; print(depthai.__version__)"
python3 -c "import cv2; print(cv2.cuda.getCudaEnabledDeviceCount())"
python3 -c "import torch; print(torch.cuda.get_device_name(0))"

python3 -c "import depthai; print(depthai.Device.getAllAvailableDevices())"
# If error pops:
[2025-02-27 13:58:36.949] [depthai] [warning] Insufficient permissions to communicate 	with X_LINK_UNBOOTED device having name "1.2.4". Make sure udev rules are set
[]

# Solution:
echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="03e7", MODE="0666"' | sudo tee /etc/udev/rules.d/80-movidius.rules
sudo udevadm control --reload-rules && sudo udevadm trigger

