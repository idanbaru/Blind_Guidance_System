# Install DepthAI on Jetson Nano

## Prerequisites

Before installing the `depthai` library on Jetson Nano, make sure the following dependencies are installed:

- OpenCV  
- PyTorch  

---

## Installation via Script

Give the installation script executable permissions:

```bash
chmod +x install_depthai.sh
```

Run the installation script as superuser:

```bash
sudo ./install_depthai.sh
```

---

## Manual Installation

If you prefer manual installation, run the following commands.  
Make sure to install `depthai` without dependencies to avoid interfering with your local OpenCV installation.

```bash
python3 -m pip install --upgrade pip
python3 -m pip install depthai --no-deps
python3 -m pip install --user jupyter
```

---

## Verify the Installation

Run the following commands to confirm the installation was succesful.

```bash
python3 -c "import depthai; print(depthai.__version__)"
```

Also verify that OpenCV and PyTorch still have CUDA support.

```bash
python3 -c "import cv2; print(cv2.cuda.getCudaEnabledDeviceCount())"
python3 -c "import torch; print(torch.cuda.get_device_name(0))"
```

---

## Check Connected OAK-D Device
Run the following command to list connected OAK-D or compatible devices.

```bash
python3 -c "import depthai; print(depthai.Device.getAllAvailableDevices())"
```

Example output should look something like:

```bash
[<XLinkDeviceDesc: name='1.2.4', mxid='14442C1091A5560F00', state=UNBOOTED, productName='OAK-D'>]
```

---

## Troubleshooting: USB Permission Error
If you encounter the following error:

```bash
[2025-01-01 12:12:12.121] [depthai] [warning] Insufficient permissions to communicate 	with X_LINK_UNBOOTED device having name "1.2.4". Make sure udev rules are set
[]
```

Apply this fix:

```bash
echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="03e7", MODE="0666"' | sudo tee /etc/udev/rules.d/80-movidius.rules
sudo udevadm control --reload-rules && sudo udevadm trigger
```
