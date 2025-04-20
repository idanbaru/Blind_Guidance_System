#!/bin/bash

# Exit the script if any command fails
set -e 

# Define colors for printing messages to terminal
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color (reset)

# Catch exit and print error message
trap 'echo -e "${RED}Something went wrong! Exiting.${NC}"' ERR

# Save the working directory of the installation script
# (so it can call the mini-installation scripts relatively to its location)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Move to HOME directory to download and install the stuff there (cv2, torch, vscode)
cd ~

# Start timer (in seconds, using built-in bash variable "SECONDS")
SECONDS=0  

###########################################################################
# Update package lists
sudo apt update

# Upgrade installed packages
sudo apt upgrade -y

# Install required dependencies
sudo apt install -y \
    build-essential \
    checkinstall \
    cmake \
    git \
    libmbedtls-dev \
    libasound2-dev \
    libavcodec-dev \
    libavdevice-dev \
    libavfilter-dev \
    libavformat-dev \
    libavutil-dev \
    libcurl4-openssl-dev \
    libfdk-aac-dev \
    libfontconfig-dev \
    libfreetype6-dev \
    libgl1-mesa-dev \
    libjack-jackd2-dev \
    libjansson-dev \
    libluajit-5.1-dev \
    libpulse-dev \
    libqt5x11extras5-dev \
    libspeexdsp-dev \
    libswresample-dev \
    libswscale-dev \
    libudev-dev \
    libv4l-dev \
    libvlc-dev \
    libx11-dev \
    libx264-dev \
    libxcb-shm0-dev \
    libxcb-xinerama0-dev \
    libxcomposite-dev \
    libxinerama-dev \
    pkg-config \
    python3-dev \
    qtbase5-dev \
    libqt5svg5-dev \
    swig \
    libxcb-randr0-dev \
    libxcb-xfixes0-dev \
    libx11-xcb-dev \
    libxcb1-dev

# Install OpenCV
chmod +x "$SCRIPT_DIR/install_open_cv/install_open_cv.sh"
"$SCRIPT_DIR/install_open_cv/install_open_cv.sh"

# Install PyTorch
chmod +x "$SCRIPT_DIR/install_pytorch/install_pytorch.sh"
"$SCRIPT_DIR/install_pytorch/install_pytorch.sh"

# Install DepthAI
chmod +x "$SCRIPT_DIR/install_depthai/install_depthai"
"$SCRIPT_DIR/install_depthai/install_depthai.sh"

# Install mpg123 (to play the synthesized audio)
sudo apt install mpg123 -y

# Install pyttsx3
sudo apt update
sudo apt install python3-pip -y
sudo -H pip3 install pyttsx3

# Install gTTS
sudo -H pip3 install gTTS

# Install requests
sudo -H pip3 install requests

# Install threading
sudo -H pip3 install thread6

# Install VSCode (with Python extensions)
git clone https://github.com/JetsonHacksNano/installVSCode
cd installVSCode
./installVSCodeWithPython.sh

###########################################################################
# Retrieve seconds timer [in sec] and print total time of installation
duration=$SECONDS
printf "${GREEN}Installation completed in %02d:%02d:%02d (hh:mm:ss).${NC}\n" \
    $(($duration / 3600)) $(($duration % 3600 / 60)) $(($duration % 60))
