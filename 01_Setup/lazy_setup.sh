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
# Install OpenCV if not available
if ! python3 -c "import cv2" &> /dev/null; then
    echo "OpenCV not found. Installing..."
    chmod +x "$SCRIPT_DIR/install_open_cv/install_open_cv.sh"
    "$SCRIPT_DIR/install_open_cv/install_open_cv.sh"
else
    echo "OpenCV is already installed. Skipping installation."
fi

# Install PyTorch if not available
if ! python3 -c "import torch" &> /dev/null; then
    echo "PyTorch not found. Installing..."
    chmod +x "$SCRIPT_DIR/install_pytorch/install_pytorch.sh"
    "$SCRIPT_DIR/install_pytorch/install_pytorch.sh"
else
    echo "PyTorch is already installed. Skipping installation."
fi

# Install DepthAI if not available
if ! python3 -c "import depthai" &> /dev/null; then
    echo "DepthAI not found. Installing..."
    chmod +x "$SCRIPT_DIR/install_depthai/install_depthai.sh"
    "$SCRIPT_DIR/install_depthai/install_depthai.sh"
else
    echo "DepthAI is already installed. Skipping installation."
fi

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

###########################################################################

# Retrieve seconds timer [in sec] and print total time of installation
duration=$SECONDS
printf "${GREEN}Installation completed in %02d:%02d:%02d (hh:mm:ss).${NC}\n" \
    $(($duration / 3600)) $(($duration % 3600 / 60)) $(($duration % 60))
