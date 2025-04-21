# Getting Started
In this folder are all the information (and scripts) you need to install the dependencies of this project.

We recommend first letting the os load and run the auto-updater if pops up.

Afterwards, you can continue to installing the system's dependencies. You can either follow the `Simple Installation` which runs automated installation scripts, or follow the `Manual Installation` which guides you how to install each dependency.

---

## Simple Installation

### Update and Verify Important lLbraries
In your new flashed Jetson Nano device run:

```bash
chmod +x first_setup.sh
sudo ./first_setup.sh
```
**Note: THIS WILL REBOOT THE SYSTEM!**

This verifies that important libraries are installed. 
**Note: if you chose to flash the Ubuntu20.04 image, wait for the system updater to pop up and run it instead of running `first_setup.sh`.**


Next, for simple (lazy) automated installation, simply run:

```bash
chmod +x lazy_setup.sh
sudo ./lazy_setup.sh
```

And let the script install everything (this might take a while). Otherwise, if you're brave, follow the manual steps.

---

## Manual Installation
Oh, you're still here? Great!

We built this folder to be very straight-forward and simple, yet include every step you need to perform and even manual explanations for those who want to study how to do it by themselves.

Each subdirectory here represents an installation needed to be done before running the project. In each directory you will find a `README.md` file explaining how to run the specific installation script, as well as how to manually install it.

If you feel like it's too much - go back to the simple explanation and save yourself the troubles! But if you want to install it manually, follow the instructions below.

### First Step: updating and verifying important libraries
Run `chmod +x first_setup.sh` followed by `sudo ./first_setup.sh` in your new flashed Jetson Nano device to verify important libraries are installed.
**Note: THIS WILL REBOOT THE SYSTEM!**

Alternatively, you can manually run the following commands:
```bash
sudo apt update
sudo apt upgrade -y
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
sudo reboot
```

### Next Steps: installing the system's dependencies
In order for this project to work, you'll now have to install the following dependencies:

* `OpenCV` - Open Source Computer Vision Library.
Used for real-time image processing and computer vision tasks such as object detection, filtering, transformations, and camera interfacing.
* `PyTorch` - Deep Learning Framework.
Provides tools for machine learning model training, inference, and deployment, with GPU acceleration and an intuitive Pythonic API.
* `DepthAI` - A library that enables you to control and run AI models on Luxonis’ OAK cameras, including real-time neural inference, depth sensing, and video streaming.
* `mpg123` - Audio Playback Utility.
A fast command-line audio player used to play `.mp3` files on Linux systems, useful for lightweight, non-GUI audio output. Its installation is described in install_text_to_speech subdirectory.

* `pyttsx3` - Text-to-Speech Conversion Library.
A Python package for offline text-to-speech that uses your system’s installed speech engines (SAPI5, NSSpeechSynthesizer, or espeak). This can run locally on the Jetson Nano.
* `gTTS` - Google Text-to-Speech API Client.
A Python library that uses Google’s online TTS API to convert text to speech and save it as an .mp3 file. This requires internet connection.
* `requests` - A popular third-party Python library for making HTTP requests easily (GET, POST, PUT, etc.).
Used for interacting with web APIs or downloading files from the internet. Its installation script is found under `install_additional_libraries` folder.
* `threading` - A built-in Python module for running code in parallel threads, allowing tasks to execute "simultaneously" (e.g., handling a camera stream and voice output at the same time). Its installation script is found under `install_additional_libraries` folder.

**We highly recommend following this exact installation order, since this is the order that was tested when we ran the system ourselves!**

For each installation, open the relevant directory (usually called `install_<name_of_library>`) where you will find a `README.md` file with all the relevant details, as well as `install_<name_of_library>.sh` installation script for a quick and easy installation. Moreover, you can also find details about potential bugs and how to fix them (which the lazy installation already attempts to solve automatically).

### Extras: Docker, Fan Control, VSCode, Jupter Notebook, etc...
In the `misc` subfolder you will find more useful guides on how to work with Docker, how to install a fan control script that controls a 5V PWM fan connected to the Jetson Nano, how to install useful tools such as VSCoda and Jupyter Notebook, and more! Please Note that the lazy script already installs VSCode.
