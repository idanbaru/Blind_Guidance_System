#!/bin/bash

# Install Google TTS (and mpg123 to play the synthesized audio)
sudo apt install mpg123 -y
sudo apt update
sudo apt install python3-pip -y
sudo -H pip3 install gTTS
