#!/bin/bash

# Install PyTorch dependencies
sudo apt-get install -y python3-pip libjpeg-dev libopenblas-dev libopenmpi-dev libomp-dev
sudo -H pip3 install future
sudo pip3 install -U --user wheel mock pillow
sudo -H pip3 install --upgrade setuptools
sudo -H pip3 install cython

# Download and install PyTorch
sudo -H pip3 install gdown
gdown https://drive.google.com/uc?id=1TqC6_2cwqiYacjoLhLgrZoap6-sVL2sd
sudo -H pip3 install torch-1.10.0a0+git36449ea-cp36-cp36m-linux_aarch64.whl

# Install torchvision dependencies
sudo apt-get install -y libjpeg-dev zlib1g-dev libpython3-dev
sudo apt-get install -y libavcodec-dev libavformat-dev libswscale-dev
sudo pip3 install -U pillow

# Install torchvision
gdown https://drive.google.com/uc?id=1C7y6VSIBkmL2RQnVy8xF9cAnrrpJiJ-K
sudo -H pip3 install torchvision-0.11.0a0+fa347eb-cp36-cp36m-linux_aarch64.whl

# Verify PyTorch and CUDA
python3 -c "import torch; import torchvision; \
           device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'); \
           print(f'Using device: {device}'); \
           print(torch.cuda.is_available()); \
           print(torch.cuda.device_count()); \
           print(torch.cuda.current_device()); \
           print(torch.cuda.get_device_name(0))"

