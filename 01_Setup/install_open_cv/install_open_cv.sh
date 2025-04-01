#!/bin/bash

# Install system-level dependencies
sudo apt-get update
sudo apt-get install -y git cmake
sudo apt-get install -y libatlas-base-dev gfortran
sudo apt-get install -y libhdf5-serial-dev hdf5-tools
sudo apt-get install -y python3-dev
sudo apt-get install -y nano locate
sudo apt-get install -y libfreetype6-dev python3-setuptools
sudo apt-get install -y protobuf-compiler libprotobuf-dev openssl
sudo apt-get install -y libssl-dev libcurl4-openssl-dev
sudo apt-get install -y cython3
sudo apt-get install -y libxml2-dev libxslt1-dev

# Update cmake
wget http://www.cmake.org/files/v3.22/cmake-3.22.0.tar.gz
tar xpvf cmake-3.22.0.tar.gz
cd cmake-3.22.0/
./bootstrap --system-curl
make -j4
cd ..

# Update bash profile
echo 'export PATH=/home/nvidia/cmake-3.22.0/bin/:$PATH' >> ~/.bashrc
source ~/.bashrc

# Install supporting libraries for OpenCV
sudo apt-get install -y build-essential pkg-config
sudo apt-get install -y libtbb2 libtbb-dev
sudo apt-get install -y libavcodec-dev libswscale-dev
sudo apt-get install -y libxvidcore-dev libavresample-dev
sudo apt-get install -y libtiff-dev libjpeg-dev libpng-dev
sudo apt-get install -y python-tk libgtk-3-dev
sudo apt-get install -y libcanberra-gtk-module libcanberra-gtk3-module
sudo apt-get install -y libv4l-dev libdc1394-22-dev
sudo apt-get install -y libavformat-dev libavutil-dev

# Install OpenCV (4.1.2) from source with CUDA support
wget -O opencv.zip https://github.com/opencv/opencv/archive/4.1.2.zip
wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/4.1.2.zip
unzip opencv.zip
unzip opencv_contrib.zip
mv opencv-4.1.2 opencv
mv opencv_contrib-4.1.2 opencv_contrib

cd opencv
mkdir build
cd build

cmake -D CMAKE_BUILD_TYPE=RELEASE \
      -D WITH_CUDA=ON \
      -D CUDA_ARCH_PTX="" \
      -D CUDA_ARCH_BIN="5.3,6.2,7.2" \
      -D WITH_CUBLAS=ON \
      -D WITH_LIBV4L=ON \
      -D BUILD_opencv_python3=ON \
      -D BUILD_opencv_python2=OFF \
      -D BUILD_opencv_java=OFF \
      -D WITH_GSTREAMER=ON \
      -D WITH_GTK=ON \
      -D BUILD_TESTS=OFF \
      -D BUILD_PERF_TESTS=OFF \
      -D BUILD_EXAMPLES=OFF \
      -D OPENCV_ENABLE_NONFREE=ON \
      -D OPENCV_EXTRA_MODULES_PATH=/home/$(whoami)/opencv_contrib/modules ..

make -j4
sudo make install

# Verify OpenCV and CUDA installation
python3 -c "import cv2; print(cv2.cuda.getCudaEnabledDeviceCount())"
