#!/bin/bash

# Jetson Orin Nano Setup Script
# This script configures an Nvidia Jetson Orin Nano SoM
# Updated with CUDA-enabled OpenCV support

set -e  # Exit on error
set -o pipefail

distro="ubuntu2204"
arch="arm64"

# Function to print section headers
print_section() {
    echo "============================================================"
    echo "  $1"
    echo "============================================================"
    echo ""
}

# Function to check if command succeeded
check_status() {
    if [ $? -eq 0 ]; then
        echo "[SUCCESS] $1"
    else
        echo "[ERROR] $1"
        exit 1
    fi
}

# Function to run commands with error checking
run_cmd() {
    echo "[RUNNING] $1"
    eval "$1"
    check_status "$1"
    echo ""
}

print_section "System Update and Basic Setup"

print_section "NVIDIA CUDA and OpenCV Setup"

# Install CUDA tools and libraries
#run_cmd "cd; wget https://developer.download.nvidia.com/compute/cuda/repos/$distro/$arch/cuda-keyring_1.1-1_all.deb; dpkg -i cuda-keyring_1.1-1_all.deb"
#run_cmd "wget https://developer.download.nvidia.com/compute/cuda/repos/$distro/$arch/cuda-archive-keyring.gpg"
#run_cmd "mv cuda-archive-keyring.gpg /usr/share/keyrings/cuda-archive-keyring.gpg"
#run_cmd "echo 'deb [signed-by=/usr/share/keyrings/cuda-archive-keyring.gpg] https://developer.download.nvidia.com/compute/cuda/repos/$distro/$arch/ /' | tee /etc/apt/sources.list.d/cuda-$distro-$arch.list"
#run_cmd "apt-get update && apt-get install -y cuda-compat-12-8 cuda-toolkit nvidia-gds"

# Add pause to allow user to break and reboot if needed
#echo "[INFO] CUDA toolkit installed. Press Ctrl+C within 10 seconds to abort and reboot if needed."
#sleep 10
#echo "[INFO] Continuing with setup..."

#run_cmd "ln -fs /usr/local/cuda-12.8 /usr/local/cuda"
#run_cmd "echo \"export PATH='/usr/local/cuda:$PATH'\" >> ${HOME}/.bashrc"
#run_cmd "source ${HOME}/.bashrc"

# Install dependencies required for OpenCV
#run_cmd "apt-get install -y libavcodec-dev libavformat-dev libswscale-dev libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev libgtk-3-dev"
#run_cmd "apt-get install -y libpng-dev libjpeg-dev libtiff-dev libwebp-dev libopenblas-dev libatlas-base-dev liblapack-dev gfortran"
#run_cmd "apt-get install -y libhdf5-serial-dev libtbb-dev libprotobuf-dev protobuf-compiler libgoogle-glog-dev libgflags-dev"

# Directory to build OpenCV
OPENCV_DIR="/opt/opencv_build"
run_cmd "mkdir -p $OPENCV_DIR"
run_cmd "cd $OPENCV_DIR"

# Clone OpenCV and OpenCV contrib repos
if [ ! -d "$OPENCV_DIR/opencv" ]; then
    run_cmd "cd $OPENCV_DIR && git clone https://github.com/opencv/opencv.git"
fi
if [ ! -d "$OPENCV_DIR/opencv_contrib" ]; then
    run_cmd "cd $OPENCV_DIR && git clone https://github.com/opencv/opencv_contrib.git"
fi

# Checkout a stable version (4.5.5 for compatibility with our project)
run_cmd "cd $OPENCV_DIR/opencv && git checkout 4.5.5"
run_cmd "cd $OPENCV_DIR/opencv_contrib && git checkout 4.5.5"

# Build OpenCV with CUDA
run_cmd "mkdir -p $OPENCV_DIR/opencv/build"
run_cmd "cd $OPENCV_DIR/opencv/build"

# Get CUDA compute capability for Jetson Orin Nano
echo "[INFO] Detecting CUDA compute capability for Jetson Orin Nano..."
COMPUTE_CAPABILITY="8.7" # Orin Nano typically uses SM 8.7

# Configure and build OpenCV
CMAKE_FLAGS="-D CMAKE_BUILD_TYPE=RELEASE \
      -D BUILD_SHARED_LIBS=ON \
      -D CMAKE_INCLUDE_PATH='/usr/include/aarch64-linux-gnu' \
      -D CMAKE_LIBRARY_PATH='/usr/lib/aarch64-linux-gnu' \
      -D CMAKE_INSTALL_PREFIX=$(python3 -c "import sys; print(sys.prefix)") \
      -D PYTHON3_EXECUTABLE=$(which python3) \
      -D BUILD_opencv_python3=ON \
      -D BUILD_opencv_python2=OFF \
      -D INSTALL_PYTHON_EXAMPLES=OFF \
      -D INSTALL_C_EXAMPLES=OFF \
      -D BUILD_EXAMPLES=OFF \
      -D BUILD_TESTS=OFF \
      -D BUILD_PERF_TESTS=OFF \
      -D OPENCV_EXTRA_MODULES_PATH=~/opencv_build/opencv_contrib/modules \
      -D OPENCV_ENABLE_NONFREE=ON \
      -D OPENCV_GENERATE_PKGCONFIG=ON \
      -D ENABLE_FAST_MATH=ON \
      -D WITH_CUDA=ON \
      -D WITH_CUDNN=OFF \
      -D OPENCV_DNN_CUDA=OFF \
      -D CUDA_ARCH_BIN=8.7 \
      -D CUDA_ARCH_PTX=8.7 \
      -D CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda \
      -D CUDNN_LIBRARY=/usr/lib/aarch64-linux-gnu/libcudnn.so \
      -D CUDNN_INCLUDE_DIR=/usr/include/aarch64-linux-gnu \
      -D WITH_CUBLAS=ON \
      -D WITH_EIGEN=ON \
      -D WITH_LAPACK=ON \
      -D WITH_PROTOBUF=ON \
      -D WITH_PTHREADS_PF=ON \
      -D WITH_FFMPEG=ON \
      -D WITH_GSTREAMER=ON \
      -D WITH_V4L=ON \
      -D WITH_LIBV4L=ON \
      -D WITH_QT=ON \
      -D WITH_OPENGL=ON \
      -D WITH_GTK=ON \
      -D WITH_CAROTENE=ON \
      -D WITH_TEGRA_OPTIMIZATION=ON \
      -D BUILD_opencv_alphamat=ON \
      -D BUILD_opencv_matlab=OFF \
      -D BUILD_opencv_viz=ON \
      -D BUILD_opencv_ts=ON \
      -D BUILD_opencv_sfm=ON \
      -D BUILD_opencv_ovis=ON \
      -D LAPACK_INCLUDE_DIR=/usr/include \
      -D LAPACK_LIBRARIES="/usr/lib/aarch64-linux-gnu/libopenblas.so" \
      -D WITH_NVCUVID=OFF \
      -D WITH_NVCUVENC=OFF "

run_cmd "cd $OPENCV_DIR/opencv/build && cmake $CMAKE_FLAGS .."

# Use all available CPU cores for build, but limit to 2 if low on memory
NUM_CORES=$(nproc)
if [ $NUM_CORES -gt 2 ]; then
    BUILD_CORES=$((NUM_CORES - 1))
else
    BUILD_CORES=1
fi

echo "[INFO] Building OpenCV with $BUILD_CORES cores. This may take a while..."
run_cmd "cd $OPENCV_DIR/opencv/build && make -j$BUILD_CORES"
run_cmd "cd $OPENCV_DIR/opencv/build && make install"
run_cmd "cd $OPENCV_DIR/opencv/build && ldconfig"
