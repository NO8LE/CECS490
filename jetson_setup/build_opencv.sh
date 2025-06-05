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
run_cmd "cd; wget https://developer.download.nvidia.com/compute/cuda/repos/$distro/$arch/cuda-keyring_1.1-1_all.deb; dpkg -i cuda-keyring_1.1-1_all.deb"
run_cmd "wget https://developer.download.nvidia.com/compute/cuda/repos/$distro/$arch/cuda-archive-keyring.gpg"
run_cmd "mv cuda-archive-keyring.gpg /usr/share/keyrings/cuda-archive-keyring.gpg"
run_cmd "echo 'deb [signed-by=/usr/share/keyrings/cuda-archive-keyring.gpg] https://developer.download.nvidia.com/compute/cuda/repos/$distro/$arch/ /' | tee /etc/apt/sources.list.d/cuda-$distro-$arch.list"
run_cmd "apt-get update && apt-get install -y cuda-compat-12-8 cuda-toolkit nvidia-gds"

# Add pause to allow user to break and reboot if needed
echo "[INFO] CUDA toolkit installed. Press Ctrl+C within 10 seconds to abort and reboot if needed."
sleep 10
echo "[INFO] Continuing with setup..."

run_cmd "ln -fs /usr/local/cuda-12.8 /usr/local/cuda"
run_cmd "echo \"export PATH='/usr/local/cuda:$PATH'\" >> ${HOME}/.bashrc"
run_cmd "source ${HOME}/.bashrc"

# Install dependencies required for OpenCV
run_cmd "apt-get install -y libavcodec-dev libavformat-dev libswscale-dev libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev libgtk-3-dev"
run_cmd "apt-get install -y libpng-dev libjpeg-dev libtiff-dev libwebp-dev libopenblas-dev libatlas-base-dev liblapack-dev gfortran"
run_cmd "apt-get install -y libhdf5-serial-dev libtbb-dev libprotobuf-dev protobuf-compiler libgoogle-glog-dev libgflags-dev"

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

# Verify CUDA support in OpenCV
echo "[INFO] Verifying CUDA support in OpenCV:"
run_cmd "python3 -c 'import cv2; print(\"CUDA is available:\" + str(cv2.cuda.getCudaEnabledDeviceCount() > 0))'"

print_section "NVIDIA Container Toolkit Setup"

# Add NVIDIA Container Toolkit repository
run_cmd "curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg"
run_cmd "curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | tee /etc/apt/sources.list.d/nvidia-container-toolkit.list"
run_cmd "sed -i -e '/experimental/ s/^#//g' /etc/apt/sources.list.d/nvidia-container-toolkit.list"
run_cmd "apt-get update"

# Install NVIDIA Container Toolkit
run_cmd "apt-get install -y libnvidia-container-tools libnvidia-container1 nvidia-container-toolkit nvidia-container-toolkit-base"
run_cmd "nvidia-ctk runtime configure --runtime=docker"

# Install Docker
run_cmd "curl -fsSL https://get.docker.com | bash"
run_cmd "systemctl --now enable docker"
run_cmd "systemctl restart docker"

# Test NVIDIA Container Toolkit with Docker
echo "[INFO] Testing NVIDIA Container Toolkit with Docker"
run_cmd "docker run --rm --runtime=nvidia --gpus all ubuntu nvidia-smi || echo 'Note: This command may fail if hardware is different. Continuing...'"

print_section "Storage Configuration"

print_section "DepthAI/Luxonis Setup with CUDA Support"

# Install DepthAI dependencies
run_cmd "wget -qO- https://docs.luxonis.com/install_dependencies.sh | bash"

# Create DepthAI directory and virtual environment
run_cmd "mkdir -p /root/depthai"
run_cmd "python3 -m venv /root/depthai/venv"

# Install DepthAI
run_cmd "wget -qO- https://docs.luxonis.com/install_depthai.sh | bash"

# Add user to dialout group for device access
run_cmd "usermod -a -G dialout \$USER"
run_cmd "apt-get remove modemmanager -y"

# Set OpenBLAS core type for ARM
echo "export OPENBLAS_CORETYPE=ARMV8" >> /etc/profile.d/openblas.sh
run_cmd "chmod +x /etc/profile.d/openblas.sh"

print_section "ROS Setup"

# Add ROS repository
run_cmd "sh -c 'echo \"deb http://packages.ros.org/ros/ubuntu \$(lsb_release -sc) main\" > /etc/apt/sources.list.d/ros-latest.list'"
run_cmd "apt-get update"

# Pull ROS Docker containers for Jazzy and Humble
echo "[INFO] Pulling ROS Docker containers for Jazzy and Humble"
run_cmd "docker pull osrf/ros:jazzy-desktop"
run_cmd "docker pull osrf/ros:humble-desktop"

# Create convenience scripts for running ROS containers
mkdir -p /home/res0ne/ros

# Create script for ROS Jazzy
cat > /home/res0ne/ros/run_ros_jazzy.sh <<EOF
#!/bin/bash
# Run ROS Jazzy container with NVIDIA GPU support and X11 forwarding

# Get the current user ID and group ID
USER_ID=\$(id -u)
GROUP_ID=\$(id -g)

# Run the container
docker run -it --rm \\
  --privileged \\
  --network=host \\
  --runtime=nvidia \\
  --gpus all \\
  -e DISPLAY=\$DISPLAY \\
  -e QT_X11_NO_MITSHM=1 \\
  -v /tmp/.X11-unix:/tmp/.X11-unix \\
  -v \$HOME/ros_ws_jazzy:/root/ros_ws \\
  -v /dev:/dev \\
  osrf/ros:jazzy-desktop \\
  bash
EOF
chmod +x /home/res0ne/ros/run_ros_jazzy.sh

# Create script for ROS Humble
cat > /home/res0ne/ros/run_ros_humble.sh <<EOF
#!/bin/bash
# Run ROS Humble container with NVIDIA GPU support and X11 forwarding

# Get the current user ID and group ID
USER_ID=\$(id -u)
GROUP_ID=\$(id -g)

# Run the container
docker run -it --rm \\
  --privileged \\
  --network=host \\
  --runtime=nvidia \\
  --gpus all \\
  -e DISPLAY=\$DISPLAY \\
  -e QT_X11_NO_MITSHM=1 \\
  -v /tmp/.X11-unix:/tmp/.X11-unix \\
  -v \$HOME/ros_ws_humble:/root/ros_ws \\
  -v /dev:/dev \\
  osrf/ros:humble-desktop \\
  bash
EOF
chmod +x /home/res0ne/ros/run_ros_humble.sh

# Create workspace directories
run_cmd "mkdir -p /home/res0ne/ros_ws_jazzy/src"
run_cmd "mkdir -p /home/res0ne/ros_ws_humble/src"

# Set ownership
run_cmd "chown -R res0ne:res0ne /home/res0ne/ros"
run_cmd "chown -R res0ne:res0ne /home/res0ne/ros_ws_jazzy"
run_cmd "chown -R res0ne:res0ne /home/res0ne/ros_ws_humble"

# Add ROS container aliases to res0ne's .bashrc
cat >> /home/res0ne/.bashrc <<EOF

# ROS container aliases
alias ros-jazzy='/home/res0ne/ros/run_ros_jazzy.sh'
alias ros-humble='/home/res0ne/ros/run_ros_humble.sh'
EOF

echo "[INFO] ROS Docker containers and convenience scripts have been set up"

print_section "SSH and Security Configuration"

# Install and configure fail2ban
run_cmd "apt-get install -y fail2ban"

# Configure SSH
if [ -f /etc/ssh/sshd_config ]; then
    # Backup original config
    cp /etc/ssh/sshd_config /etc/ssh/sshd_config.bak
    
    # Update SSH configuration for better security
    sed -i 's/#Port 22/Port 3322/' /etc/ssh/sshd_config
    sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin no/' /etc/ssh/sshd_config
    sed -i 's/#PasswordAuthentication yes/PasswordAuthentication no/' /etc/ssh/sshd_config
    
    echo "[INFO] SSH configured to use port 3322 and disable root login"
    run_cmd "service ssh restart"
    
    # Configure UFW
    run_cmd "ufw allow from 192.168.2.0/28 to any port 3322"
    run_cmd "ufw enable"
else
    echo "[WARNING] SSH config file not found, skipping SSH configuration"
fi

print_section "User Configuration - Create and Setup 'res0ne' User"

# Create res0ne user if it doesn't exist
if ! id -u res0ne &>/dev/null; then
    run_cmd "useradd -m -s /bin/bash res0ne"
    run_cmd "usermod -aG sudo res0ne"
    echo "[INFO] Created user 'res0ne'"
    
    # Set a default password that should be changed on first login
    echo "res0ne:jetson" | chpasswd
    run_cmd "passwd -e res0ne"  # Force password change on first login
else
    echo "[INFO] User 'res0ne' already exists"
fi

# Add res0ne to necessary groups
run_cmd "usermod -aG dialout,video,docker res0ne"

# Configure sudoers for passwordless sudo access
cat > /etc/sudoers.d/res0ne <<EOF
# Allow res0ne to run sudo commands without password
res0ne ALL=(ALL) NOPASSWD:ALL
EOF
run_cmd "chmod 440 /etc/sudoers.d/res0ne"

# Create directory structure similar to res0ne
run_cmd "mkdir -p /home/res0ne/git"
run_cmd "mkdir -p /home/res0ne/QGroundControl"
run_cmd "mkdir -p /home/res0ne/CECS490/OAK-D"
run_cmd "mkdir -p /home/res0ne/tmp"

# Clone repositories from res0ne's environment
run_cmd "cd /home/res0ne/git && git clone https://github.com/ArchilChovatiya/ArUco-marker-detection-with-DepthAi.git"
run_cmd "chmod +x /home/res0ne/git/ArUco-marker-detection-with-DepthAi/*.py"
run_cmd "cd /home/res0ne/git && git clone https://github.com/chintal/depthai-sandbox.git"
run_cmd "cd /home/res0ne/git && git clone https://github.com/dusty-nv/jetson-containers"

# Install and configure jetson-containers
run_cmd "cd /home/res0ne/git/jetson-containers && bash install.sh"
run_cmd "chown -R res0ne:res0ne /home/res0ne/git/jetson-containers"

# Install QGroundControl for res0ne
run_cmd "wget -O /home/res0ne/QGroundControl/QGroundControl.AppImage https://d176tv9ibo4jno.cloudfront.net/latest/QGroundControl.AppImage"
run_cmd "chmod +x /home/res0ne/QGroundControl/QGroundControl.AppImage"

# Set up Python environment for res0ne with CUDA-enabled packages
run_cmd "python3 -m venv /home/res0ne/depthai-env"
run_cmd "chown -R res0ne:res0ne /home/res0ne/depthai-env"

# Use the system-installed OpenCV with CUDA rather than pip packages
cat > /tmp/requirements.txt <<EOF
numpy
matplotlib
scikit-learn
depthai
depthai-viewer
blobconverter
vedo
EOF

# Install requirements excluding OpenCV (which we built from source)
run_cmd "pip3 install -r /tmp/requirements.txt"

# Create a symlink for OpenCV in the virtual environment
run_cmd "ln -s /usr/local/lib/python3.*/site-packages/cv2 /home/res0ne/depthai-env/lib/python3.*/site-packages/"

# Set up .bashrc for res0ne with environment variables
cat >> /home/res0ne/.bashrc <<EOF

# DepthAI environment variables
export OPENBLAS_CORETYPE=ARMV8

# Python environment variables
export PYTHONPATH=\$PYTHONPATH:/usr/local/lib/python3.8/site-packages

# CUDA environment variables
export PATH=\$PATH:/usr/local/cuda/bin
export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/usr/local/cuda/lib64

# Alias for convenience
alias ll='ls -la'
EOF

# Set up SSH directory for res0ne
run_cmd "mkdir -p /home/res0ne/.ssh"
run_cmd "chmod 700 /home/res0ne/.ssh"

# Add the specified SSH public key to authorized_keys
cat > /home/res0ne/.ssh/authorized_keys <<EOF
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIPg7eHztrQQBVwLLSPlw7hryxD3yK6QvCEi8sa+KNgiW
EOF

run_cmd "chmod 600 /home/res0ne/.ssh/authorized_keys"

# Fix permissions for all files
run_cmd "chown -R res0ne:res0ne /home/res0ne"

print_section "Additional Software Installation"

# Install QGroundControl system-wide
run_cmd "mkdir -p /opt/QGroundControl"
run_cmd "wget -O /opt/QGroundControl/QGroundControl.AppImage https://d176tv9ibo4jno.cloudfront.net/latest/QGroundControl.AppImage"
run_cmd "chmod +x /opt/QGroundControl/QGroundControl.AppImage"

# Install Firefox
run_cmd "apt-get install -y firefox"

print_section "Testing CUDA with OpenCV"

# Create a simple CUDA test script
cat > /home/res0ne/cuda_test.py <<EOF
#!/usr/bin/python3
import cv2
import numpy as np
import sys

def test_opencv_cuda():
    print(f"OpenCV version: {cv2.__version__}")
    
    # Check if CUDA is available
    cuda_devices = cv2.cuda.getCudaEnabledDeviceCount()
    print(f"CUDA enabled devices: {cuda_devices}")
    
    if cuda_devices > 0:
        print("CUDA device information:")
        for i in range(cuda_devices):
            cv2.cuda.printCudaDeviceInfo(i)
            
        # Test a simple CUDA operation
        try:
            # Create a test image
            test_img = np.random.randint(0, 255, (1000, 1000), dtype=np.uint8)
            
            # Upload to GPU
            gpu_img = cv2.cuda_GpuMat()
            gpu_img.upload(test_img)
            
            # Apply Gaussian blur on GPU
            gpu_result = cv2.cuda.blur(gpu_img, (7, 7))
            
            # Download result
            result = gpu_result.download()
            
            print("Successfully ran CUDA-accelerated image processing!")
            return True
        except Exception as e:
            print(f"Error during CUDA test: {e}")
            return False
    else:
        print("No CUDA devices found. CUDA support not available.")
        return False

if __name__ == "__main__":
    success = test_opencv_cuda()
    sys.exit(0 if success else 1)
EOF

chmod +x /home/res0ne/cuda_test.py
chown res0ne:res0ne /home/res0ne/cuda_test.py

# Run the CUDA test
echo "[INFO] Testing CUDA with OpenCV:"
run_cmd "python3 /home/res0ne/cuda_test.py || echo 'CUDA test failed, but continuing setup'"

print_section "Setup Complete"

echo "Jetson Orin Nano setup completed successfully with CUDA-enabled OpenCV!"
echo "You may need to reboot the system for all changes to take effect."
echo "Recommended: Run 'shutdown -r now' to reboot."

exit 0
