#!/bin/bash

# Jetson Orin Nano Setup Script
# This script configures an Nvidia Jetson Orin Nano SoM

set -e  # Exit on error
set -o pipefail

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

# Update and upgrade system packages
run_cmd "apt-get update && apt-get upgrade -y"
run_cmd "apt-get clean && apt-get autoremove -y"

# Add repositories
run_cmd "apt-add-repository universe -y"
run_cmd "apt-add-repository multiverse -y"
run_cmd "apt-add-repository restricted -y"

# Install basic tools and dependencies
run_cmd "apt-get install -y dkms git build-essential python3 python3-venv python3-pip cmake udev libusb-1.0-0-dev"
run_cmd "apt-get install -y gstreamer1.0-plugins-bad gstreamer1.0-libav gstreamer1.0-gl"
run_cmd "apt-get install -y libfuse2 libxcb-xinerama0 libxkbcommon-x11-0 libxcb-cursor-dev"

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

# Check if NVMe drive exists
if [ -e /dev/nvme0n1 ]; then
    echo "[INFO] NVMe drive detected, setting up..."
    
    # Format and mount NVMe drive
    run_cmd "mkfs.ext4 -F /dev/nvme0n1"
    run_cmd "mkdir -p /mnt"
    run_cmd "mount /dev/nvme0n1 /mnt"
    
    # Add to fstab for persistence
    if ! grep -q "/dev/nvme0n1" /etc/fstab; then
        echo "/dev/nvme0n1 /mnt ext4 defaults 0 2" >> /etc/fstab
        echo "[INFO] Added NVMe mount to /etc/fstab"
    fi
    
    # Move Docker data directory to NVMe
    echo "[INFO] Moving Docker data directory to NVMe"
    run_cmd "systemctl stop docker"
    run_cmd "mkdir -p /mnt/docker"
    
    # Check if Docker data directory exists and is not empty
    if [ -d /var/lib/docker ] && [ "$(ls -A /var/lib/docker)" ]; then
        run_cmd "rsync -axPS /var/lib/docker/ /mnt/docker/"
        run_cmd "mv /var/lib/docker /var/lib/docker.old"
    fi
    
    # Configure Docker to use the new data directory
    cat > /etc/docker/daemon.json <<EOF
{
    "data-root": "/mnt/docker",
    "runtimes": {
        "nvidia": {
            "path": "nvidia-container-runtime",
            "runtimeArgs": []
        }
    }
}
EOF
    echo "[INFO] Updated Docker daemon configuration"
    
    run_cmd "systemctl daemon-reload"
    run_cmd "systemctl restart docker"
    
    # Create swap file on NVMe
    echo "[INFO] Creating swap file on NVMe"
    run_cmd "systemctl disable nvzramconfig"
    run_cmd "fallocate -l 16G /mnt/jetson.swap"
    run_cmd "chmod 600 /mnt/jetson.swap"
    run_cmd "mkswap /mnt/jetson.swap"
    run_cmd "swapon /mnt/jetson.swap"
    
    # Add swap to fstab
    if ! grep -q "jetson.swap" /etc/fstab; then
        echo "/mnt/jetson.swap none swap sw 0 0" >> /etc/fstab
        echo "[INFO] Added swap to /etc/fstab"
    fi
else
    echo "[WARNING] NVMe drive not detected, skipping storage configuration"
fi

print_section "DepthAI/Luxonis Setup"

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
mkdir -p /home/seeker/ros

# Create script for ROS Jazzy
cat > /home/seeker/ros/run_ros_jazzy.sh <<EOF
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
chmod +x /home/seeker/ros/run_ros_jazzy.sh

# Create script for ROS Humble
cat > /home/seeker/ros/run_ros_humble.sh <<EOF
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
chmod +x /home/seeker/ros/run_ros_humble.sh

# Create workspace directories
run_cmd "mkdir -p /home/seeker/ros_ws_jazzy/src"
run_cmd "mkdir -p /home/seeker/ros_ws_humble/src"

# Set ownership
run_cmd "chown -R seeker:seeker /home/seeker/ros"
run_cmd "chown -R seeker:seeker /home/seeker/ros_ws_jazzy"
run_cmd "chown -R seeker:seeker /home/seeker/ros_ws_humble"

# Add ROS container aliases to seeker's .bashrc
cat >> /home/seeker/.bashrc <<EOF

# ROS container aliases
alias ros-jazzy='/home/seeker/ros/run_ros_jazzy.sh'
alias ros-humble='/home/seeker/ros/run_ros_humble.sh'
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

print_section "User Configuration - Create and Setup 'seeker' User"

# Create seeker user if it doesn't exist
if ! id -u seeker &>/dev/null; then
    run_cmd "useradd -m -s /bin/bash seeker"
    run_cmd "usermod -aG sudo seeker"
    echo "[INFO] Created user 'seeker'"
    
    # Set a default password that should be changed on first login
    echo "seeker:jetson" | chpasswd
    run_cmd "passwd -e seeker"  # Force password change on first login
else
    echo "[INFO] User 'seeker' already exists"
fi

# Add seeker to necessary groups
run_cmd "usermod -aG dialout,video,docker seeker"

# Configure sudoers for passwordless sudo access
cat > /etc/sudoers.d/seeker <<EOF
# Allow seeker to run sudo commands without password
seeker ALL=(ALL) NOPASSWD:ALL
EOF
run_cmd "chmod 440 /etc/sudoers.d/seeker"

# Create directory structure similar to seeker
run_cmd "mkdir -p /home/seeker/git"
run_cmd "mkdir -p /home/seeker/QGroundControl"
run_cmd "mkdir -p /home/seeker/CECS490/OAK-D"
run_cmd "mkdir -p /home/seeker/tmp"

# Clone repositories from seeker's environment
run_cmd "cd /home/seeker/git && git clone https://github.com/ArchilChovatiya/ArUco-marker-detection-with-DepthAi.git"
run_cmd "chmod +x /home/seeker/git/ArUco-marker-detection-with-DepthAi/*.py"
run_cmd "cd /home/seeker/git && git clone https://github.com/chintal/depthai-sandbox.git"
run_cmd "cd /home/seeker/git && git clone https://github.com/dusty-nv/jetson-containers"

# Install and configure jetson-containers
run_cmd "cd /home/seeker/git/jetson-containers && bash install.sh"
run_cmd "chown -R seeker:seeker /home/seeker/git/jetson-containers"

# Install QGroundControl for seeker
run_cmd "wget -O /home/seeker/QGroundControl/QGroundControl.AppImage https://d176tv9ibo4jno.cloudfront.net/latest/QGroundControl.AppImage"
run_cmd "chmod +x /home/seeker/QGroundControl/QGroundControl.AppImage"

# Set up Python environment for seeker
run_cmd "python3 -m venv /home/seeker/depthai-env"
run_cmd "chown -R seeker:seeker /home/seeker/depthai-env"

# Install Python packages for ArUco detection
cat > /tmp/requirements.txt <<EOF
opencv-contrib-python
numpy
matplotlib
scikit-learn
depthai
depthai-viewer
blobconverter
vedo
EOF

run_cmd "pip3 install -r /tmp/requirements.txt"

# Set up .bashrc for seeker with environment variables
cat >> /home/seeker/.bashrc <<EOF

# DepthAI environment variables
export OPENBLAS_CORETYPE=ARMV8

# Alias for convenience
alias ll='ls -la'
EOF

# Set up SSH directory for seeker
run_cmd "mkdir -p /home/seeker/.ssh"
run_cmd "chmod 700 /home/seeker/.ssh"

# Add the specified SSH public key to authorized_keys
cat > /home/seeker/.ssh/authorized_keys <<EOF
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIPg7eHztrQQBVwLLSPlw7hryxD3yK6QvCEi8sa+KNgiW
EOF

run_cmd "chmod 600 /home/seeker/.ssh/authorized_keys"

# Fix permissions for all files
run_cmd "chown -R seeker:seeker /home/seeker"

print_section "Additional Software Installation"

# Install QGroundControl system-wide
run_cmd "mkdir -p /opt/QGroundControl"
run_cmd "wget -O /opt/QGroundControl/QGroundControl.AppImage https://d176tv9ibo4jno.cloudfront.net/latest/QGroundControl.AppImage"
run_cmd "chmod +x /opt/QGroundControl/QGroundControl.AppImage"

# Install Firefox
run_cmd "apt-get install -y firefox"

print_section "Setup Complete"

echo "Jetson Orin Nano setup completed successfully!"
echo "You may need to reboot the system for all changes to take effect."
echo "Recommended: Run 'shutdown -r now' to reboot."

exit 0
