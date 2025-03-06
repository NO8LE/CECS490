#!/usr/bin/env python3

"""
Setup Dependencies for OAK-D ArUco Marker Detection

This script checks for and installs the required dependencies for the OAK-D ArUco marker detection scripts.

Usage:
  python3 setup_dependencies.py

The script will:
1. Check if required packages are installed
2. Install missing packages
3. Verify OpenCV ArUco module is available
"""

import sys
import subprocess
import importlib.util
import platform

# Required packages
REQUIRED_PACKAGES = {
    "opencv-python": "cv2",
    "opencv-contrib-python": "cv2.aruco",  # For ArUco module
    "numpy": "numpy",
    "scipy": "scipy",
    "depthai": "depthai"
}

def check_package(package_name, module_name):
    """
    Check if a package is installed
    
    Args:
        package_name: The name of the package to check
        module_name: The name of the module to import
        
    Returns:
        True if the package is installed, False otherwise
    """
    try:
        spec = importlib.util.find_spec(module_name)
        if spec is None:
            return False
        return True
    except ImportError:
        return False

def install_package(package_name):
    """
    Install a package using pip
    
    Args:
        package_name: The name of the package to install
        
    Returns:
        True if the installation was successful, False otherwise
    """
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        return True
    except subprocess.CalledProcessError:
        return False

def check_aruco_module():
    """
    Check if the OpenCV ArUco module is available
    
    Returns:
        True if the ArUco module is available, False otherwise
    """
    try:
        import cv2
        # Try to access ArUco module
        aruco_dict = cv2.aruco.Dictionary.get(cv2.aruco.DICT_6X6_250)
        return True
    except (ImportError, AttributeError):
        return False

def main():
    """
    Main function
    """
    print("Checking dependencies for OAK-D ArUco Marker Detection...")
    
    # Check Python version
    python_version = sys.version_info
    print(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 6):
        print("Error: Python 3.6 or higher is required.")
        sys.exit(1)
    
    # Check operating system
    os_name = platform.system()
    print(f"Operating system: {os_name}")
    
    # Check and install required packages
    missing_packages = []
    for package_name, module_name in REQUIRED_PACKAGES.items():
        if check_package(module_name, module_name):
            print(f"✓ {package_name} is installed")
        else:
            print(f"✗ {package_name} is not installed")
            missing_packages.append(package_name)
    
    # Install missing packages
    if missing_packages:
        print("\nInstalling missing packages...")
        for package_name in missing_packages:
            print(f"Installing {package_name}...")
            if install_package(package_name):
                print(f"✓ {package_name} installed successfully")
            else:
                print(f"✗ Failed to install {package_name}")
                print(f"  Please install {package_name} manually using:")
                print(f"  pip install {package_name}")
    
    # Check ArUco module
    print("\nChecking OpenCV ArUco module...")
    if check_aruco_module():
        print("✓ OpenCV ArUco module is available")
    else:
        print("✗ OpenCV ArUco module is not available")
        print("  This may be due to an incompatible OpenCV version.")
        print("  Try installing a different version of OpenCV:")
        print("  pip install opencv-python==4.5.4.60")
    
    # Final message
    if not missing_packages and check_aruco_module():
        print("\nAll dependencies are installed and ready to use!")
        print("You can now run the ArUco marker detection scripts:")
        print("  python3 generate_aruco_markers.py")
        print("  python3 calibrate_camera.py")
        print("  python3 oak_d_aruco_6x6_detector.py")
    else:
        print("\nSome dependencies are missing or not properly configured.")
        print("Please resolve the issues above before running the scripts.")

if __name__ == "__main__":
    main()
