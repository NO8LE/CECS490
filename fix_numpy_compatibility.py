#!/usr/bin/env python3

"""
Fix NumPy Compatibility for OpenCV

This script downgrades NumPy to a version compatible with OpenCV.
OpenCV is not compatible with NumPy 2.x, so this script installs NumPy 1.x.

Usage:
  python3 fix_numpy_compatibility.py

The script will:
1. Check the current NumPy version
2. Downgrade NumPy to 1.26.4 if needed
3. Verify the installation
"""

import sys
import subprocess
import importlib.util
import platform

# Target NumPy version (latest 1.x version)
TARGET_NUMPY_VERSION = "1.26.4"

# Check if running on Jetson platform
def is_jetson_platform():
    """
    Check if running on a Jetson platform
    
    Returns:
        True if running on a Jetson platform, False otherwise
    """
    try:
        # Check for Jetson-specific files
        import os
        if os.path.exists('/etc/nv_tegra_release'):
            return True
        
        # Check CPU info for Jetson-specific info
        with open('/proc/cpuinfo', 'r') as f:
            cpuinfo = f.read()
            if 'tegra' in cpuinfo.lower():
                return True
        
        return False
    except:
        return False

def check_numpy_version():
    """
    Check the current NumPy version
    
    Returns:
        (version, is_compatible): Tuple with version string and compatibility status
    """
    try:
        import numpy as np
        version = np.__version__
        major_version = int(version.split('.')[0])
        return (version, major_version < 2)
    except ImportError:
        return (None, False)

def install_numpy_version(version):
    """
    Install a specific version of NumPy
    
    Args:
        version: The version to install
        
    Returns:
        True if the installation was successful, False otherwise
    """
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", f"numpy=={version}"])
        return True
    except subprocess.CalledProcessError:
        return False

def main():
    """
    Main function
    """
    print("Checking NumPy compatibility for OpenCV...")
    
    # Check if running on Jetson platform
    jetson_platform = is_jetson_platform()
    if jetson_platform:
        print("Detected Jetson platform")
        print("Note: On Jetson platforms, OpenCV is typically installed system-wide")
        print("and may have specific NumPy version requirements.")
        print("\nRecommended approach for Jetson:")
        print("1. Use the system Python and OpenCV installation")
        print("2. Install compatible NumPy with: sudo apt-get install python3-numpy")
        print("3. If needed, create a virtual environment with access to system packages:")
        print("   python3 -m venv --system-site-packages my_env\n")
    
    # Check current NumPy version
    current_version, is_compatible = check_numpy_version()
    
    if current_version is None:
        print("NumPy is not installed.")
        if jetson_platform:
            print("For Jetson platforms, install NumPy with:")
            print("  sudo apt-get install python3-numpy")
            return
        
        print(f"Installing NumPy {TARGET_NUMPY_VERSION}...")
        if install_numpy_version(TARGET_NUMPY_VERSION):
            print(f"✓ NumPy {TARGET_NUMPY_VERSION} installed successfully")
        else:
            print(f"✗ Failed to install NumPy {TARGET_NUMPY_VERSION}")
            print("  Please install NumPy manually using:")
            print(f"  pip install numpy=={TARGET_NUMPY_VERSION}")
        return
    
    print(f"Current NumPy version: {current_version}")
    
    if is_compatible:
        print("✓ Your NumPy version is compatible with OpenCV")
        print("No action needed.")
        return
    
    # Downgrade NumPy
    print(f"NumPy {current_version} is not compatible with OpenCV.")
    
    if jetson_platform:
        print("For Jetson platforms, it's recommended to use the system packages:")
        print("  sudo apt-get remove python3-numpy")
        print("  sudo apt-get install python3-numpy")
        print("\nAlternatively, you can try installing a specific version:")
        print(f"  pip install numpy=={TARGET_NUMPY_VERSION}")
        return
    
    print(f"Downgrading to NumPy {TARGET_NUMPY_VERSION}...")
    
    if install_numpy_version(TARGET_NUMPY_VERSION):
        print(f"✓ NumPy downgraded to {TARGET_NUMPY_VERSION} successfully")
        
        # Verify the installation
        new_version, is_compatible = check_numpy_version()
        if is_compatible:
            print(f"✓ NumPy {new_version} is now compatible with OpenCV")
            print("You can now run the ArUco marker detection scripts.")
        else:
            print(f"✗ NumPy {new_version} is still not compatible with OpenCV")
            print("  Please try restarting your Python environment or terminal.")
    else:
        print(f"✗ Failed to downgrade NumPy to {TARGET_NUMPY_VERSION}")
        print("  Please downgrade NumPy manually using:")
        print(f"  pip install numpy=={TARGET_NUMPY_VERSION}")

if __name__ == "__main__":
    main()
