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

# Target NumPy version (latest 1.x version)
TARGET_NUMPY_VERSION = "1.26.4"

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
    
    # Check current NumPy version
    current_version, is_compatible = check_numpy_version()
    
    if current_version is None:
        print("NumPy is not installed.")
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
