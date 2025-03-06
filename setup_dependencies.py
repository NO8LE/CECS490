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

def check_numpy_version():
    """
    Check NumPy version and compatibility with OpenCV
    
    Returns:
        (is_compatible, message): Tuple with compatibility status and message
    """
    try:
        import numpy as np
        numpy_version = np.__version__
        major_version = int(numpy_version.split('.')[0])
        
        if major_version >= 2:
            return (False, f"NumPy version {numpy_version} may be incompatible with OpenCV. "
                   f"Consider downgrading NumPy: pip install numpy<2.0")
        return (True, f"NumPy version {numpy_version} is compatible with OpenCV")
    except ImportError:
        return (False, "NumPy is not installed")
    except Exception as e:
        return (False, f"Error checking NumPy version: {str(e)}")

def check_aruco_module():
    """
    Check if the OpenCV ArUco module is available
    
    Returns:
        True if the ArUco module is available, False otherwise
    """
    try:
        # Check NumPy compatibility first
        numpy_compatible, numpy_message = check_numpy_version()
        if not numpy_compatible:
            print(f"Warning: {numpy_message}")
        
        import cv2
        print(f"OpenCV version: {cv2.__version__}")
        
        # Check if aruco module is available
        if hasattr(cv2, 'aruco'):
            print("ArUco module found in cv2.aruco")
            aruco = cv2.aruco
        else:
            # Try to import as a separate module (older OpenCV versions)
            try:
                from cv2 import aruco
                print("ArUco module imported from cv2.aruco")
                # Make it available as cv2.aruco for consistency
                cv2.aruco = aruco
            except ImportError:
                # For Jetson with custom OpenCV builds
                try:
                    sys.path.append('/usr/lib/python3/dist-packages/cv2/python-3.10')
                    from cv2 import aruco
                    print("ArUco module imported from Jetson-specific path")
                    cv2.aruco = aruco
                except (ImportError, FileNotFoundError):
                    print("ArUco module not found in any location")
                    return False
        
        # Try to access ArUco module with both old and new API
        try:
            # Try old API first
            aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
            print("ArUco module verified using old API (Dictionary_get)")
            return True
        except Exception as e:
            # If old API fails, try new API
            try:
                aruco_dict = cv2.aruco.Dictionary.get(cv2.aruco.DICT_6X6_250)
                print("ArUco module verified using new API (Dictionary.get)")
                return True
            except Exception as e2:
                print(f"ArUco module found but not working: {str(e2)}")
                return False
    except Exception as e:
        print(f"Error checking ArUco module: {str(e)}")
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
    
    # Check NumPy version compatibility
    numpy_compatible, numpy_message = check_numpy_version()
    if not numpy_compatible:
        print(f"\nWarning: {numpy_message}")
        print("OpenCV is not compatible with NumPy 2.x. You need to downgrade NumPy:")
        print("  pip install numpy<2.0")
        print("After downgrading NumPy, run this script again.\n")
    
    # Check and install required packages
    missing_packages = []
    for package_name, module_name in REQUIRED_PACKAGES.items():
        try:
            if check_package(module_name, module_name):
                print(f"✓ {package_name} is installed")
            else:
                print(f"✗ {package_name} is not installed")
                missing_packages.append(package_name)
        except Exception as e:
            print(f"✗ Error checking {package_name}: {str(e)}")
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
    try:
        if check_aruco_module():
            print("✓ OpenCV ArUco module is available")
        else:
            print("✗ OpenCV ArUco module is not available")
            print("  This may be due to an incompatible OpenCV version or NumPy version.")
            if not numpy_compatible:
                print("  First downgrade NumPy: pip install numpy<2.0")
            print("  Then try installing a different version of OpenCV:")
            print("  pip install opencv-python==4.5.4.60 opencv-contrib-python==4.5.4.60")
    except Exception as e:
        print(f"✗ Error checking ArUco module: {str(e)}")
        print("  This is likely due to NumPy 2.x incompatibility with OpenCV.")
        print("  Please downgrade NumPy: pip install numpy<2.0")
    
    # Final message
    if not missing_packages and numpy_compatible and check_aruco_module():
        print("\nAll dependencies are installed and ready to use!")
        print("You can now run the ArUco marker detection scripts:")
        print("  python3 generate_aruco_markers.py")
        print("  python3 calibrate_camera.py")
        print("  python3 oak_d_aruco_6x6_detector.py")
    else:
        print("\nSome dependencies are missing or not properly configured.")
        if not numpy_compatible:
            print("The main issue is NumPy 2.x incompatibility with OpenCV.")
            print("Please downgrade NumPy: pip install numpy<2.0")
        print("Please resolve the issues above before running the scripts.")

if __name__ == "__main__":
    main()
