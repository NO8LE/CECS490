#!/usr/bin/env python3

"""
Install Dependencies for OAK-D ArUco Marker Detection

This script checks for and installs the required dependencies for the OAK-D ArUco marker detection scripts.
It combines the functionality of setup_dependencies.py, fix_numpy_compatibility.py, and install_requirements.py.

Usage:
  python3 install_dependencies.py [--force]

  --force: Force reinstallation of all packages

The script will:
1. Check if required packages are installed
2. Install missing packages with specific versions known to work
3. Verify OpenCV ArUco module is available
4. Handle NumPy compatibility issues
5. Provide Jetson-specific instructions when needed
"""

import sys
import subprocess
import importlib.util
import platform
import os
import argparse

# Required packages with specific versions known to work
REQUIRED_PACKAGES = {
    "numpy": ("numpy", "1.26.4"),
    "opencv-contrib-python": ("cv2", "4.5.5.62"),
    "depthai": ("depthai", "2.24.0.0"),
    "scipy": ("scipy", "1.15.2"),
}

# Check if running on Jetson platform
def is_jetson_platform():
    """
    Check if running on a Jetson platform
    
    Returns:
        True if running on a Jetson platform, False otherwise
    """
    try:
        # Check for Jetson-specific files
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

def check_package(module_name):
    """
    Check if a package is installed
    
    Args:
        module_name: The name of the module to import
        
    Returns:
        (is_installed, version): Tuple with installation status and version string
    """
    try:
        spec = importlib.util.find_spec(module_name)
        if spec is None:
            return (False, None)
        
        # Try to get the version
        try:
            module = importlib.import_module(module_name)
            version = getattr(module, "__version__", "unknown")
            return (True, version)
        except:
            return (True, "unknown")
    except ImportError:
        return (False, None)

def install_package(package_name, version=None):
    """
    Install a package using pip
    
    Args:
        package_name: The name of the package to install
        version: Optional specific version to install
        
    Returns:
        True if the installation was successful, False otherwise
    """
    try:
        if version:
            package_spec = f"{package_name}=={version}"
        else:
            package_spec = package_name
            
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_spec])
        return True
    except subprocess.CalledProcessError:
        return False

def check_aruco_module():
    """
    Check if the OpenCV ArUco module is available
    
    Returns:
        (is_available, method): Tuple with availability status and method used
    """
    try:
        import cv2
        print(f"OpenCV version: {cv2.__version__}")
        
        # Check if aruco module is available
        if not hasattr(cv2, 'aruco'):
            # Try to import aruco from opencv-contrib-python
            try:
                from cv2 import aruco
                cv2.aruco = aruco
                print("ArUco module imported from cv2.aruco")
            except ImportError:
                # For Jetson with custom OpenCV builds
                try:
                    sys.path.append('/usr/lib/python3/dist-packages/cv2/python-3.10')
                    from cv2 import aruco
                    cv2.aruco = aruco
                    print("ArUco module imported from Jetson-specific path")
                except (ImportError, FileNotFoundError):
                    return (False, None)
        else:
            print("ArUco module found in cv2.aruco")
        
        # Try to access ArUco module with different methods
        try:
            # Try old API first
            aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
            print("ArUco module verified using Dictionary_get")
            return (True, "Dictionary_get")
        except Exception as e1:
            # If old API fails, try new API
            try:
                aruco_dict = cv2.aruco.Dictionary.get(cv2.aruco.DICT_6X6_250)
                print("ArUco module verified using Dictionary.get")
                return (True, "Dictionary.get")
            except Exception as e2:
                # If both methods fail, try to create a dictionary directly
                try:
                    aruco_dict = cv2.aruco.Dictionary.create(cv2.aruco.DICT_6X6_250)
                    print("ArUco module verified using Dictionary.create")
                    return (True, "Dictionary.create")
                except Exception as e3:
                    # Last resort: try to create a dictionary with parameters
                    try:
                        aruco_dict = cv2.aruco.Dictionary(cv2.aruco.DICT_6X6_250)
                        print("ArUco module verified using Dictionary constructor")
                        return (True, "Dictionary")
                    except Exception as e4:
                        print(f"ArUco module found but not working correctly")
                        print(f"Dictionary_get error: {str(e1)}")
                        print(f"Dictionary.get error: {str(e2)}")
                        print(f"Dictionary.create error: {str(e3)}")
                        print(f"Dictionary constructor error: {str(e4)}")
                        return (False, None)
    except Exception as e:
        print(f"Error checking ArUco module: {str(e)}")
        return (False, None)

def main():
    """
    Main function
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Install dependencies for OAK-D ArUco marker detection')
    parser.add_argument('--force', action='store_true', help='Force reinstallation of all packages')
    args = parser.parse_args()
    
    print("Checking and installing dependencies for OAK-D ArUco Marker Detection...")
    
    # Check Python version
    python_version = sys.version_info
    print(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 6):
        print("Error: Python 3.6 or higher is required.")
        sys.exit(1)
    
    # Check operating system
    os_name = platform.system()
    print(f"Operating system: {os_name}")
    
    # Check if running on Jetson platform
    jetson_platform = is_jetson_platform()
    if jetson_platform:
        print("Detected Jetson platform")
        print("Note: On Jetson platforms, OpenCV is typically installed system-wide")
        print("and may have specific version requirements.")
    
    # Check and install required packages
    missing_packages = []
    installed_packages = []
    
    for package_name, (module_name, version) in REQUIRED_PACKAGES.items():
        is_installed, current_version = check_package(module_name)
        
        if is_installed and not args.force:
            print(f"✓ {package_name} is installed (version: {current_version})")
            installed_packages.append(package_name)
        else:
            if is_installed and args.force:
                print(f"! {package_name} is installed (version: {current_version}) but will be reinstalled")
            else:
                print(f"✗ {package_name} is not installed")
            missing_packages.append((package_name, version))
    
    # Install missing packages
    if missing_packages:
        print("\nInstalling packages...")
        for package_name, version in missing_packages:
            print(f"Installing {package_name} {version}...")
            if install_package(package_name, version):
                print(f"✓ {package_name} {version} installed successfully")
                installed_packages.append(package_name)
            else:
                print(f"✗ Failed to install {package_name} {version}")
                print(f"  Please install {package_name} manually using:")
                print(f"  pip install {package_name}=={version}")
    
    # Check NumPy version for compatibility with OpenCV
    numpy_installed, numpy_version = check_package("numpy")
    if numpy_installed:
        try:
            major_version = int(numpy_version.split('.')[0])
            if major_version >= 2:
                print(f"\nWarning: NumPy version {numpy_version} may be incompatible with some OpenCV versions.")
                print("If you encounter issues, try downgrading NumPy:")
                print("  pip install numpy<2.0")
        except:
            pass
    
    # Check ArUco module
    print("\nChecking OpenCV ArUco module...")
    aruco_available, aruco_method = check_aruco_module()
    
    if aruco_available:
        print(f"✓ OpenCV ArUco module is available (using {aruco_method})")
    else:
        print("✗ OpenCV ArUco module is not available")
        print("  This may be due to an incompatible OpenCV version.")
        print("  Try installing a specific version of OpenCV:")
        print("  pip install opencv-contrib-python==4.5.5.62")
        
        if jetson_platform:
            print("\nFor Jetson platforms, you might need to install it differently:")
            print("  sudo apt-get install python3-opencv")
    
    # Final message
    if len(installed_packages) == len(REQUIRED_PACKAGES) and aruco_available:
        print("\nAll dependencies are installed and ready to use!")
        print("You can now run the ArUco marker detection scripts:")
        print("  python3 generate_aruco_markers.py")
        print("  python3 calibrate_camera.py")
        print("  python3 oak_d_aruco_6x6_detector.py")
    else:
        print("\nSome dependencies are missing or not properly configured.")
        print("Please resolve the issues above before running the scripts.")
        
        # Print the working package list from the user's feedback
        print("\nHere's a list of packages known to work together:")
        print("  numpy==1.26.4")
        print("  opencv-contrib-python==4.5.5.62")
        print("  depthai==2.24.0.0")
        print("  scipy==1.15.2")
        print("\nYou can install them all at once with:")
        print("  pip install numpy==1.26.4 opencv-contrib-python==4.5.5.62 depthai==2.24.0.0 scipy==1.15.2")

if __name__ == "__main__":
    main()
