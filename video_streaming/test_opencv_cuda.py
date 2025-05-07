#!/usr/bin/env python3

"""
OpenCV CUDA Detection and Testing Script

This script tests OpenCV's CUDA functionality and checks for properly configured CUDA devices.
Can be used to diagnose issues with CUDA detection on Jetson platforms.

Usage:
  python3 test_opencv_cuda.py
"""

import cv2
import numpy as np
import sys
import os
import platform

def check_system_cuda():
    """Check system CUDA configuration"""
    print("\n=== SYSTEM CUDA CONFIGURATION ===")

    # Check CUDA_HOME and PATH
    cuda_home = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')
    print(f"CUDA_HOME: {cuda_home or 'Not set'}")

    # Check for CUDA in PATH
    path = os.environ.get('PATH', '')
    cuda_in_path = any('/cuda' in p for p in path.split(':'))
    print(f"CUDA in PATH: {'Yes' if cuda_in_path else 'No'}")

    # Check LD_LIBRARY_PATH
    ld_path = os.environ.get('LD_LIBRARY_PATH', '')
    cuda_in_ld = any('/cuda' in p for p in ld_path.split(':'))
    print(f"CUDA in LD_LIBRARY_PATH: {'Yes' if cuda_in_ld else 'No'}")

    # System info
    print(f"Platform: {platform.platform()}")
    print(f"Python version: {platform.python_version()}")

    # Try executing nvidia-smi
    try:
        import subprocess
        print("\nnvidia-smi output:")
        result = subprocess.run(['nvidia-smi'], check=False, capture_output=True, text=True)
        if result.returncode == 0:
            print(result.stdout)
            print("nvidia-smi executed successfully")
        else:
            print(f"nvidia-smi error: {result.stderr}")
    except FileNotFoundError:
        print("nvidia-smi not found in PATH")

def check_opencv_cuda():
    """Check OpenCV CUDA configuration"""
    print("\n=== OPENCV CUDA CONFIGURATION ===")

    # Print OpenCV version
    print(f"OpenCV version: {cv2.__version__}")

    # Check if CUDA is available in OpenCV
    cuda_devices = cv2.cuda.getCudaEnabledDeviceCount()
    print(f"\nCUDA enabled devices: {cuda_devices}")

    if cuda_devices == 0:
        print("\nERROR: OpenCV was not built with CUDA support or CUDA is not properly configured.")
        print("Possible solutions:")
        print("1. Install OpenCV with CUDA support using the updated jetson_orin_setup.sh script")
        print("2. Check that CUDA toolkit is properly installed")
        print("3. Ensure LD_LIBRARY_PATH includes CUDA libraries")
        return False

    # Print CUDA device information
    print("\nCUDA device information:")
    for i in range(cuda_devices):
        try:
            cv2.cuda.printCudaDeviceInfo(i)
        except Exception as e:
            print(f"Error getting CUDA device info: {e}")

    return True

def test_cuda_operations():
    """Test basic CUDA operations"""
    print("\n=== TESTING CUDA OPERATIONS ===")

    if cv2.cuda.getCudaEnabledDeviceCount() == 0:
        print("Skipping tests: No CUDA devices detected")
        return False

    try:
        # Create a test image
        test_img = np.random.randint(0, 255, (1000, 1000), dtype=np.uint8)

        # Upload to GPU
        start_time = cv2.getTickCount()
        gpu_img = cv2.cuda_GpuMat()
        gpu_img.upload(test_img)

        # Apply Gaussian blur on GPU
        #gpu_result = cv2.cuda.blur(gpu_img, (7, 7))
        # Create a box filter (equivalent to cv2.blur)
        blur_filter = cv2.cuda.createBoxFilter(cv2.CV_8UC1, cv2.CV_8UC1, (7, 7))
        gpu_result = blur_filter.apply(gpu_img)

        # Download result
        result = gpu_result.download()
        end_time = cv2.getTickCount()

        # Calculate processing time
        processing_time = (end_time - start_time) / cv2.getTickFrequency()
        print(f"CUDA processing time: {processing_time:.4f} seconds")

        # Apply same operation on CPU for comparison
        start_time = cv2.getTickCount()
        cpu_result = cv2.blur(test_img, (7, 7))
        end_time = cv2.getTickCount()

        # Calculate processing time
        cpu_processing_time = (end_time - start_time) / cv2.getTickFrequency()
        print(f"CPU processing time: {cpu_processing_time:.4f} seconds")

        # Compare results
        diff = np.abs(result - cpu_result).mean()
        print(f"Average difference between CPU and GPU results: {diff:.4f}")

        speedup = cpu_processing_time / processing_time
        print(f"GPU speedup: {speedup:.2f}x")

        print("\nSuccessfully ran CUDA-accelerated image processing!")
        return True
    except Exception as e:
        print(f"Error during CUDA test: {e}")
        return False

def main():
    """Main function"""
    print("===== OPENCV CUDA TEST =====")

    # Check system CUDA configuration
    check_system_cuda()

    # Check OpenCV CUDA configuration
    cuda_available = check_opencv_cuda()

    # If CUDA is available, run test operations
    success = False
    if cuda_available:
        success = test_cuda_operations()

    # Print summary
    print("\n===== TEST SUMMARY =====")
    if cuda_available and success:
        print("✅ OpenCV CUDA is working correctly!")
        print("✅ CUDA acceleration is available for ArUco detection!")
    else:
        print("❌ OpenCV CUDA test failed.")
        print("To fix this issue on Jetson Orin Nano:")
        print("1. Run the updated jetson_orin_setup.sh script to build OpenCV with CUDA support")
        print("2. Ensure the correct CUDA paths are set in environment variables")
        print("3. Check that the correct CUDA compute capability is used for the Jetson Orin Nano (8.7)")

    return 0 if (cuda_available and success) else 1

if __name__ == "__main__":
    sys.exit(main())
