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
