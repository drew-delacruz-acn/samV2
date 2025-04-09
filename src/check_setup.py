import sys
import torch
import torchvision
import cv2
import numpy as np
from pathlib import Path

def check_cuda():
    print("\n=== CUDA Configuration ===")
    print(f"CUDA is available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name()}")
        print(f"Device count: {torch.cuda.device_count()}")
    else:
        print("No CUDA devices available. Running on CPU only.")

def check_versions():
    print("\n=== Package Versions ===")
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Torchvision version: {torchvision.__version__}")
    print(f"OpenCV version: {cv2.__version__}")
    print(f"NumPy version: {np.__version__}")

def check_directories():
    print("\n=== Directory Structure ===")
    required_dirs = ['src', 'data', 'models', 'outputs']
    root_dir = Path(__file__).parent.parent
    
    for dir_name in required_dirs:
        dir_path = root_dir / dir_name
        exists = dir_path.exists()
        print(f"{dir_name}: {'✓' if exists else '✗'}")
        if not exists:
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"Created directory: {dir_path}")

def main():
    print("=== SAM2 Setup Verification ===")
    check_versions()
    check_cuda()
    check_directories()
    
    print("\n=== Setup Verification Complete ===")
    if not torch.cuda.is_available():
        print("\nNote: CUDA is not available. For optimal performance:")
        print("- On Windows: Install CUDA Toolkit and cuDNN")
        print("- On macOS: CUDA is not supported, but you can use MPS if available")
        if torch.backends.mps.is_available():
            print("  MPS (Metal Performance Shaders) is available on this Mac")
        else:
            print("  MPS is not available on this Mac")

if __name__ == "__main__":
    main() 