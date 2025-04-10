# SAM2 Fine-Tuning Project

This repository contains tools and scripts for fine-tuning the Segment Anything Model 2 (SAM2) for specialized object tracking tasks. The project includes image and video inference testing capabilities, along with a framework for synthetic data generation and model fine-tuning.

## System Requirements

### Hardware Requirements
- **CPU**: 4+ cores recommended
- **RAM**: 16GB minimum, 32GB recommended
- **GPU**: NVIDIA GPU with CUDA support (optional but recommended for training)
- **Storage**: 10GB minimum for base setup, 1TB+ recommended for synthetic dataset generation

### Software Requirements
- Python 3.8 or higher
- CUDA Toolkit 11.7 or higher (for GPU support)
- Git

## Setup Instructions

### macOS Setup

1. **Install Command Line Tools**
   ```bash
   xcode-select --install
   ```

2. **Install Homebrew** (if not already installed)
   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```

3. **Install Python**
   ```bash
   brew install python@3.11
   ```

4. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/samV2.git
   cd samV2
   ```

5. **Create and Activate Virtual Environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

6. **Install Dependencies**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

### Windows Setup

1. **Install Python**
   - Download Python 3.11 from [python.org](https://www.python.org/downloads/)
   - During installation, check "Add Python to PATH"

2. **Install Git**
   - Download and install Git from [git-scm.com](https://git-scm.com/download/win)

3. **Install Visual Studio Build Tools**
   - Download from [visualstudio.microsoft.com](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
   - Install "Desktop development with C++"

4. **Clone the Repository**
   ```cmd
   git clone https://github.com/yourusername/samV2.git
   cd samV2
   ```

5. **Create and Activate Virtual Environment**
   ```cmd
   python -m venv venv
   .\venv\Scripts\activate
   ```

6. **Install Dependencies**
   ```cmd
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

### GPU Support (Optional)

For GPU acceleration, install PyTorch with CUDA support:

**macOS**: 
- Note: PyTorch with CUDA is not supported on macOS. The model will run on CPU only.

**Windows**:
```cmd
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## Running the Tests

### Image Inference Test
```bash
# From the project root directory
python src/run_image_inference.py
```

### Video Inference Test
```bash
# From the project root directory
python src/run_video_inference.py
```

Results will be saved in:
- `image_predictor_results/` for image inference
- `video_predictor_results/` for video inference

## Project Structure

```
samV2/
├── src/
│   ├── run_image_inference.py
│   ├── run_video_inference.py
│   └── debug_sam2.py
├── sam2/
│   └── sam2_video_predictor.py
├── requirements.txt
├── README.md
└── SAM2_MARVEL_TRACKING_FEASIBILITY.md
```

## Common Issues and Solutions

### macOS
1. **OpenCV Installation Issues**
   ```bash
   brew install opencv
   pip install opencv-python
   ```

2. **Permission Denied Errors**
   ```bash
   chmod +x src/*.py
   ```

### Windows
1. **CUDA Not Found**
   - Verify CUDA installation: `nvidia-smi`
   - Ensure PyTorch CUDA version matches installed CUDA version

2. **DLL Load Failed**
   - Add Python and CUDA to system PATH
   - Reinstall Visual C++ Redistributable

## Additional Resources

- [SAM2 Documentation](https://github.com/facebookresearch/segment-anything)
- [PyTorch Installation Guide](https://pytorch.org/get-started/locally/)
- [CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/)

## License

[Add your license information here]

## Contributing

[Add contribution guidelines here]

## Running on Windows (CPU Only)

These instructions guide you through setting up and running the SAM2 fine-tuning process on a Windows machine using only the CPU. Note that training on a CPU will be significantly slower than using a GPU.

### 1. Prerequisites
- **Python:** Ensure you have Python installed. You can download it from [python.org](https://www.python.org/). Make sure to check the option "Add Python to PATH" during installation.
- **Git:** Install Git for Windows from [git-scm.com](https://git-scm.com/) to clone the repository.

### 2. Setup Steps
1.  **Clone the Repository:**
    Open Command Prompt (cmd) or PowerShell and run:
    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```
    (Replace `<repository_url>` and `<repository_directory>` with the actual URL and folder name).

2.  **Create a Virtual Environment:**
    It's highly recommended to use a virtual environment:
    ```bash
    python -m venv venv
    .\venv\Scripts\activate
    ```
    You should see `(venv)` appear at the beginning of your command prompt line.

3.  **Install Dependencies (CPU Version):**
    Install the required Python packages. Crucially, install the CPU-specific version of PyTorch:
    ```bash
    pip install --upgrade pip
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    pip install -r requirements.txt
    ```
    *Note: The `requirements.txt` might contain a GPU-specific version of PyTorch. Installing the CPU version first might resolve this, but you may need to manually edit `requirements.txt` to remove any specific `torch` lines if conflicts occur.*

### 3. Prepare Dataset and Configuration
1.  **Dataset:** Ensure you have the dataset (e.g., the dummy dataset) available. Follow the instructions in [Generating the Dummy Dataset](#generating-the-dummy-dataset) (or link to relevant doc) if you haven't created it yet.
2.  **Configuration:**
    *   Locate the relevant training configuration file (e.g., within `sam2_configs/` or specified via command-line).
    *   Modify the configuration to explicitly use the CPU. This often involves setting a `device` parameter to `"cpu"`. For example, if using `src/train_sam2_official.py`, you might need to pass a command-line argument like `--device cpu` or modify a configuration file loaded by the script. Check the script's argument parser or associated config files for the exact parameter name.

### 4. Run Fine-Tuning
Execute the training script, ensuring it's configured to use the CPU. The exact command will depend on the specific training script and its arguments. For example:
```bash
python src/train_sam2_official.py --config-name <your_config_file> --device cpu [other_arguments...]
```
*(Replace `<your_config_file>` and `[other_arguments...]` as needed)*

### 5. Performance Note
Training deep learning models like SAM2 is computationally intensive. Expect the fine-tuning process to take a very long time on a CPU compared to a GPU. This setup is primarily for testing or environments where a GPU is unavailable. 