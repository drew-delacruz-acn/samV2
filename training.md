# Training Guide for SAM2 Segmentation Model

This guide provides step-by-step instructions for training the segmentation model on your custom dataset. It's designed for easy migration to a VM with GPU support.

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Project Structure](#project-structure)
3. [Dataset Preparation](#dataset-preparation)
4. [Environment Setup](#environment-setup)
5. [Training Process](#training-process)
6. [Training Options](#training-options)
7. [Monitoring Training](#monitoring-training)
8. [Troubleshooting](#troubleshooting)

## Prerequisites

- Python 3.8+ installed
- CUDA-compatible GPU (recommended for faster training)
- 8+ GB RAM (16+ GB recommended)
- 10+ GB free disk space for dataset and models

## Project Structure

The recommended project structure is:

```
segmentation-project/
├── src/                      # Source code directory
│   ├── train.py              # Training script
│   ├── dataset.py            # Dataset loading functionality
│   ├── dataset_generator.py  # (Optional) Synthetic dataset generator
│   ├── predict.py            # Prediction script for single images
│   └── predict_on_dataset.py # Evaluation script for test dataset
├── dataset/                  # Dataset directory
│   ├── train/                # Training data
│   │   ├── video1/
│   │   │   ├── images/       # Images for video1
│   │   │   └── masks/        # Corresponding masks for video1
│   │   ├── video2/
│   │   └── ...
│   └── test/                 # Testing data (same structure as train)
├── models/                   # Directory to save model checkpoints
├── predictions/              # Directory for prediction outputs
├── evaluations/              # Directory for evaluation results
├── requirements.txt          # Dependencies
└── README.md                 # Project information
```

## Dataset Preparation

### Option 1: Using the Built-in Synthetic Dataset Generator

The project includes a script to generate a synthetic dataset with moving circles:

```bash
python src/dataset_generator.py
```

This will create a dataset with 3 training videos and 2 test videos, each with 100 frames.

### Option 2: Using Your Custom Dataset

Follow these steps to prepare your custom dataset:

1. Create the directory structure:
   ```
   dataset/
   ├── train/
   │   ├── [video_name1]/
   │   │   ├── images/
   │   │   └── masks/
   │   └── [video_name2]/
   │       ├── images/
   │       └── masks/
   └── test/
       └── [video_name]/
           ├── images/
           └── masks/
   ```

2. Image requirements:
   - Put frames as JPG files in the `images/` folders
   - Name them sequentially (e.g., 0000.jpg, 0001.jpg, etc.)
   - Any resolution is acceptable (will be resized during training)

3. Mask requirements:
   - Put binary segmentation masks as PNG files in the `masks/` folders
   - Masks must have the same names as their corresponding images (e.g., 0000.png for 0000.jpg)
   - Masks should be binary (0 for background, 255 for foreground)

4. Organizing videos:
   - Each separate "video" should be in its own subdirectory
   - You can have any number of videos in train and test sets

## Environment Setup

### Local Setup

1. Clone the repository (if applicable)
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### VM Setup with GPU Support

1. Launch a VM with GPU support (e.g., AWS EC2 p2/p3/g4 instances, GCP N1 with GPU)

2. Install CUDA and cuDNN (if not pre-installed)
   ```bash
   # Example for Ubuntu with CUDA 11.6
   wget https://developer.download.nvidia.com/compute/cuda/11.6.0/local_installers/cuda_11.6.0_510.39.01_linux.run
   sudo sh cuda_11.6.0_510.39.01_linux.run
   ```

3. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. Install PyTorch with CUDA support:
   ```bash
   # For CUDA 11.6
   pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu116
   ```

5. Install other dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Training Process

1. **Prepare your dataset** as described in the Dataset Preparation section.

2. **Create output directories**:
   ```bash
   mkdir -p models predictions evaluations
   ```

3. **Start training**:
   ```bash
   python src/train.py --data_dir dataset --output_dir models --model_size tiny --num_epochs 10 --batch_size 8
   ```

4. **Use GPU acceleration** (if available):
   ```bash
   # Simply run the same command - GPU will be auto-detected
   python src/train.py --data_dir dataset --output_dir models --model_size tiny --num_epochs 10 --batch_size 8
   ```

5. **For higher GPU memory utilization**, increase batch size:
   ```bash
   python src/train.py --batch_size 16 --num_epochs 10
   ```

## Training Options

The `train.py` script supports various command-line arguments:

| Argument | Description | Default |
|----------|-------------|---------|
| `--data_dir` | Path to dataset directory | `dataset` |
| `--output_dir` | Directory to save model checkpoints | `models` |
| `--img_size` | Input image size for training | `224` |
| `--batch_size` | Batch size | `8` |
| `--num_workers` | Number of workers for data loading | `4` |
| `--learning_rate` | Learning rate | `1e-4` |
| `--num_epochs` | Number of epochs | `10` |
| `--model_size` | Model size (tiny, small, base, large) | `small` |
| `--use_combined_loss` | Use combined BCE and Dice loss | `False` |
| `--use_autocast` | Use automatic mixed precision | `False` |

## Monitoring Training

During training, the script outputs these metrics for each epoch:

- **Training Loss**: Average loss on the training dataset
- **Validation Loss**: Average loss on the validation dataset
- **Best Model**: Saved whenever validation loss improves

Example output:
```
Epoch 1/10
Training: 100%|█████████████| 75/75 [00:58<00:00, 1.29it/s]
Validation: 100%|█████████| 50/50 [00:30<00:00, 1.62it/s]
Train Loss: 1.2274, Validation Loss: 0.1300
Saved best model checkpoint to models/best_model.pth
```

The best model is saved to `models/best_model.pth` and can be used for inference.

## Troubleshooting

### Common Issues

1. **Out of Memory (OOM) error**:
   - Reduce batch size: `--batch_size 4` or even `--batch_size 2`
   - Reduce image size: `--img_size 160`
   - Reduce model size: `--model_size tiny`

2. **CUDA errors**:
   - Ensure PyTorch was installed with the correct CUDA version
   - Try reinstalling: `pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu113`

3. **Slow training on CPU**:
   - Use GPU if available
   - Reduce the number of workers: `--num_workers 0`
   - Reduce batch size to avoid swapping

4. **Dataset loading errors**:
   - Check that your directory structure follows the expected format
   - Ensure image and mask filenames match exactly (except for file extension)
   - Verify that masks are properly formatted (binary, with values 0 and 255)

5. **Poor convergence**:
   - Try longer training: increase `--num_epochs` to 20 or 50
   - Adjust learning rate: `--learning_rate 1e-3` or `--learning_rate 5e-5`
   - Use the combined loss function: add `--use_combined_loss`

### Performance Tips

1. For better GPU utilization:
   - Increase batch size if memory allows
   - Set `--num_workers` to (number of CPU cores - 1)

2. For faster training:
   - Use a smaller model size: `--model_size tiny`
   - Consider using mixed precision: add `--use_autocast`

3. For better results:
   - Train longer (more epochs)
   - Use the combined loss function: `--use_combined_loss`
   - Consider using a larger model size if GPU memory allows

4. For VM with limited disk space:
   - Delete older model checkpoints if not needed
   - Use smaller input images to reduce dataset size 