# Training SAM2 on Custom Data

This guide explains how to fine-tune Segment Anything Model 2 (SAM2) on custom data using the official SAM2 training framework.

## Prerequisites

- Python 3.8 or higher
- PyTorch 2.0 or higher
- CUDA-compatible GPU (recommended)
- SAM2 repository (already cloned in the `sam2` directory)

## Setup

1. Install the necessary dependencies:
   ```bash
   pip install -e "sam2/.[dev]"
   pip install tqdm opencv-python matplotlib
   ```

2. Check CUDA availability:
   ```bash
   python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, Device count: {torch.cuda.device_count()}')"
   ```

## Dataset Preparation

Our scripts are designed to work with the dummy dataset that has this structure:
```
dataset/
├── train/
│   ├── video1/
│   │   ├── images/  # JPEG frames
│   │   │   ├── 0000.jpg
│   │   │   └── ...
│   │   └── masks/   # PNG masks
│   │       ├── 0000.png
│   │       └── ...
│   └── ...
└── test/
    └── ...
```

The dataset preparation script (`src/prepare_sam2_dataset.py`) will automatically convert this to the DAVIS-style format required by SAM2:
```
sam2_dataset/
├── JPEGImages/
│   ├── video1/
│   │   ├── 00000.jpg
│   │   └── ...
│   └── ...
└── Annotations/
    ├── video1/
    │   ├── 00000.png
    │   └── ...
    └── ...
```

## Training SAM2

Use the training script to fine-tune SAM2 on your dataset:

```bash
python src/train_sam2_official.py --model-size base --batch-size 1 --epochs 10 --gpus 1
```

### Arguments

- `--model-size`: SAM2 model size to use (`tiny`, `small`, `base`, or `large`)
- `--batch-size`: Batch size for training (default: 1)
- `--epochs`: Number of epochs to train for (default: 10)
- `--gpus`: Number of GPUs to use for training (default: 1)

## What Happens During Training

1. The script will first prepare your dataset in the DAVIS-style format if it hasn't been prepared yet.
2. It will create a configuration file (`sam2_configs/custom_finetune_runtime.yaml`) based on the template.
3. It will then run the SAM2 training process using the official training code.

The training logs and checkpoints will be saved in the `sam2_logs/custom_finetune` directory.

## Inference with Fine-tuned Model

After training, you can run inference with your fine-tuned model using the inference script:

```bash
python src/inference_sam2_finetuned.py \
    --checkpoint sam2_logs/custom_finetune/checkpoints/model_epoch_10.pt \
    --video sample_video.mp4 \
    --output-dir inference_results \
    --model-size base \
    --save-frames
```

### Arguments

- `--checkpoint`: Path to the fine-tuned model checkpoint (required)
- `--video`: Path to the input video file (required)
- `--output-dir`: Directory to save the results (default: `inference_results`)
- `--model-size`: SAM2 model size that was fine-tuned (default: `base`)
- `--save-frames`: Save individual frames and masks (optional)

## Monitoring Training

You can monitor the training progress using TensorBoard:

```bash
tensorboard --logdir sam2_logs/custom_finetune/tensorboard
```

## Troubleshooting

1. **Out of Memory Errors**: 
   - Reduce batch size
   - Use a smaller model size (e.g., `tiny` instead of `base`)
   - Reduce the number of frames per batch (`num_frames` in the config)

2. **Dataset Loading Issues**:
   - Ensure your dataset is correctly formatted before training
   - Check the paths in the configuration file

3. **Training Speed**:
   - Training on GPU is significantly faster than on CPU
   - The `base` and `large` models require more GPU memory
   - For faster iterations, start with the `tiny` model

## Advanced Configuration

The training configuration can be customized by editing `sam2_configs/custom_finetune.yaml` before running the training script. Important parameters include:

- `resolution`: Image resolution for training
- `train_batch_size`: Batch size for training
- `num_frames`: Number of frames per video sample
- `base_lr` and `vision_lr`: Learning rates
- `num_epochs`: Number of epochs to train for

## Further Resources

- SAM2 GitHub Repository: [https://github.com/facebookresearch/sam2](https://github.com/facebookresearch/sam2)
- SAM2 Paper: [https://arxiv.org/abs/2401.05948](https://arxiv.org/abs/2401.05948) 