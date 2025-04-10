# Direct Training Approach for Video Segmentation

Since we encountered issues with the official SAM2 training framework, this alternative approach provides a simpler way to train a segmentation model on our custom dataset.

## What This Provides

- A lightweight U-Net style model for image segmentation
- Direct PyTorch training without complex dependencies
- Simplified training and inference scripts

## Training

To train the model on your dataset:

```bash
python src/train_direct.py --batch_size 4 --num_epochs 20 --img_size 640
```

### Arguments

- `--data_dir`: Path to the dataset directory (default: 'dataset')
- `--output_dir`: Directory to save model checkpoints (default: 'models/direct')
- `--img_size`: Image size for training (default: 640)
- `--batch_size`: Batch size for training (default: 8)
- `--num_workers`: Number of workers for data loading (default: 4)
- `--learning_rate`: Learning rate (default: 1e-4)
- `--num_epochs`: Number of epochs to train for (default: 20)

## Inference

After training, you can run inference on videos:

```bash
python src/inference_direct.py \
    --checkpoint models/direct/best_model.pth \
    --video sample_video.mp4 \
    --output_dir inference_results/direct \
    --img_size 640 \
    --threshold 0.5 \
    --save_frames
```

### Arguments

- `--checkpoint`: Path to model checkpoint (required)
- `--video`: Path to input video (required)
- `--output_dir`: Path to output directory (default: 'inference_results/direct')
- `--img_size`: Size to resize frames to for model input (default: 640)
- `--threshold`: Threshold for mask prediction (default: 0.5)
- `--save_frames`: Save individual frames and masks (optional flag)

## Differences from SAM2

While this approach doesn't use the full SAM2 architecture, it provides:

1. **Simplicity**: Easy to understand and modify
2. **Speed**: Faster training on CPU or limited hardware
3. **Reliability**: Fewer dependencies and complex configurations

The model is a simplified U-Net that produces binary segmentation masks, which is suitable for our task of segmenting moving objects in videos.

## Dataset Format

This approach uses the original dataset format:

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

## Performance Considerations

- Reduce `img_size` for faster training (e.g., 320 or 256)
- Use GPU if available for significant speedup
- Start with fewer epochs (5-10) to test if the model is learning correctly 