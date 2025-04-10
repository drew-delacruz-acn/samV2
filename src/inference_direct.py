#!/usr/bin/env python
import os
import argparse
import torch
import torch.nn as nn
import numpy as np
import cv2
from tqdm import tqdm

# For dataset loading
from dataset import SegmentationTransform

def create_model(img_size=640):
    """
    Create the same model architecture used for training.
    This must match the architecture used in train_direct.py.
    
    Returns:
        model: Segmentation model
    """
    model = nn.Sequential(
        # Encoder
        nn.Conv2d(3, 32, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),
        
        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),
        
        nn.Conv2d(64, 128, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),
        
        # Bottleneck
        nn.Conv2d(128, 256, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        
        # Decoder
        nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
        nn.ReLU(inplace=True),
        
        nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
        nn.ReLU(inplace=True),
        
        nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
        nn.ReLU(inplace=True),
        
        # Output layer
        nn.Conv2d(32, 1, kernel_size=1),
        nn.Sigmoid()  # Output probabilities between 0 and 1
    )
    
    return model


def load_model(checkpoint_path, device):
    """
    Load trained model from checkpoint.
    
    Args:
        checkpoint_path (str): Path to model checkpoint
        device: Device to load the model on
        
    Returns:
        model: Loaded model
    """
    # Create model
    model = create_model().to(device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Loaded model from checkpoint {checkpoint_path} (epoch {checkpoint['epoch']})")
    
    # Set model to evaluation mode
    model.eval()
    
    return model


def run_inference_on_video(model, video_path, output_dir, device, img_size=640, mask_threshold=0.5, save_frames=False):
    """
    Run inference on a video.
    
    Args:
        model: The trained model
        video_path (str): Path to input video
        output_dir (str): Path to output directory
        device: The device to run inference on
        img_size (int): Size to resize frames to
        mask_threshold (float): Threshold to binarize output masks
        save_frames (bool): Whether to save individual frames
        
    Returns:
        str: Path to output video
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    if save_frames:
        frames_dir = os.path.join(output_dir, 'frames')
        masks_dir = os.path.join(output_dir, 'masks')
        os.makedirs(frames_dir, exist_ok=True)
        os.makedirs(masks_dir, exist_ok=True)
    
    # Open the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file {video_path}")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create VideoWriter for output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_path = os.path.join(output_dir, 'segmentation.mp4')
    out = cv2.VideoWriter(out_path, fourcc, fps, (frame_width, frame_height))
    
    transform = SegmentationTransform(img_size=(img_size, img_size))
    
    # Get frame count for progress bar
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Process each frame
    with torch.no_grad():
        for frame_idx in tqdm(range(total_frames), desc="Processing frames"):
            ret, frame = cap.read()
            if not ret:
                break
            
            # Preprocess frame
            # Resize for model input
            orig_frame = frame.copy()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Transform like our training data
            sample = {'image': frame_rgb, 'mask': np.zeros((img_size, img_size), dtype=np.float32)}
            sample = transform(sample)
            
            # Convert to tensor and add batch dimension
            frame_tensor = sample['image'].unsqueeze(0).to(device)
            
            # Run inference
            mask_pred = model(frame_tensor)
            
            # Convert mask to numpy and resize to original frame size
            mask_np = mask_pred.cpu().squeeze().numpy()
            mask_np = cv2.resize(mask_np, (frame_width, frame_height))
            
            # Binarize the mask using threshold
            mask_binary = (mask_np > mask_threshold).astype(np.uint8) * 255
            
            # Create color mask for visualization
            color_mask = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
            color_mask[mask_binary > 0] = [0, 255, 0]  # Green for the mask
            
            # Blend original frame with mask
            alpha = 0.5
            blended = cv2.addWeighted(orig_frame, 1, color_mask, alpha, 0)
            
            # Write to output video
            out.write(blended)
            
            # Save individual frames if requested
            if save_frames:
                cv2.imwrite(os.path.join(frames_dir, f'frame_{frame_idx:05d}.jpg'), orig_frame)
                cv2.imwrite(os.path.join(masks_dir, f'mask_{frame_idx:05d}.png'), mask_binary)
    
    # Release resources
    cap.release()
    out.release()
    
    print(f"Saved segmentation video to {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Run inference with trained segmentation model")
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--video', type=str, required=True,
                        help='Path to input video')
    parser.add_argument('--output_dir', type=str, default='inference_results/direct',
                        help='Path to output directory')
    parser.add_argument('--img_size', type=int, default=640,
                        help='Size to resize frames to for model input')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Threshold for mask prediction')
    parser.add_argument('--save_frames', action='store_true',
                        help='Save individual frames and masks')
    
    args = parser.parse_args()
    
    # Select device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = load_model(args.checkpoint, device)
    
    # Run inference
    run_inference_on_video(
        model, 
        args.video, 
        args.output_dir,
        device,
        img_size=args.img_size,
        mask_threshold=args.threshold,
        save_frames=args.save_frames
    )


if __name__ == "__main__":
    main() 