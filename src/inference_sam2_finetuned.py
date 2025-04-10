#!/usr/bin/env python
import os
import argparse
import torch
import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm

# Get the absolute path to the project root directory
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_finetuned_model(checkpoint_path, model_size='base'):
    """
    Load a fine-tuned SAM2 model from checkpoint.
    
    Args:
        checkpoint_path (str): Path to the model checkpoint
        model_size (str): Model size (tiny, small, base, or large)
        
    Returns:
        model: SAM2 predictor
    """
    try:
        # Import from SAM2 package
        from sam2.sam2_video_predictor import SAM2VideoPredictor
        
        # Map model size to model name
        size_to_name = {
            'tiny': 'facebook/sam2-hiera-tiny',
            'small': 'facebook/sam2-hiera-small',
            'base': 'facebook/sam2-hiera-base-plus',
            'large': 'facebook/sam2-hiera-large',
        }
        model_name = size_to_name.get(model_size, 'facebook/sam2-hiera-base-plus')
        
        # Load the model from checkpoint
        print(f"Loading SAM2 {model_size} model from checkpoint: {checkpoint_path}")
        model = SAM2VideoPredictor.from_checkpoint(
            model_path=checkpoint_path,
            model_name=model_name
        )
        
        # Set device
        if torch.cuda.is_available():
            model.to("cuda")
            print("Model loaded on GPU")
        else:
            print("CUDA not available, using CPU")
        
        return model
    
    except ImportError:
        raise ImportError("Failed to import SAM2VideoPredictor. Make sure SAM2 is installed properly.")
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")

def run_inference_on_video(model, video_path, output_dir, save_frames=False):
    """
    Run inference on a video and save the results.
    
    Args:
        model: SAM2 Video Predictor
        video_path (str): Path to the input video
        output_dir (str): Directory to save the results
        save_frames (bool): Whether to save individual frames
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
        raise ValueError(f"Cannot open video file: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Prepare for video writing
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_path = os.path.join(output_dir, 'segmentation.mp4')
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    
    # For SAM2 video predictor
    frame_idx = 0
    frames = []
    prompts = None  # We'll add a prompt on the first frame
    
    # Read the first frame to initialize
    ret, first_frame = cap.read()
    if not ret:
        raise ValueError("Cannot read the first frame from the video")
    
    # Reset the video capture
    cap.release()
    cap = cv2.VideoCapture(video_path)
    
    # Process each frame
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Add frame to list
        frames.append(frame_rgb.copy())
        
        # On the first frame, add a prompt at the center of the frame
        if frame_idx == 0:
            # Create a center point prompt
            center_x, center_y = width // 2, height // 2
            prompts = [
                {
                    'points': [[center_x, center_y]],
                    'labels': [1],
                    'frame_idx': 0
                }
            ]
            
            # Visualize the prompt point
            cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
        
        frame_idx += 1
    
    print(f"Loaded {len(frames)} frames")
    
    # Process the video with SAM2
    if len(frames) > 0:
        print("Running SAM2 inference...")
        video_output = model(frames, prompts)
        
        # Process the output masks
        for i, (frame, mask) in enumerate(zip(frames, video_output['masks'])):
            # Convert mask to uint8 for visualization
            mask_vis = (mask * 255).astype(np.uint8)
            
            # Apply color to mask
            colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
            colored_mask[mask > 0] = [0, 255, 0]  # Green mask
            
            # Blend mask with frame
            alpha = 0.5
            blended = cv2.addWeighted(frame, 1, colored_mask, alpha, 0)
            
            # Convert back to BGR for saving
            blended_bgr = cv2.cvtColor(blended, cv2.COLOR_RGB2BGR)
            out.write(blended_bgr)
            
            # Save individual frames if requested
            if save_frames:
                cv2.imwrite(os.path.join(frames_dir, f'{i:05d}.jpg'), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                cv2.imwrite(os.path.join(masks_dir, f'{i:05d}.png'), mask_vis)
    
    # Release resources
    cap.release()
    out.release()
    
    print(f"Segmentation saved to {out_path}")
    return out_path

def main():
    parser = argparse.ArgumentParser(description="Inference with fine-tuned SAM2")
    parser.add_argument('--checkpoint', type=str, required=True, 
                        help='Path to the fine-tuned model checkpoint')
    parser.add_argument('--video', type=str, required=True,
                        help='Path to the input video file')
    parser.add_argument('--output-dir', type=str, default='inference_results',
                        help='Directory to save the results')
    parser.add_argument('--model-size', type=str, default='base',
                        choices=['tiny', 'small', 'base', 'large'],
                        help='SAM2 model size that was fine-tuned')
    parser.add_argument('--save-frames', action='store_true',
                        help='Save individual frames and masks')
    
    args = parser.parse_args()
    
    # Resolve paths
    if not os.path.isabs(args.checkpoint):
        args.checkpoint = os.path.join(ROOT_DIR, args.checkpoint)
    
    if not os.path.isabs(args.video):
        args.video = os.path.join(ROOT_DIR, args.video)
    
    if not os.path.isabs(args.output_dir):
        args.output_dir = os.path.join(ROOT_DIR, args.output_dir)
    
    # Load the model
    model = load_finetuned_model(args.checkpoint, args.model_size)
    
    # Run inference
    run_inference_on_video(model, args.video, args.output_dir, args.save_frames)

if __name__ == "__main__":
    main() 