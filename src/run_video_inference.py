#!/usr/bin/env python
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import argparse
from tqdm import tqdm
from pathlib import Path

# Import SAM2 video predictor
from sam2.sam2_video_predictor import SAM2VideoPredictor

def run_video_inference(video_path, output_dir, model_id="facebook/sam2-hiera-tiny", device="cpu"):
    """
    Run SAM2 video predictor inference on a video.
    
    Args:
        video_path (str): Path to the video file
        output_dir (str): Directory to save output masks and visualizations
        model_id (str): Model ID for SAM2 (e.g., facebook/sam2-hiera-tiny)
        device (str): Device to run inference on (cpu or cuda)
    """
    print(f"Running SAM2 video inference on {video_path}")
    print(f"Using model: {model_id} on device: {device}")
    
    # Create output directory
    masks_dir = os.path.join(output_dir, "masks")
    frames_dir = os.path.join(output_dir, "frames")
    vis_dir = os.path.join(output_dir, "visualizations")
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)
    
    # Initialize SAM2 video predictor
    print("Loading the SAM2 video predictor...")
    predictor = SAM2VideoPredictor.from_pretrained(model_id, device=device)
    
    # Initialize state with the video
    print("Initializing state with video...")
    with torch.inference_mode():
        state = predictor.init_state(video_path, offload_video_to_cpu=True)
        print(f"Successfully initialized state with {state['num_frames']} frames")
        print(f"Video dimensions: {state['video_width']}x{state['video_height']}")
        
        # Extract and save frames from the video using OpenCV instead of accessing from state
        video_capture = cv2.VideoCapture(video_path)
        if not video_capture.isOpened():
            print(f"Error: Could not open video {video_path}")
            return
            
        # Read first frame
        ret, first_frame_bgr = video_capture.read()
        if not ret:
            print("Error: Could not read first frame")
            return
            
        # Convert BGR to RGB for processing
        first_frame = cv2.cvtColor(first_frame_bgr, cv2.COLOR_BGR2RGB)
        
        # Save the first frame
        cv2.imwrite(os.path.join(frames_dir, "frame_0000.jpg"), first_frame_bgr)
        
        # Define tracking points - select central point for simplicity
        h, w = state["video_height"], state["video_width"]
        center_x, center_y = w // 2, h // 2
        
        # For better results, we'll add multiple points to specify the target object
        # Create a grid of points around the center
        grid_size = 3
        grid_spread = 50  # pixels
        points = []
        
        for i in range(grid_size):
            for j in range(grid_size):
                x = center_x + (i - grid_size//2) * grid_spread
                y = center_y + (j - grid_size//2) * grid_spread
                points.append([x, y])
        
        # Convert to numpy array
        points = np.array([points])
        labels = np.ones((1, len(points[0])), dtype=np.int64)  # All foreground points
        
        print(f"Adding {len(points[0])} points to track in the first frame...")
        
        # Add points to the first frame
        frame_idx, obj_ids, masks = predictor.add_new_points_or_box(
            inference_state=state,
            frame_idx=0,
            obj_id=0,
            points=points,
            labels=labels
        )
        
        print(f"Successfully added points to frame {frame_idx}")
        print(f"Object IDs: {obj_ids}")
        
        # Save the first frame mask
        first_mask = masks[0].cpu().numpy()
        
        # Ensure mask is in the correct format for imwrite (single channel)
        if len(first_mask.shape) > 2:
            # If it's a multi-channel mask, convert to single channel
            if first_mask.shape[0] == 1:
                first_mask = first_mask[0]  # Take first channel if it's [1, H, W]
            else:
                first_mask = np.mean(first_mask, axis=0)  # Average if multi-channel
        
        # Ensure binary mask
        first_mask_binary = (first_mask > 0).astype(np.uint8) * 255
        cv2.imwrite(os.path.join(masks_dir, f"mask_0000.png"), first_mask_binary)
        
        # Create visualization for the first frame
        create_frame_visualization(first_frame, first_mask, points[0], 
                                  os.path.join(vis_dir, "vis_0000.jpg"))
        
        # Propagate through the video
        print("Propagating segmentation through the video...")
        
        # Process one frame at a time to save frames and masks
        frame_count = 1
        for frame_idx, obj_ids, masks in tqdm(predictor.propagate_in_video(state), 
                                           desc="Tracking through video",
                                           total=state["num_frames"] - 1):
            # Save mask for current frame
            for i, mask in enumerate(masks):
                mask_np = mask.cpu().numpy()
                
                # Ensure mask is in the correct format for imwrite (single channel)
                if len(mask_np.shape) > 2:
                    # If it's a multi-channel mask, convert to single channel
                    if mask_np.shape[0] == 1:
                        mask_np = mask_np[0]  # Take first channel if it's [1, H, W]
                    else:
                        mask_np = np.mean(mask_np, axis=0)  # Average if multi-channel
                
                # Ensure binary mask
                mask_binary = (mask_np > 0).astype(np.uint8) * 255
                cv2.imwrite(os.path.join(masks_dir, f"mask_{frame_count:04d}.png"), mask_binary)
            
            # Read the next frame from the video
            ret, curr_frame_bgr = video_capture.read()
            if not ret:
                print(f"Error: Could not read frame {frame_count}")
                break
                
            # Convert BGR to RGB for processing
            curr_frame = cv2.cvtColor(curr_frame_bgr, cv2.COLOR_BGR2RGB)
            
            # Save the current frame
            cv2.imwrite(os.path.join(frames_dir, f"frame_{frame_count:04d}.jpg"), curr_frame_bgr)
            
            # Create visualization for this frame
            create_frame_visualization(curr_frame, mask_np, None,
                                     os.path.join(vis_dir, f"vis_{frame_count:04d}.jpg"))
            
            frame_count += 1
        
        # Close the video capture
        video_capture.release()
        
        print(f"Processed {frame_count} frames")
    
    # Create a video from the visualizations
    create_mp4_from_frames(vis_dir, os.path.join(output_dir, "tracking_result.mp4"))
    
    print(f"Saved all results to {output_dir}")
    print(f"- Masks: {masks_dir}")
    print(f"- Frames: {frames_dir}")
    print(f"- Visualizations: {vis_dir}")
    print(f"- Result video: {os.path.join(output_dir, 'tracking_result.mp4')}")

def create_frame_visualization(frame, mask, points=None, output_path=None):
    """
    Create a visualization of a frame with its mask and optional tracking points.
    
    Args:
        frame (np.ndarray): The video frame
        mask (np.ndarray): Binary mask
        points (np.ndarray, optional): Points used for tracking
        output_path (str, optional): Path to save the visualization
    """
    # Ensure mask is in the correct format for visualization
    if len(mask.shape) > 2:
        # If it's a multi-channel mask, convert to single channel
        if mask.shape[0] == 1:
            mask = mask[0]  # Take first channel if it's [1, H, W]
        else:
            mask = np.mean(mask, axis=0)  # Average if multi-channel
    
    # Ensure binary mask
    mask_binary = (mask > 0).astype(np.uint8)
    
    # Create a colored overlay
    overlay = np.zeros_like(frame)
    overlay[mask_binary > 0] = [0, 255, 0]  # Green for the masked area
    
    # Blend with original frame
    alpha = 0.5
    blended = cv2.addWeighted(frame, 1, overlay, alpha, 0)
    
    # Add points if provided
    if points is not None:
        for x, y in points:
            cv2.circle(blended, (int(x), int(y)), 5, (255, 0, 0), -1)  # Red circle
    
    # Add a border around the mask
    contours, _ = cv2.findContours(mask_binary, 
                                  cv2.RETR_EXTERNAL, 
                                  cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(blended, contours, -1, (0, 0, 255), 2)  # Blue contour
    
    # Save if output path is provided
    if output_path:
        cv2.imwrite(output_path, cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))
    
    return blended

def create_mp4_from_frames(frames_dir, output_video_path, fps=15):
    """
    Create an MP4 video from a sequence of frames.
    
    Args:
        frames_dir (str): Directory containing frame images
        output_video_path (str): Path to save the output video
        fps (int): Frames per second for the output video
    """
    print(f"Creating video from frames in {frames_dir}...")
    
    # Get all frame paths
    frame_paths = sorted([os.path.join(frames_dir, f) for f in os.listdir(frames_dir) 
                        if f.endswith('.jpg') or f.endswith('.png')])
    
    if not frame_paths:
        print(f"No frames found in {frames_dir}")
        return
    
    # Read the first frame to get dimensions
    first_frame = cv2.imread(frame_paths[0])
    h, w, _ = first_frame.shape
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))
    
    # Write frames to video
    for frame_path in tqdm(frame_paths, desc="Writing frames to video"):
        frame = cv2.imread(frame_path)
        video_writer.write(frame)
    
    video_writer.release()
    print(f"Video saved to {output_video_path}")

def main():
    parser = argparse.ArgumentParser(description="Run SAM2 video predictor inference")
    parser.add_argument("--video", type=str, default="sample_video.mp4", 
                        help="Path to the video file")
    parser.add_argument("--output", type=str, default="video_predictor_results",
                        help="Directory to save results")
    parser.add_argument("--model", type=str, default="facebook/sam2-hiera-tiny",
                        help="SAM2 model ID")
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["cpu", "cuda"], help="Device to run inference on")
    
    args = parser.parse_args()
    
    # Convert relative to absolute path if needed
    video_path = os.path.abspath(args.video)
    output_dir = os.path.abspath(args.output)
    
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return 1
    
    # Run inference
    run_video_inference(video_path, output_dir, args.model, args.device)
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main()) 