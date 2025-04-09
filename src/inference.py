import os
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
from pathlib import Path

# Import SAM2 image predictor which we confirmed works
from sam2.sam2_image_predictor import SAM2ImagePredictor


def load_frames_from_dir(image_dir):
    """
    Load video frames from a directory of images.
    
    Args:
        image_dir (str): Path to directory containing frames
        
    Returns:
        frames (list): List of frames as numpy arrays
    """
    frame_paths = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.jpg')])
    frames = []
    
    for path in frame_paths:
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frames.append(img)
    
    return frames


def run_sam2_inference(video_dir, output_dir, model_size='small'):
    """
    Run SAM2 inference on a directory of video frames.
    
    Args:
        video_dir (str): Path to directory containing video frames
        output_dir (str): Path to save output masks
        model_size (str): SAM2 model size to use
    """
    # Load frames
    images_dir = os.path.join(video_dir, 'images')
    if not os.path.exists(images_dir):
        print(f"Images directory not found: {images_dir}")
        return
    
    print(f"Loading frames from {images_dir}")
    frames = load_frames_from_dir(images_dir)
    
    # Determine model name
    if model_size == "tiny":
        model_name = "facebook/sam2-hiera-tiny"
    elif model_size == "small":
        model_name = "facebook/sam2-hiera-small"
    elif model_size == "base":
        model_name = "facebook/sam2-hiera-base-plus"
    else:  # Default to large
        model_name = "facebook/sam2-hiera-large"
    
    # Initialize SAM2 predictor
    print(f"Loading SAM2 model: {model_name}")
    try:
        # Use the image predictor which we confirmed works
        predictor = SAM2ImagePredictor.from_pretrained(model_name, device="cpu")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Run inference on each frame
        print("Running SAM2 inference on each frame...")
        for i, frame in enumerate(tqdm(frames)):
            # Process each frame individually with the image predictor
            predictor.set_image(frame)
            
            # Create an automatic prompt in the center of the frame
            h, w = frame.shape[:2]
            center_y, center_x = h // 2, w // 2
            point_coords = np.array([[center_x, center_y]])
            point_labels = np.array([1])  # 1 for foreground
            
            # Get the prediction
            masks, _, _ = predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels
            )
            
            # Save the mask
            mask = masks[0].astype(np.uint8) * 255
            cv2.imwrite(os.path.join(output_dir, f'mask_{i:04d}.png'), mask)
        
        print(f"Inference complete. Results saved to {output_dir}")
        
        # Create a visualization of some frames
        visualize_results(video_dir, output_dir, max_frames=5)
        
    except Exception as e:
        print(f"Error running SAM2 inference: {e}")
        import traceback
        traceback.print_exc()


def visualize_results(video_dir, output_dir, max_frames=5):
    """
    Visualize the inference results for a few frames.
    
    Args:
        video_dir (str): Path to directory containing original frames
        output_dir (str): Path to directory containing predicted masks
        max_frames (int): Maximum number of frames to visualize
    """
    images_dir = os.path.join(video_dir, 'images')
    gt_masks_dir = os.path.join(video_dir, 'masks')
    
    # Get frame filenames
    frame_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.jpg')])
    
    # Limit to max_frames
    frame_files = frame_files[:max_frames]
    
    for i, frame_file in enumerate(frame_files):
        # Load original frame
        frame_path = os.path.join(images_dir, frame_file)
        frame = cv2.imread(frame_path)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Load ground truth mask if available
        gt_mask_path = os.path.join(gt_masks_dir, frame_file.replace('.jpg', '.png'))
        if os.path.exists(gt_mask_path):
            gt_mask = cv2.imread(gt_mask_path, cv2.IMREAD_GRAYSCALE)
        else:
            gt_mask = np.zeros_like(frame[:,:,0])
        
        # Load predicted mask
        pred_mask_path = os.path.join(output_dir, f'mask_{i:04d}.png')
        if os.path.exists(pred_mask_path):
            pred_mask = cv2.imread(pred_mask_path, cv2.IMREAD_GRAYSCALE)
        else:
            pred_mask = np.zeros_like(frame[:,:,0])
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(frame)
        axes[0].set_title(f'Frame {i}')
        axes[0].axis('off')
        
        axes[1].imshow(gt_mask, cmap='gray')
        axes[1].set_title('Ground Truth')
        axes[1].axis('off')
        
        axes[2].imshow(pred_mask, cmap='gray')
        axes[2].set_title('SAM2 Prediction')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        # Save visualization
        vis_dir = os.path.join(output_dir, 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)
        plt.savefig(os.path.join(vis_dir, f'comparison_{i:04d}.png'))
        plt.close()
    
    print(f"Visualizations saved to {os.path.join(output_dir, 'visualizations')}")


def main():
    parser = argparse.ArgumentParser(description="Run SAM2 inference on video frames")
    
    parser.add_argument('--video_dir', type=str, required=True,
                        help='Path to directory containing video frames')
    parser.add_argument('--output_dir', type=str, default='inference_results',
                        help='Directory to save inference results')
    parser.add_argument('--model_size', type=str, default='tiny',
                        choices=['tiny', 'small', 'base', 'large'],
                        help='SAM2 model size to use')
    
    args = parser.parse_args()
    
    # Make sure video_dir exists
    video_dir = Path(args.video_dir)
    if not video_dir.exists():
        print(f"Video directory not found: {video_dir}")
        return
    
    # Create output directory
    output_dir = Path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Run inference
    run_sam2_inference(str(video_dir), str(output_dir), args.model_size)


if __name__ == "__main__":
    main() 