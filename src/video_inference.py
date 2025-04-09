import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
from pathlib import Path
import torch

# Import SAM2 video predictor
from sam2.sam2_video_predictor import SAM2VideoPredictor


def create_mp4_from_frames(images_dir, output_video_path, fps=30):
    """
    Create an MP4 video from a sequence of frames.
    
    Args:
        images_dir (str): Directory containing frame images (.jpg)
        output_video_path (str): Path to save the output video
        fps (int): Frames per second for the output video
    
    Returns:
        str: Path to the created video
    """
    print(f"Creating MP4 video from frames in {images_dir}")
    
    # Get all frame paths
    frame_paths = sorted([os.path.join(images_dir, f) for f in os.listdir(images_dir) if f.endswith('.jpg')])
    
    if not frame_paths:
        raise ValueError(f"No frames found in {images_dir}")
    
    # Read the first frame to get dimensions
    first_frame = cv2.imread(frame_paths[0])
    height, width, _ = first_frame.shape
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    # Write frames to video
    for frame_path in tqdm(frame_paths, desc="Writing frames to video"):
        frame = cv2.imread(frame_path)
        video_writer.write(frame)
    
    video_writer.release()
    print(f"Created video: {output_video_path}")
    
    return output_video_path


def find_object_centers(masks_dir, frame_basename):
    """
    Find object centers in a mask.
    
    Args:
        masks_dir (str): Directory containing mask files
        frame_basename (str): Base name of the frame file
    
    Returns:
        list: List of (x, y) center coordinates for objects in the mask
    """
    mask_path = os.path.join(masks_dir, frame_basename.replace('.jpg', '.png'))
    
    if not os.path.exists(mask_path):
        return []
    
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    # Find connected components (objects) in the mask
    _, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    
    # Skip the first component (background)
    centers = []
    for i in range(1, len(centroids)):
        x, y = centroids[i]
        centers.append((int(x), int(y)))
    
    return centers


def run_video_predictor_inference(video_dir, output_dir, model_size='tiny'):
    """
    Run SAM2 video predictor inference on a test video.
    
    Args:
        video_dir (str): Directory containing video frames and masks
        output_dir (str): Directory to save output masks and visualizations
        model_size (str): Size of the SAM2 model to use
    """
    images_dir = os.path.join(video_dir, 'images')
    masks_dir = os.path.join(video_dir, 'masks')
    
    if not os.path.exists(images_dir) or not os.path.exists(masks_dir):
        print(f"Images or masks directory not found in {video_dir}")
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create MP4 video from frames
    video_path = os.path.join(output_dir, 'input_video.mp4')
    create_mp4_from_frames(images_dir, video_path)
    
    # Determine model name
    if model_size == "tiny":
        model_name = "facebook/sam2-hiera-tiny"
    elif model_size == "small":
        model_name = "facebook/sam2-hiera-small"
    elif model_size == "base":
        model_name = "facebook/sam2-hiera-base-plus"
    else:  # Default to large
        model_name = "facebook/sam2-hiera-large"
    
    # Initialize SAM2 video predictor
    print(f"Loading SAM2 model: {model_name}")
    device = "cpu"  # Use CPU for compatibility
    
    try:
        # Load the model
        predictor = SAM2VideoPredictor.from_pretrained(model_name, device=device)
        predictor.to(device)
        print("Successfully loaded SAM2 video predictor")
        
        # Initialize state with the video
        print("Initializing state with video...")
        with torch.inference_mode():
            state = predictor.init_state(video_path, offload_video_to_cpu=True)
            print(f"Successfully initialized state with {state['num_frames']} frames")
            
            # Get the first frame from the test video to use as a prompt point
            first_frame_basename = sorted(os.listdir(images_dir))[0]
            object_centers = find_object_centers(masks_dir, first_frame_basename)
            
            if not object_centers:
                # If no objects found, use the center of the frame
                h, w = state["video_height"], state["video_width"]
                object_centers = [(w // 2, h // 2)]
                print(f"No objects found in masks, using center point: {object_centers[0]}")
            else:
                print(f"Found {len(object_centers)} objects in first frame: {object_centers}")
            
            # Create points and labels for the first frame
            points = torch.tensor([object_centers], dtype=torch.float32).to(device)
            labels = torch.ones((1, len(object_centers)), dtype=torch.int64).to(device)
            
            # Add points to the first frame to establish the object to track
            print("Adding points to the first frame...")
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
            cv2.imwrite(os.path.join(output_dir, f'mask_{frame_idx:04d}.png'), first_mask * 255)
            
            # Propagate through the video
            print("Propagating segmentation through the video...")
            for frame_idx, obj_ids, masks in tqdm(predictor.propagate_in_video(state), 
                                                 desc="Tracking through video",
                                                 total=state["num_frames"] - 1):
                # Save mask for current frame
                for i, mask in enumerate(masks):
                    mask_np = mask.cpu().numpy()
                    cv2.imwrite(os.path.join(output_dir, f'mask_{frame_idx:04d}.png'), mask_np * 255)
            
            print("Video processing complete")
            
            # Create visualizations
            create_visualizations(video_dir, output_dir, max_frames=5)
            
    except Exception as e:
        print(f"Error using video predictor: {e}")
        import traceback
        traceback.print_exc()


def create_visualizations(video_dir, output_dir, max_frames=5):
    """
    Create visualizations comparing ground truth masks with predictions.
    
    Args:
        video_dir (str): Directory containing original frames and ground truth masks
        output_dir (str): Directory containing predicted masks
        max_frames (int): Maximum number of frames to visualize
    """
    print("Creating visualizations...")
    
    images_dir = os.path.join(video_dir, 'images')
    gt_masks_dir = os.path.join(video_dir, 'masks')
    
    # Create visualization directory
    vis_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    
    # Get frame files
    frame_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.jpg')])
    
    # Limit to max_frames
    frame_files = frame_files[:max_frames]
    
    for i, frame_file in enumerate(frame_files):
        # Get frame number
        frame_idx = i
        
        # Load original frame
        frame_path = os.path.join(images_dir, frame_file)
        frame = cv2.imread(frame_path)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Load ground truth mask
        gt_mask_path = os.path.join(gt_masks_dir, frame_file.replace('.jpg', '.png'))
        if os.path.exists(gt_mask_path):
            gt_mask = cv2.imread(gt_mask_path, cv2.IMREAD_GRAYSCALE)
        else:
            gt_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
        
        # Load predicted mask
        pred_mask_path = os.path.join(output_dir, f'mask_{frame_idx:04d}.png')
        if os.path.exists(pred_mask_path):
            pred_mask = cv2.imread(pred_mask_path, cv2.IMREAD_GRAYSCALE)
        else:
            pred_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(frame)
        axes[0].set_title(f'Frame {frame_idx}')
        axes[0].axis('off')
        
        axes[1].imshow(gt_mask, cmap='gray')
        axes[1].set_title('Ground Truth')
        axes[1].axis('off')
        
        axes[2].imshow(pred_mask, cmap='gray')
        axes[2].set_title('SAM2 Prediction')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        # Save visualization
        plt.savefig(os.path.join(vis_dir, f'comparison_{frame_idx:04d}.png'))
        plt.close()
    
    print(f"Visualizations saved to {vis_dir}")


def calculate_metrics(pred_dir, gt_dir, output_path=None):
    """Calculate evaluation metrics between predicted and ground truth masks"""
    pred_masks_dir = os.path.join(pred_dir, 'masks')
    all_metrics = []
    
    # Get list of predicted mask files
    pred_files = sorted([f for f in os.listdir(pred_masks_dir) if f.endswith('.png')])
    
    if not pred_files:
        print(f"No prediction files found in {pred_masks_dir}")
        return {}
    
    for pred_file in pred_files:
        frame_number = pred_file.split('.')[0].split('_')[-1]
        gt_file = f"mask_{frame_number}.png"
        gt_path = os.path.join(gt_dir, 'masks', gt_file)
        
        if os.path.exists(gt_path):
            pred_mask = cv2.imread(os.path.join(pred_masks_dir, pred_file), cv2.IMREAD_GRAYSCALE)
            gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
            
            # Convert to binary masks (1 for mask, 0 for background)
            pred_binary = (pred_mask > 0).astype(np.uint8)
            gt_binary = (gt_mask > 0).astype(np.uint8)
            
            # Calculate metrics
            intersection = np.logical_and(pred_binary, gt_binary).sum()
            union = np.logical_or(pred_binary, gt_binary).sum()
            
            true_positives = intersection
            false_positives = pred_binary.sum() - intersection
            false_negatives = gt_binary.sum() - intersection
            
            # Avoid division by zero
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            iou = intersection / union if union > 0 else 0
            dice = 2 * intersection / (pred_binary.sum() + gt_binary.sum()) if (pred_binary.sum() + gt_binary.sum()) > 0 else 0
            
            metrics = {
                'frame': frame_number,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'iou': iou,
                'dice': dice
            }
            all_metrics.append(metrics)
    
    # Calculate average metrics
    if not all_metrics:
        print("No metrics could be calculated. Check if ground truth masks exist.")
        avg_metrics = {
            'avg_precision': 0,
            'avg_recall': 0,
            'avg_f1': 0,
            'avg_iou': 0,
            'avg_dice': 0
        }
    else:
        avg_metrics = {
            'avg_precision': sum(m['precision'] for m in all_metrics) / len(all_metrics),
            'avg_recall': sum(m['recall'] for m in all_metrics) / len(all_metrics),
            'avg_f1': sum(m['f1'] for m in all_metrics) / len(all_metrics),
            'avg_iou': sum(m['iou'] for m in all_metrics) / len(all_metrics),
            'avg_dice': sum(m['dice'] for m in all_metrics) / len(all_metrics)
        }
    
    # Print and save metrics
    print("Video Predictor Metrics:")
    for key, value in avg_metrics.items():
        print(f"{key}: {value:.4f}")
    
    if output_path:
        with open(output_path, 'w') as f:
            f.write("Video Predictor Metrics:\n")
            for key, value in avg_metrics.items():
                f.write(f"{key}: {value:.4f}\n")
            
            if all_metrics:
                f.write("\nPer-frame metrics:\n")
                for m in all_metrics:
                    f.write(f"Frame {m['frame']}: IoU={m['iou']:.4f}, Dice={m['dice']:.4f}\n")
    
    return avg_metrics


def main():
    parser = argparse.ArgumentParser(description="Run SAM2 video predictor inference on test videos")
    
    parser.add_argument('--video_dir', type=str, required=True,
                        help='Directory containing video frames and masks')
    parser.add_argument('--output_dir', type=str, default='video_predictor_results',
                        help='Directory to save output masks and visualizations')
    parser.add_argument('--model_size', type=str, default='tiny',
                        choices=['tiny', 'small', 'base', 'large'],
                        help='Size of the SAM2 model to use')
    parser.add_argument('--calculate_metrics', action='store_true',
                        help='Calculate metrics after inference')
    
    args = parser.parse_args()
    
    # Check if video_dir exists
    if not os.path.exists(args.video_dir):
        print(f"Video directory not found: {args.video_dir}")
        return
    
    # Run video predictor inference
    run_video_predictor_inference(args.video_dir, args.output_dir, args.model_size)
    
    # Calculate metrics if requested
    if args.calculate_metrics:
        gt_masks_dir = os.path.join(args.video_dir, 'masks')
        metrics_path = os.path.join(args.output_dir, 'metrics.txt')
        calculate_metrics(args.output_dir, gt_masks_dir, metrics_path)


if __name__ == "__main__":
    main() 