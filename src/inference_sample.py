import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from sklearn.cluster import KMeans

# Import SAM2 image predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor


def find_object_center(mask):
    """
    Find the center of an object in a binary mask.
    
    Args:
        mask (ndarray): Binary mask
        
    Returns:
        tuple: (center_x, center_y) coordinates of the object center
    """
    # Use moments to find centroid
    M = cv2.moments(mask)
    
    if M["m00"] > 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        return (cX, cY)
    else:
        # If no object, return center of image
        h, w = mask.shape
        return (w // 2, h // 2)


def sample_points_from_mask(mask, num_points=5):
    """
    Sample multiple points from a binary mask.
    
    Args:
        mask (ndarray): Binary mask
        num_points (int): Number of points to sample
        
    Returns:
        ndarray: Array of (x, y) coordinates
    """
    # Find coordinates of non-zero pixels
    y_indices, x_indices = np.where(mask > 0)
    
    if len(y_indices) == 0:
        # If mask is empty, return center of image
        h, w = mask.shape
        return np.array([[w // 2, h // 2]])
    
    if len(y_indices) <= num_points:
        # If mask has fewer pixels than requested points, return all
        return np.column_stack((x_indices, y_indices))
    
    # Use K-means to find evenly distributed points
    coords = np.column_stack((x_indices, y_indices))
    kmeans = KMeans(n_clusters=num_points, random_state=42).fit(coords)
    
    # Get the closest point to each cluster center
    centers = []
    for center in kmeans.cluster_centers_:
        # Find closest point in mask to the center
        distances = np.sum((coords - center) ** 2, axis=1)
        closest_idx = np.argmin(distances)
        centers.append(coords[closest_idx])
    
    return np.array(centers)


def run_inference_on_sample(video_dir, output_dir, model_size='tiny', max_frames=5, num_points=3):
    """
    Run SAM2 inference on a small sample of frames.
    
    Args:
        video_dir (str): Path to directory containing video frames
        output_dir (str): Path to save output masks
        model_size (str): SAM2 model size to use
        max_frames (int): Maximum number of frames to process
        num_points (int): Number of points to sample from ground truth
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get frame paths
    images_dir = os.path.join(video_dir, 'images')
    masks_dir = os.path.join(video_dir, 'masks')
    
    if not os.path.exists(images_dir):
        print(f"Images directory not found: {images_dir}")
        return
    
    frame_paths = sorted([os.path.join(images_dir, f) for f in os.listdir(images_dir) if f.endswith('.jpg')])
    # Take only a few frames
    frame_paths = frame_paths[:max_frames]
    
    print(f"Found {len(frame_paths)} frames, processing {min(len(frame_paths), max_frames)}")
    
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
        predictor = SAM2ImagePredictor.from_pretrained(model_name, device="cpu")
        
        # Process frames
        for i, frame_path in enumerate(frame_paths):
            frame_basename = os.path.basename(frame_path)
            print(f"Processing frame {i+1}/{len(frame_paths)}: {frame_basename}")
            
            # Load image
            frame = cv2.imread(frame_path)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Load corresponding ground truth mask if available
            mask_path = os.path.join(masks_dir, frame_basename.replace('.jpg', '.png'))
            
            if os.path.exists(mask_path):
                gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                
                # Sample multiple points from ground truth
                point_coords = sample_points_from_mask(gt_mask, num_points)
                print(f"  Using {len(point_coords)} points from ground truth")
            else:
                # Default to center of image
                h, w = frame.shape[:2]
                point_coords = np.array([[w // 2, h // 2]])
                print(f"  Using image center: ({w // 2}, {h // 2})")
            
            # Create an array of point labels (all foreground)
            point_labels = np.ones(len(point_coords), dtype=np.int64)
            
            # Set image in predictor
            predictor.set_image(frame)
            
            # Get prediction
            masks, _, _ = predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels
            )
            
            # Save mask
            mask = masks[0].astype(np.uint8) * 255
            cv2.imwrite(os.path.join(output_dir, f'mask_{i:04d}.png'), mask)
            
            # Load ground truth mask (already loaded above if exists)
            if not os.path.exists(mask_path):
                gt_mask = np.zeros_like(mask)
            
            # Create visualization
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Plot original image with points
            axes[0].imshow(frame)
            for point in point_coords:
                axes[0].scatter(point[0], point[1], c='red', s=40)
            axes[0].set_title(f'Frame {i} with {len(point_coords)} prompts')
            axes[0].axis('off')
            
            # Plot ground truth mask
            axes[1].imshow(gt_mask, cmap='gray')
            axes[1].set_title('Ground Truth')
            axes[1].axis('off')
            
            # Plot predicted mask
            axes[2].imshow(mask, cmap='gray')
            axes[2].set_title('SAM2 Prediction')
            axes[2].axis('off')
            
            plt.tight_layout()
            
            # Save visualization
            vis_dir = os.path.join(output_dir, 'visualizations')
            os.makedirs(vis_dir, exist_ok=True)
            plt.savefig(os.path.join(vis_dir, f'comparison_{i:04d}.png'))
            plt.close()
        
        print(f"Inference complete. Results saved to {output_dir}")
        
    except Exception as e:
        print(f"Error running SAM2 inference: {e}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description="Run SAM2 inference on a sample of video frames")
    
    parser.add_argument('--video_dir', type=str, required=True,
                        help='Path to directory containing video frames')
    parser.add_argument('--output_dir', type=str, default='inference_sample',
                        help='Directory to save inference results')
    parser.add_argument('--model_size', type=str, default='tiny',
                        choices=['tiny', 'small', 'base', 'large'],
                        help='SAM2 model size to use')
    parser.add_argument('--max_frames', type=int, default=5,
                        help='Maximum number of frames to process')
    parser.add_argument('--num_points', type=int, default=3,
                        help='Number of points to sample from ground truth')
    
    args = parser.parse_args()
    
    # Make sure video_dir exists
    video_dir = Path(args.video_dir)
    if not video_dir.exists():
        print(f"Video directory not found: {video_dir}")
        return
    
    # Run inference
    run_inference_on_sample(
        str(video_dir), 
        args.output_dir, 
        args.model_size,
        args.max_frames,
        args.num_points
    )


if __name__ == "__main__":
    main() 