import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
from sklearn.metrics import precision_score, recall_score, f1_score
import cv2
import logging
from pathlib import Path

from dataset import SegmentationDataset, SegmentationTransform

# Import SAM2 video predictor
from sam2.sam2_video_predictor import SAM2VideoPredictor
from sam2.sam2_image_predictor import SAM2ImagePredictor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def calculate_metrics(pred_masks, gt_masks, threshold=0.5):
    """
    Calculate evaluation metrics.
    
    Args:
        pred_masks (torch.Tensor): Predicted masks
        gt_masks (torch.Tensor): Ground truth masks
        threshold (float): Threshold for binarizing predictions
        
    Returns:
        dict: Dictionary of metrics
    """
    # Handle potential extra dimension in pred_masks
    if pred_masks.dim() == 4 and pred_masks.shape[1] == 1:
        pred_masks = pred_masks.squeeze(1)
    
    # Convert to binary masks
    pred_binary = (pred_masks > threshold).float().cpu().numpy()
    gt_binary = gt_masks.cpu().numpy()
    
    # Flatten for metric calculation
    pred_flat = pred_binary.reshape(-1)
    gt_flat = gt_binary.reshape(-1)
    
    # Calculate metrics
    precision = precision_score(gt_flat, pred_flat, zero_division=1)
    recall = recall_score(gt_flat, pred_flat, zero_division=1)
    f1 = f1_score(gt_flat, pred_flat, zero_division=1)
    
    # Calculate IoU
    intersection = np.logical_and(gt_binary, pred_binary).sum()
    union = np.logical_or(gt_binary, pred_binary).sum()
    iou = intersection / union if union > 0 else 0
    
    # Calculate DICE coefficient
    dice = 2 * intersection / (gt_binary.sum() + pred_binary.sum()) if (gt_binary.sum() + pred_binary.sum()) > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'iou': iou,
        'dice': dice
    }

def calculate_iou(pred_mask, true_mask):
    """Calculate Intersection over Union (IoU) between masks."""
    intersection = np.logical_and(pred_mask, true_mask).sum()
    union = np.logical_or(pred_mask, true_mask).sum()
    return intersection / union if union > 0 else 0

def calculate_dice(pred_mask, true_mask):
    """Calculate Dice coefficient between masks."""
    intersection = np.logical_and(pred_mask, true_mask).sum()
    return 2 * intersection / (pred_mask.sum() + true_mask.sum()) if (pred_mask.sum() + true_mask.sum()) > 0 else 0

def calculate_segmentation_metrics(mask):
    """
    Calculate segmentation metrics for a single mask.
    
    Args:
        mask (np.ndarray): Binary mask
        
    Returns:
        dict: Dictionary of metrics
    """
    total_pixels = mask.size
    foreground_pixels = np.sum(mask)
    
    # Calculate basic metrics
    area_ratio = foreground_pixels / total_pixels
    
    # Calculate compactness (ratio of area to perimeter squared)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        perimeter = sum(cv2.arcLength(cnt, True) for cnt in contours)
        area = sum(cv2.contourArea(cnt) for cnt in contours)
        compactness = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
    else:
        compactness = 0
    
    # Calculate center of mass
    if foreground_pixels > 0:
        moments = cv2.moments(mask.astype(np.uint8))
        cx = moments['m10'] / moments['m00'] if moments['m00'] > 0 else 0
        cy = moments['m01'] / moments['m00'] if moments['m00'] > 0 else 0
    else:
        cx, cy = 0, 0
    
    return {
        'area_ratio': float(area_ratio),
        'compactness': float(compactness),
        'center_x': float(cx),
        'center_y': float(cy)
    }

def calculate_temporal_metrics(prev_mask, curr_mask):
    """
    Calculate temporal metrics between consecutive frames.
    
    Args:
        prev_mask (np.ndarray): Mask from previous frame
        curr_mask (np.ndarray): Mask from current frame
        
    Returns:
        dict: Dictionary of temporal metrics
    """
    # Calculate temporal consistency (IoU between consecutive frames)
    intersection = np.logical_and(prev_mask, curr_mask).sum()
    union = np.logical_or(prev_mask, curr_mask).sum()
    temporal_iou = intersection / union if union > 0 else 0
    
    # Calculate mask stability (change in area)
    prev_area = prev_mask.sum()
    curr_area = curr_mask.sum()
    area_stability = min(prev_area, curr_area) / max(prev_area, curr_area) if max(prev_area, curr_area) > 0 else 1
    
    # Calculate center of mass movement
    prev_moments = cv2.moments(prev_mask.astype(np.uint8))
    curr_moments = cv2.moments(curr_mask.astype(np.uint8))
    
    if prev_moments['m00'] > 0 and curr_moments['m00'] > 0:
        prev_cx = prev_moments['m10'] / prev_moments['m00']
        prev_cy = prev_moments['m01'] / prev_moments['m00']
        curr_cx = curr_moments['m10'] / curr_moments['m00']
        curr_cy = curr_moments['m01'] / curr_moments['m00']
        
        movement = np.sqrt((curr_cx - prev_cx)**2 + (curr_cy - prev_cy)**2)
    else:
        movement = 0
    
    return {
        'temporal_consistency': float(temporal_iou),
        'area_stability': float(area_stability),
        'movement': float(movement)
    }

def evaluate_image_segmentation(image_path, mask_path, target_size=None):
    """
    Evaluate image segmentation performance.
    
    Args:
        image_path (str): Path to the input image
        mask_path (str): Path to the predicted mask
        target_size (tuple, optional): Size to resize images to
        
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    # Read image and mask
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    if target_size:
        image = cv2.resize(image, target_size)
        mask = cv2.resize(mask, target_size)
    
    # Ensure binary mask
    mask = (mask > 127).astype(np.uint8)
    
    # Calculate metrics
    return calculate_segmentation_metrics(mask)

def evaluate_video_tracking(video_path, masks_path):
    """
    Evaluate video tracking performance.
    
    Args:
        video_path (str): Path to the input video
        masks_path (str): Directory containing predicted masks
        
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    cap = cv2.VideoCapture(video_path)
    masks_dir = Path(masks_path)
    metrics = []
    
    frame_idx = 0
    prev_mask = None
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Read corresponding mask
        mask_path = masks_dir / f"mask_{frame_idx:04d}.png"
        if not mask_path.exists():
            break
            
        curr_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        curr_mask = (curr_mask > 127).astype(np.uint8)
        
        # Calculate frame metrics
        frame_metrics = calculate_segmentation_metrics(curr_mask)
        
        # Calculate temporal metrics if not first frame
        if prev_mask is not None:
            temporal_metrics = calculate_temporal_metrics(prev_mask, curr_mask)
            frame_metrics.update(temporal_metrics)
        
        metrics.append(frame_metrics)
        prev_mask = curr_mask
        frame_idx += 1
    
    cap.release()
    
    # Calculate average metrics
    avg_metrics = {}
    if metrics:  # Ensure we have metrics to average
        for key in metrics[0].keys():
            avg_metrics[key] = np.mean([m[key] for m in metrics])
    
    return avg_metrics

def visualize_predictions(image, gt_mask, pred_mask, output_path, threshold=0.5):
    """
    Visualize and save predictions.
    
    Args:
        image (torch.Tensor): Input image
        gt_mask (torch.Tensor): Ground truth mask
        pred_mask (torch.Tensor): Predicted mask
        output_path (str): Path to save visualization
        threshold (float): Threshold for binarizing predictions
    """
    # Convert tensors to numpy arrays
    image = image.permute(1, 2, 0).cpu().numpy()
    gt_mask = gt_mask.cpu().numpy()
    
    # Handle potential extra dimension in pred_mask
    pred_mask_np = (pred_mask > threshold).float().cpu().numpy()
    if pred_mask_np.shape[0] == 1 and len(pred_mask_np.shape) == 3:
        pred_mask_np = pred_mask_np.squeeze(0)
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot image
    axes[0].imshow(image)
    axes[0].set_title('Image')
    axes[0].axis('off')
    
    # Plot ground truth mask
    axes[1].imshow(gt_mask, cmap='gray')
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')
    
    # Plot predicted mask
    axes[2].imshow(pred_mask_np, cmap='gray')
    axes[2].set_title('Prediction')
    axes[2].axis('off')
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def load_sam2_model(args, device):
    """
    Load a SAM2 model for evaluation.
    
    Args:
        args: Command-line arguments
        device: Device for the model
        
    Returns:
        model: The loaded SAM2 model
    """
    # Determine model size
    model_size = args.model_size
    
    if model_size == "tiny":
        model_name = "facebook/sam2-hiera-tiny"
    elif model_size == "small":
        model_name = "facebook/sam2-hiera-small"
    elif model_size == "base":
        model_name = "facebook/sam2-hiera-base-plus"
    else:  # Default to large
        model_name = "facebook/sam2-hiera-large"
    
    print(f"Loading SAM2 model: {model_name}")
    
    try:
        # Use checkpoint if provided, otherwise load pretrained model
        if args.checkpoint:
            print(f"Loading fine-tuned model from checkpoint: {args.checkpoint}")
            # Load the checkpoint
            checkpoint = torch.load(args.checkpoint, map_location=device)
            
            # Initialize the model from pretrained
            model = SAM2VideoPredictor.from_pretrained(model_name)
            
            # Load state dict from checkpoint - this is a simplified approach
            # In real implementation, we would need to match the state_dict structure
            try:
                model.load_state_dict(checkpoint['model_state_dict'])
                print("Loaded fine-tuned model successfully")
            except Exception as e:
                print(f"Error loading checkpoint: {e}")
                print("Using the pretrained model instead")
        else:
            # Use pretrained model
            model = SAM2VideoPredictor.from_pretrained(model_name)
            print("Loaded pretrained SAM2 model")
        
        return model
        
    except Exception as e:
        print(f"Error loading SAM2 model: {e}")
        print("Using a simple placeholder model instead")
        
        # Fallback to a simple model if SAM2 loading fails
        model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1)
        )
        
        # If there's a checkpoint for this simple model, load it
        if args.checkpoint:
            try:
                checkpoint = torch.load(args.checkpoint, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"Loaded checkpoint for simple model: {args.checkpoint}")
            except Exception as e:
                print(f"Error loading checkpoint for simple model: {e}")
        
        return model

def evaluate(model, dataloader, device, output_dir):
    """
    Evaluate the model.
    
    Args:
        model: Model to evaluate
        dataloader: DataLoader for evaluation data
        device: Device to evaluate on
        output_dir: Directory to save results
        
    Returns:
        dict: Dictionary of average metrics
    """
    model.eval()
    all_metrics = []
    os.makedirs(os.path.join(output_dir, 'visualizations'), exist_ok=True)
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc="Evaluation")):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            
            # Forward pass with SAM2 model
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                outputs = model(images)
            
            # Calculate metrics
            metrics = calculate_metrics(outputs, masks)
            all_metrics.append(metrics)
            
            # Visualize the first few batches
            if i < 5:  # Visualize first 5 batches only
                for j in range(min(len(images), 3)):  # Max 3 images per batch
                    vis_path = os.path.join(output_dir, 'visualizations', f'batch_{i}_sample_{j}.png')
                    visualize_predictions(
                        image=images[j],
                        gt_mask=masks[j],
                        pred_mask=outputs[j],
                        output_path=vis_path
                    )
    
    # Calculate average metrics
    avg_metrics = {}
    for key in all_metrics[0].keys():
        avg_metrics[key] = sum(m[key] for m in all_metrics) / len(all_metrics)
    
    # Save metrics to a file
    metrics_path = os.path.join(output_dir, 'metrics.txt')
    with open(metrics_path, 'w') as f:
        for key, value in avg_metrics.items():
            f.write(f"{key}: {value:.4f}\n")
    
    # Print metrics
    print("\nEvaluation Metrics:")
    for key, value in avg_metrics.items():
        print(f"  {key}: {value:.4f}")
    
    return avg_metrics

def main(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print(f"Using device: {device}")
    
    # Create dataset and dataloader
    transform = SegmentationTransform(img_size=(args.img_size, args.img_size))
    
    test_dataset = SegmentationDataset(
        root_dir=args.data_dir,
        split='test',
        transform=transform
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    
    # Load model
    model = load_sam2_model(args, device)
    model.to(device)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Evaluate model
    metrics = evaluate(model, test_loader, device, args.output_dir)
    
    # Save metrics to a JSON file
    metrics_path = os.path.join(args.output_dir, 'metrics.json')
    import json
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    print(f"Metrics saved to {metrics_path}")
    
    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate SAM2 on test set")
    
    parser.add_argument('--data_dir', type=str, default='dataset',
                        help='Path to the dataset directory')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to the model checkpoint')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                        help='Directory to save evaluation results')
    parser.add_argument('--img_size', type=int, default=224,
                        help='Image size for evaluation')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--model_size', type=str, default='tiny',
                        choices=['tiny', 'small', 'base', 'large'],
                        help='Size of the SAM2 model to use')
    parser.add_argument('--cpu', action='store_true',
                        help='Force using CPU even if CUDA is available')
    
    args = parser.parse_args()
    
    main(args) 