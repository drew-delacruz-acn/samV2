import os
import cv2
import numpy as np
import argparse
from pathlib import Path
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

def calculate_metrics(pred_mask, gt_mask, threshold=0.5):
    """
    Calculate evaluation metrics between a prediction and ground truth mask.
    
    Args:
        pred_mask (ndarray): Predicted mask (uint8)
        gt_mask (ndarray): Ground truth mask (uint8)
        threshold (float): Threshold for binarizing masks (if not already binary)
        
    Returns:
        dict: Dictionary of metrics
    """
    # Ensure binary masks
    pred_binary = (pred_mask > threshold * 255).astype(np.uint8)
    gt_binary = (gt_mask > threshold * 255).astype(np.uint8)
    
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

def evaluate_masks(pred_dir, gt_dir, output_path=None):
    """
    Evaluate all masks in a directory against ground truth.
    
    Args:
        pred_dir (str): Directory containing predicted masks
        gt_dir (str): Directory containing ground truth masks
        output_path (str): Path to save metrics summary (optional)
        
    Returns:
        dict: Dictionary of averaged metrics
    """
    # Get all mask files
    pred_files = sorted([f for f in os.listdir(pred_dir) if f.endswith('.png')])
    
    all_metrics = []
    
    for pred_file in pred_files:
        # Get ground truth filename (might need to adjust depending on naming conventions)
        frame_idx = int(pred_file.split('_')[-1].split('.')[0])
        gt_file = f"{frame_idx:04d}.png"
        
        # Load masks
        pred_path = os.path.join(pred_dir, pred_file)
        gt_path = os.path.join(gt_dir, gt_file)
        
        if not os.path.exists(gt_path):
            print(f"Warning: Ground truth mask not found for {pred_file}")
            continue
        
        pred_mask = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
        gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        
        # Calculate metrics
        metrics = calculate_metrics(pred_mask, gt_mask)
        all_metrics.append(metrics)
        
        print(f"Metrics for {pred_file}:")
        print(f"  IoU: {metrics['iou']:.4f}")
        print(f"  Dice: {metrics['dice']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1: {metrics['f1']:.4f}")
    
    # Calculate average metrics
    avg_metrics = {}
    for key in all_metrics[0].keys():
        avg_metrics[key] = np.mean([m[key] for m in all_metrics])
    
    print("\nAverage Metrics:")
    print(f"  IoU: {avg_metrics['iou']:.4f}")
    print(f"  Dice: {avg_metrics['dice']:.4f}")
    print(f"  Precision: {avg_metrics['precision']:.4f}")
    print(f"  Recall: {avg_metrics['recall']:.4f}")
    print(f"  F1: {avg_metrics['f1']:.4f}")
    
    # Save metrics to file if output_path is provided
    if output_path:
        with open(output_path, 'w') as f:
            f.write("Per-frame metrics:\n")
            for i, metrics in enumerate(all_metrics):
                f.write(f"Frame {i}:\n")
                for key, value in metrics.items():
                    f.write(f"  {key}: {value:.4f}\n")
            
            f.write("\nAverage metrics:\n")
            for key, value in avg_metrics.items():
                f.write(f"  {key}: {value:.4f}\n")
        
        print(f"Metrics saved to {output_path}")
        
        # Create a bar chart of average metrics
        plt.figure(figsize=(10, 6))
        plt.bar(avg_metrics.keys(), avg_metrics.values())
        plt.title('Average Metrics')
        plt.ylim(0, 1)
        plt.savefig(os.path.join(os.path.dirname(output_path), 'metrics_chart.png'))
        plt.close()
    
    return avg_metrics

def main():
    parser = argparse.ArgumentParser(description="Calculate metrics between predicted and ground truth masks")
    
    parser.add_argument('--pred_dir', type=str, required=True,
                        help='Directory containing predicted masks')
    parser.add_argument('--gt_dir', type=str, required=True,
                        help='Directory containing ground truth masks')
    parser.add_argument('--output_path', type=str, default=None,
                        help='Path to save metrics summary (optional)')
    
    args = parser.parse_args()
    
    # Make sure directories exist
    pred_dir = Path(args.pred_dir)
    gt_dir = Path(args.gt_dir)
    
    if not pred_dir.exists():
        print(f"Prediction directory not found: {pred_dir}")
        return
    
    if not gt_dir.exists():
        print(f"Ground truth directory not found: {gt_dir}")
        return
    
    # Evaluate masks
    evaluate_masks(str(pred_dir), str(gt_dir), args.output_path)

if __name__ == "__main__":
    main() 