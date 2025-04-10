#!/usr/bin/env python
import os
import argparse
import numpy as np
import cv2
from pathlib import Path
import torch
import torch.nn as nn
from tqdm import tqdm
import json
import matplotlib.pyplot as plt

def calculate_precision_recall_f1(pred_mask, gt_mask, threshold=0.5):
    """
    Calculate precision, recall, and F1 score.
    
    Args:
        pred_mask (np.ndarray): Predicted mask
        gt_mask (np.ndarray): Ground truth mask
        threshold (float): Threshold for binarizing predictions
        
    Returns:
        dict: Dictionary of metrics
    """
    # Convert to binary masks
    pred_binary = (pred_mask > threshold).astype(np.uint8)
    gt_binary = (gt_mask > threshold).astype(np.uint8)
    
    # Calculate TP, FP, FN
    tp = np.sum(np.logical_and(pred_binary == 1, gt_binary == 1))
    fp = np.sum(np.logical_and(pred_binary == 1, gt_binary == 0))
    fn = np.sum(np.logical_and(pred_binary == 0, gt_binary == 1))
    
    # Calculate metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Calculate IoU
    intersection = np.sum(np.logical_and(pred_binary, gt_binary))
    union = np.sum(np.logical_or(pred_binary, gt_binary))
    iou = intersection / union if union > 0 else 0
    
    # Calculate DICE coefficient
    dice = 2 * intersection / (np.sum(pred_binary) + np.sum(gt_binary)) if (np.sum(pred_binary) + np.sum(gt_binary)) > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'iou': iou,
        'dice': dice
    }

def evaluate_video(inference_video_path, gt_dataset_path, video_name, output_dir, threshold=0.5):
    """
    Evaluate video segmentation by extracting frames from inference video and comparing with ground truth.
    
    Args:
        inference_video_path (str): Path to the inference video with green mask overlay
        gt_dataset_path (str): Path to the dataset containing ground truth masks
        video_name (str): Name of the video to evaluate (like "video1", "video2")
        output_dir (str): Directory to save results
        threshold (float): Threshold for extracting mask from inference video
    
    Returns:
        dict: Dictionary of average metrics
    """
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    vis_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    
    # Open the inference video
    cap = cv2.VideoCapture(inference_video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open inference video: {inference_video_path}")
    
    # Get video properties
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Path to ground truth
    gt_split = 'test' if video_name.startswith('test_') else 'train'
    video_name_clean = video_name.replace('test_', '')
    gt_masks_dir = os.path.join(gt_dataset_path, gt_split, video_name_clean, 'masks')
    
    if not os.path.exists(gt_masks_dir):
        raise ValueError(f"Ground truth masks directory not found: {gt_masks_dir}")
    
    # Collect all metrics
    all_metrics = []
    metrics_per_frame = {}
    
    # Save a reference frame without any mask overlay
    # We'll use this to extract the mask by comparing each frame to the original
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, first_frame = cap.read()
    if not ret:
        raise ValueError("Cannot read first frame from video")
    
    # Debug directory to save intermediate processing steps
    debug_dir = os.path.join(output_dir, 'debug')
    os.makedirs(debug_dir, exist_ok=True)
    
    # Reset video capture
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    # Process each frame
    for frame_idx in tqdm(range(min(frame_count, 100)), desc=f"Evaluating {video_name}"):  # Limit to 100 frames
        ret, frame = cap.read()
        if not ret:
            break
        
        # Method 1: Extract green channel differences
        # Since our overlay is green (0,255,0), look at pixels where green is significantly 
        # higher than red and blue
        b, g, r = cv2.split(frame)
        
        # Find pixels where green is higher than red and blue by a threshold
        green_mask = np.logical_and(g > r + 50, g > b + 50)
        
        # Convert to binary mask
        pred_mask = green_mask.astype(np.uint8) * 255
        
        # Save debug images
        if frame_idx % 10 == 0:
            cv2.imwrite(os.path.join(debug_dir, f"frame_{frame_idx:04d}_orig.jpg"), frame)
            cv2.imwrite(os.path.join(debug_dir, f"frame_{frame_idx:04d}_mask.jpg"), pred_mask)
        
        # Find corresponding ground truth mask
        gt_mask_path = os.path.join(gt_masks_dir, f"{frame_idx:04d}.png")
        
        # Handle case where ground truth doesn't exist for this frame (e.g., test video has fewer frames)
        if not os.path.exists(gt_mask_path):
            print(f"Warning: Ground truth mask not found for frame {frame_idx}")
            continue
        
        # Read ground truth mask
        gt_mask = cv2.imread(gt_mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Resize if needed
        if gt_mask.shape != pred_mask.shape:
            gt_mask = cv2.resize(gt_mask, (pred_mask.shape[1], pred_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        # Save both masks for debugging
        if frame_idx % 10 == 0:
            cv2.imwrite(os.path.join(debug_dir, f"frame_{frame_idx:04d}_gt_mask.jpg"), gt_mask)
        
        # Calculate metrics
        metrics = calculate_precision_recall_f1(pred_mask, gt_mask, threshold)
        all_metrics.append(metrics)
        metrics_per_frame[str(frame_idx)] = metrics
        
        # Create visualization
        if frame_idx % 10 == 0:  # Save every 10th frame to avoid too many images
            vis_frame = frame.copy()
            
            # Create comparison image
            height, width = frame.shape[:2]
            vis_img = np.zeros((height, width * 2, 3), dtype=np.uint8)
            
            # Show pred_mask on left
            pred_color = np.zeros_like(frame)
            pred_color[pred_mask > 0] = [0, 255, 0]  # Green for prediction
            pred_overlay = cv2.addWeighted(frame, 1.0, pred_color, 0.5, 0)
            vis_img[:, :width] = pred_overlay
            
            # Show gt_mask on right
            gt_color = np.zeros_like(frame)
            gt_color[gt_mask > 0] = [0, 0, 255]  # Red for ground truth
            gt_overlay = cv2.addWeighted(frame, 1.0, gt_color, 0.5, 0)
            vis_img[:, width:] = gt_overlay
            
            # Add text
            cv2.putText(vis_img, f"IoU: {metrics['iou']:.4f}", (10, 30), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(vis_img, f"Dice: {metrics['dice']:.4f}", (10, 60), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.putText(vis_img, "Prediction", (width//4, height-20), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(vis_img, "Ground Truth", (width + width//4, height-20), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Save
            cv2.imwrite(os.path.join(vis_dir, f"{video_name}_frame_{frame_idx:04d}.jpg"), vis_img)
    
    cap.release()
    
    # Handle edge case where no metrics were calculated
    if not all_metrics:
        print(f"Warning: No metrics calculated for {video_name}")
        return {
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'iou': 0.0,
            'dice': 0.0
        }
    
    # Calculate average metrics
    avg_metrics = {}
    for key in all_metrics[0].keys():
        avg_metrics[key] = np.mean([m[key] for m in all_metrics])
    
    # Save metrics
    with open(os.path.join(output_dir, f"{video_name}_metrics.json"), 'w') as f:
        json.dump({
            'average': avg_metrics,
            'per_frame': metrics_per_frame
        }, f, indent=4)
    
    # Write text summary
    with open(os.path.join(output_dir, f"{video_name}_metrics.txt"), 'w') as f:
        for metric, value in avg_metrics.items():
            f.write(f"{metric}: {value:.4f}\n")
    
    # Create plots for metrics over time
    plt.figure(figsize=(12, 8))
    
    metrics_over_time = {}
    for key in all_metrics[0].keys():
        metrics_over_time[key] = [m[key] for m in all_metrics]
    
    for i, (metric, values) in enumerate(metrics_over_time.items()):
        plt.subplot(2, 3, i+1)
        plt.plot(values)
        plt.title(f"{metric.capitalize()} Over Time")
        plt.xlabel("Frame")
        plt.ylabel(metric.capitalize())
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{video_name}_metrics_plot.png"))
    
    return avg_metrics

def evaluate_all_videos(inference_results_dir, dataset_path, output_dir):
    """
    Evaluate all videos in the inference results directory.
    
    Args:
        inference_results_dir (str): Directory containing the inference results
        dataset_path (str): Path to the dataset containing ground truth masks
        output_dir (str): Directory to save results
    """
    # Look for videos to evaluate
    videos_to_evaluate = {}
    for item in os.listdir(inference_results_dir):
        if item.endswith('.mp4'):
            video_path = os.path.join(inference_results_dir, item)
            
            # Check if this is for a specific video
            if 'video1' in item or 'video2' in item:
                video_name = 'video1' if 'video1' in item else 'video2'
            else:
                # Default to video1 for generic segmentation.mp4
                video_name = 'video1'
            
            videos_to_evaluate[video_name] = video_path
    
    if not videos_to_evaluate:
        print(f"No inference videos found in {inference_results_dir}")
        return
    
    # Evaluate each video
    all_metrics = {}
    for video_name, video_path in videos_to_evaluate.items():
        print(f"Evaluating {video_name} from {video_path}")
        try:
            metrics = evaluate_video(
                inference_video_path=video_path,
                gt_dataset_path=dataset_path,
                video_name=video_name,
                output_dir=os.path.join(output_dir, video_name),
                threshold=0.5
            )
            all_metrics[video_name] = metrics
        except Exception as e:
            print(f"Error evaluating {video_name}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Save overall metrics
    if all_metrics:
        # Calculate overall average
        overall_metrics = {}
        for metric in all_metrics[list(all_metrics.keys())[0]].keys():
            overall_metrics[metric] = np.mean([m[metric] for m in all_metrics.values()])
        
        with open(os.path.join(output_dir, "metrics.txt"), 'w') as f:
            for metric, value in overall_metrics.items():
                f.write(f"{metric}: {value:.4f}\n")
        
        with open(os.path.join(output_dir, "metrics.json"), 'w') as f:
            json.dump({
                'overall': overall_metrics,
                'per_video': all_metrics
            }, f, indent=4)
        
        # Create a simple summary of results
        with open(os.path.join(output_dir, "summary.txt"), 'w') as f:
            f.write("\nSummary of Metrics:\n\n")
            for video_name, metrics in all_metrics.items():
                f.write(f"{video_name} IoU: {metrics['iou']:.4f}\n")
                f.write(f"{video_name} Dice: {metrics['dice']:.4f}\n\n")
    
    return all_metrics

def main():
    parser = argparse.ArgumentParser(description="Evaluate segmentation results from direct model")
    parser.add_argument("--inference-dir", type=str, default="inference_results/direct",
                      help="Directory containing the inference results")
    parser.add_argument("--dataset-path", type=str, default="dataset",
                      help="Path to the dataset containing ground truth masks")
    parser.add_argument("--output-dir", type=str, default="inference_results/evaluation",
                      help="Directory to save evaluation results")
    
    args = parser.parse_args()
    
    # Make sure the output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run evaluation
    evaluate_all_videos(
        inference_results_dir=args.inference_dir,
        dataset_path=args.dataset_path,
        output_dir=args.output_dir
    )

if __name__ == "__main__":
    main() 