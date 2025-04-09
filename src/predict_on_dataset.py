#!/usr/bin/env python
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import argparse
from tqdm import tqdm

# Import the prediction functions from predict.py
from predict import create_model, preprocess_image

def evaluate_on_test_set(model_path, dataset_path, output_dir, threshold=0.5, device="cpu"):
    """
    Evaluate the trained model on the test dataset and compute metrics.
    
    Args:
        model_path (str): Path to the trained model checkpoint
        dataset_path (str): Path to the test dataset directory
        output_dir (str): Directory to save prediction results
        threshold (float): Threshold for binary mask conversion
        device (str): Device to run evaluation on
    """
    # Set device
    device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the model
    model = create_model()
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model from {model_path} (epoch {checkpoint['epoch']+1})")
    
    # Set model to evaluation mode
    model.to(device)
    model.eval()
    
    # Get list of test videos
    test_dir = os.path.join(dataset_path, 'test')
    test_videos = [d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))]
    
    if not test_videos:
        print(f"No test videos found in {test_dir}")
        return
    
    print(f"Found {len(test_videos)} test videos")
    
    # Overall metrics
    total_iou = 0.0
    total_accuracy = 0.0
    total_frames = 0
    
    # Process each test video
    for video_name in test_videos:
        video_dir = os.path.join(test_dir, video_name)
        images_dir = os.path.join(video_dir, 'images')
        masks_dir = os.path.join(video_dir, 'masks')
        
        # Create output directory for this video
        video_output_dir = os.path.join(output_dir, video_name)
        os.makedirs(video_output_dir, exist_ok=True)
        
        # Get all image files
        image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.jpg')])
        
        if not image_files:
            print(f"No image files found in {images_dir}")
            continue
        
        print(f"Processing {len(image_files)} frames from {video_name}...")
        
        # Video metrics
        video_iou = 0.0
        video_accuracy = 0.0
        
        # Process each frame
        for image_file in tqdm(image_files, desc=f"Video {video_name}"):
            image_path = os.path.join(images_dir, image_file)
            mask_path = os.path.join(masks_dir, image_file.replace('.jpg', '.png'))
            
            if not os.path.exists(mask_path):
                print(f"Warning: Mask not found for {image_file}")
                continue
            
            # Preprocess the image
            image_tensor, original_image = preprocess_image(image_path)
            image_tensor = image_tensor.to(device)
            
            # Run prediction
            with torch.no_grad():
                output = model(image_tensor)
            
            # Convert prediction to numpy array
            pred_mask = output.cpu().squeeze().numpy()
            
            # Apply threshold to get binary mask
            binary_pred = (pred_mask > threshold).astype(np.uint8)
            
            # Load ground truth mask
            gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            gt_mask = cv2.resize(gt_mask, (pred_mask.shape[1], pred_mask.shape[0]))
            gt_mask = (gt_mask > 0).astype(np.uint8)  # Convert to binary
            
            # Calculate IoU
            intersection = np.logical_and(binary_pred, gt_mask).sum()
            union = np.logical_or(binary_pred, gt_mask).sum()
            iou = intersection / (union + 1e-10)  # Add small epsilon to avoid division by zero
            
            # Calculate accuracy
            accuracy = (binary_pred == gt_mask).mean()
            
            # Accumulate metrics
            video_iou += iou
            video_accuracy += accuracy
            
            # Save prediction
            output_mask_path = os.path.join(video_output_dir, f"{os.path.splitext(image_file)[0]}_pred.png")
            cv2.imwrite(output_mask_path, binary_pred * 255)
            
            # Create visualization (every 10th frame to save space)
            if int(os.path.splitext(image_file)[0]) % 10 == 0:
                # Create overlay visualization
                plt.figure(figsize=(12, 4))
                
                # Original image
                plt.subplot(1, 3, 1)
                plt.imshow(original_image)
                plt.title("Original")
                plt.axis("off")
                
                # Ground truth mask
                plt.subplot(1, 3, 2)
                plt.imshow(gt_mask, cmap="gray")
                plt.title("Ground Truth")
                plt.axis("off")
                
                # Prediction
                plt.subplot(1, 3, 3)
                plt.imshow(binary_pred, cmap="gray")
                plt.title(f"Prediction (IoU: {iou:.3f})")
                plt.axis("off")
                
                # Save visualization
                viz_path = os.path.join(video_output_dir, f"{os.path.splitext(image_file)[0]}_viz.png")
                plt.tight_layout()
                plt.savefig(viz_path)
                plt.close()
        
        # Calculate average video metrics
        frames_count = len(image_files)
        avg_video_iou = video_iou / frames_count
        avg_video_accuracy = video_accuracy / frames_count
        
        print(f"Video {video_name} metrics:")
        print(f"  Average IoU: {avg_video_iou:.4f}")
        print(f"  Average Accuracy: {avg_video_accuracy:.4f}")
        
        # Save video metrics
        with open(os.path.join(video_output_dir, "metrics.txt"), "w") as f:
            f.write(f"Average IoU: {avg_video_iou:.4f}\n")
            f.write(f"Average Accuracy: {avg_video_accuracy:.4f}\n")
        
        # Accumulate overall metrics
        total_iou += video_iou
        total_accuracy += video_accuracy
        total_frames += frames_count
    
    # Calculate overall average metrics
    avg_iou = total_iou / total_frames
    avg_accuracy = total_accuracy / total_frames
    
    print("\nOverall metrics:")
    print(f"  Average IoU: {avg_iou:.4f}")
    print(f"  Average Accuracy: {avg_accuracy:.4f}")
    
    # Save overall metrics
    with open(os.path.join(output_dir, "overall_metrics.txt"), "w") as f:
        f.write(f"Total frames: {total_frames}\n")
        f.write(f"Average IoU: {avg_iou:.4f}\n")
        f.write(f"Average Accuracy: {avg_accuracy:.4f}\n")
    
    print(f"Results saved to {output_dir}")
    
    return avg_iou, avg_accuracy

def main():
    parser = argparse.ArgumentParser(description="Evaluate segmentation model on test dataset")
    parser.add_argument("--model", type=str, default="models/best_model.pth",
                        help="Path to the trained model checkpoint")
    parser.add_argument("--dataset", type=str, default="dataset",
                        help="Path to the dataset directory")
    parser.add_argument("--output", type=str, default="evaluations",
                        help="Directory to save evaluation results")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Threshold for binary mask (0-1)")
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["cpu", "cuda"], help="Device to run evaluation on")
    
    args = parser.parse_args()
    
    # Convert relative to absolute path if needed
    model_path = os.path.abspath(args.model)
    dataset_path = os.path.abspath(args.dataset)
    output_dir = os.path.abspath(args.output)
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return 1
    
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset directory not found at {dataset_path}")
        return 1
    
    # Run evaluation
    evaluate_on_test_set(
        model_path=model_path,
        dataset_path=dataset_path,
        output_dir=output_dir,
        threshold=args.threshold,
        device=args.device
    )
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main()) 