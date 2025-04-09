#!/usr/bin/env python
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import argparse
from PIL import Image

# Import SAM2 image predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor

def run_image_inference(image_path, output_dir, model_id="facebook/sam2-hiera-tiny", device="cpu"):
    """
    Run SAM2 image predictor inference on a test image.
    
    Args:
        image_path (str): Path to the test image
        output_dir (str): Directory to save output masks and visualizations
        model_id (str): Model ID for SAM2 (e.g., facebook/sam2-hiera-tiny)
        device (str): Device to run inference on (cpu or cuda)
    """
    print(f"Running SAM2 image inference on {image_path}")
    print(f"Using model: {model_id} on device: {device}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the image
    image = Image.open(image_path).convert("RGB")
    
    # Initialize the predictor
    print("Loading the SAM2 image predictor...")
    predictor = SAM2ImagePredictor.from_pretrained(model_id, device=device)
    
    # Set the image
    print("Setting the image...")
    predictor.set_image(image)
    
    # Get image dimensions
    width, height = image.size
    
    # Define test points 
    # We'll use a grid of points to generate multiple predictions
    grid_size = 3
    x_points = np.linspace(width * 0.2, width * 0.8, grid_size)
    y_points = np.linspace(height * 0.2, height * 0.8, grid_size)
    
    all_points = []
    for x in x_points:
        for y in y_points:
            all_points.append([int(x), int(y)])
    
    np_image = np.array(image)
    
    # Create a figure to visualize the results
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.flatten()
    
    # Run inference with each point
    for i, (x, y) in enumerate(all_points[:9]):  # Limit to 9 points for display
        print(f"Running prediction for point ({x}, {y})...")
        
        # Set up point input
        point_coords = np.array([[x, y]])
        point_labels = np.array([1])  # 1 for foreground
        
        # Get masks from the predictor
        masks, scores, _ = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=True
        )
        
        # Find the mask with the highest score
        best_mask_idx = np.argmax(scores)
        best_mask = masks[best_mask_idx]
        best_score = scores[best_mask_idx]
        
        # Save the best mask
        mask_path = os.path.join(output_dir, f"mask_point_{i+1}.png")
        cv2.imwrite(mask_path, best_mask.astype(np.uint8) * 255)
        
        # Visualize the results
        if i < 9:  # Only show 9 results
            ax = axes[i]
            # Show the original image
            ax.imshow(np_image)
            
            # Plot the point
            ax.plot(x, y, 'ro', markersize=8)
            
            # Show the mask as an overlay
            colored_mask = np.zeros_like(np_image)
            colored_mask[best_mask > 0] = [0, 255, 0]  # Green overlay
            ax.imshow(colored_mask, alpha=0.5)
            
            ax.set_title(f"Point {i+1}, Score: {best_score:.2f}")
            ax.axis('off')
    
    # Adjust the layout and save the figure
    plt.tight_layout()
    visualization_path = os.path.join(output_dir, "sam2_image_predictions.png")
    plt.savefig(visualization_path)
    plt.close()
    
    print(f"Saved visualization to {visualization_path}")
    print(f"Saved all masks to {output_dir}")
    
    # Create a combined visualization
    combined_image = np.array(image)
    combined_mask = np.zeros((height, width, 4), dtype=np.uint8)
    
    # Use different colors for different predictions
    colors = [
        [255, 0, 0, 128],    # Red
        [0, 255, 0, 128],    # Green
        [0, 0, 255, 128],    # Blue
        [255, 255, 0, 128],  # Yellow
        [255, 0, 255, 128],  # Magenta
        [0, 255, 255, 128],  # Cyan
        [128, 0, 0, 128],    # Dark Red
        [0, 128, 0, 128],    # Dark Green
        [0, 0, 128, 128]     # Dark Blue
    ]
    
    for i, (x, y) in enumerate(all_points[:9]):
        # Load the mask
        mask_path = os.path.join(output_dir, f"mask_point_{i+1}.png")
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Add the colored mask
        color = colors[i % len(colors)]
        mask_colored = np.zeros((height, width, 4), dtype=np.uint8)
        mask_colored[mask > 0] = color
        
        # Blend with existing mask
        alpha = mask_colored[:, :, 3:4] / 255.0
        combined_mask = (1 - alpha) * combined_mask + alpha * mask_colored
    
    # Save the combined visualization
    plt.figure(figsize=(10, 10))
    plt.imshow(combined_image)
    plt.imshow(combined_mask.astype(np.uint8))
    
    # Plot all points
    for i, (x, y) in enumerate(all_points[:9]):
        plt.plot(x, y, 'o', color='white', markersize=10, markeredgecolor='black')
        plt.text(x+10, y+10, str(i+1), color='white', fontsize=12, 
                 bbox=dict(facecolor='black', alpha=0.7))
    
    plt.axis('off')
    plt.title("Combined SAM2 Predictions")
    
    combined_path = os.path.join(output_dir, "combined_predictions.png")
    plt.savefig(combined_path)
    plt.close()
    
    print(f"Saved combined visualization to {combined_path}")

def main():
    parser = argparse.ArgumentParser(description="Run SAM2 image predictor inference")
    parser.add_argument("--image", type=str, default="test_image.jpg", 
                        help="Path to the test image")
    parser.add_argument("--output", type=str, default="image_predictor_results",
                        help="Directory to save results")
    parser.add_argument("--model", type=str, default="facebook/sam2-hiera-tiny",
                        help="SAM2 model ID")
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["cpu", "cuda"], help="Device to run inference on")
    
    args = parser.parse_args()
    
    # Convert relative to absolute path if needed
    image_path = os.path.abspath(args.image)
    output_dir = os.path.abspath(args.output)
    
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        return 1
    
    # Run inference
    run_image_inference(image_path, output_dir, args.model, args.device)
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main()) 