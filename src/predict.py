#!/usr/bin/env python
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import argparse
from PIL import Image
from torchvision import transforms

# Define the same model architecture that was used for training
# This must match the architecture used during training
def create_model():
    """
    Create the same model architecture that was used during training.
    """
    model = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(64, 128, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(128, 256, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(256, 128, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
        nn.ReLU(inplace=True),
        nn.Conv2d(32, 1, kernel_size=1),
        nn.Sigmoid()  # Output probabilities between 0 and 1
    )
    return model

def preprocess_image(image_path, target_size=(224, 224)):
    """
    Load and preprocess an image for prediction.
    
    Args:
        image_path (str): Path to the image file
        target_size (tuple): Target size for resizing
    
    Returns:
        torch.Tensor: Preprocessed image tensor
    """
    # Load the image
    image = Image.open(image_path).convert("RGB")
    
    # Resize the image
    image = image.resize(target_size)
    
    # Convert to tensor and normalize
    transform = transforms.Compose([
        transforms.ToTensor(),
        # Add normalization if used during training
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    image_tensor = transform(image)
    
    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0)
    
    return image_tensor, np.array(image)

def run_prediction(image_path, model_path, output_dir, threshold=0.5, device="cpu"):
    """
    Run prediction on a single image using the trained model.
    
    Args:
        image_path (str): Path to the input image
        model_path (str): Path to the trained model checkpoint
        output_dir (str): Directory to save the prediction results
        threshold (float): Threshold for converting probabilities to binary mask
        device (str): Device to run prediction on (cpu or cuda)
    """
    print(f"Running prediction on {image_path}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set device
    device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
    print(f"Using device: {device}")
    
    # Load the model
    model = create_model()
    
    # Load the trained weights
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model from {model_path} (trained for {checkpoint['epoch']+1} epochs)")
    
    # Set model to evaluation mode
    model.to(device)
    model.eval()
    
    # Preprocess the image
    image_tensor, original_image = preprocess_image(image_path)
    image_tensor = image_tensor.to(device)
    
    # Run prediction
    with torch.no_grad():
        output = model(image_tensor)
    
    # Convert to numpy array
    pred_mask = output.cpu().squeeze().numpy()
    
    # Apply threshold to get binary mask
    binary_mask = (pred_mask > threshold).astype(np.uint8) * 255
    
    # Save the binary mask
    mask_path = os.path.join(output_dir, os.path.basename(image_path).replace(".", "_mask."))
    cv2.imwrite(mask_path, binary_mask)
    print(f"Saved mask to {mask_path}")
    
    # Create visualization
    plt.figure(figsize=(12, 6))
    
    # Original image
    plt.subplot(1, 3, 1)
    plt.imshow(original_image)
    plt.title("Original Image")
    plt.axis("off")
    
    # Predicted probability
    plt.subplot(1, 3, 2)
    plt.imshow(pred_mask, cmap="jet")
    plt.title("Prediction Probability")
    plt.colorbar()
    plt.axis("off")
    
    # Binary mask
    plt.subplot(1, 3, 3)
    plt.imshow(binary_mask, cmap="gray")
    plt.title(f"Binary Mask (threshold={threshold})")
    plt.axis("off")
    
    # Save the visualization
    viz_path = os.path.join(output_dir, os.path.basename(image_path).replace(".", "_viz."))
    plt.tight_layout()
    plt.savefig(viz_path)
    plt.close()
    print(f"Saved visualization to {viz_path}")
    
    # Create overlay visualization
    overlay_img = original_image.copy()
    colored_mask = np.zeros_like(overlay_img)
    colored_mask[..., 1] = binary_mask  # Green channel
    
    plt.figure(figsize=(8, 8))
    plt.imshow(original_image)
    plt.imshow(colored_mask, alpha=0.5)
    plt.title("Segmentation Overlay")
    plt.axis("off")
    
    # Save the overlay
    overlay_path = os.path.join(output_dir, os.path.basename(image_path).replace(".", "_overlay."))
    plt.savefig(overlay_path)
    plt.close()
    print(f"Saved overlay to {overlay_path}")
    
    return binary_mask

def main():
    parser = argparse.ArgumentParser(description="Run prediction using trained segmentation model")
    parser.add_argument("--image", type=str, required=True, 
                        help="Path to the input image")
    parser.add_argument("--model", type=str, default="models/best_model.pth",
                        help="Path to the trained model checkpoint")
    parser.add_argument("--output", type=str, default="predictions",
                        help="Directory to save prediction results")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Threshold for binary mask (0-1)")
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["cpu", "cuda"], help="Device to run prediction on")
    
    args = parser.parse_args()
    
    # Convert relative to absolute path if needed
    image_path = os.path.abspath(args.image)
    model_path = os.path.abspath(args.model)
    output_dir = os.path.abspath(args.output)
    
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        return 1
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return 1
    
    # Run prediction
    run_prediction(
        image_path=image_path,
        model_path=model_path,
        output_dir=output_dir,
        threshold=args.threshold,
        device=args.device
    )
    
    return 0

def predict_batch(image_dir, model_path, output_dir, threshold=0.5, device="cpu"):
    """
    Run prediction on all images in a directory.
    
    Args:
        image_dir (str): Directory containing input images
        model_path (str): Path to the trained model checkpoint
        output_dir (str): Directory to save prediction results
        threshold (float): Threshold for converting probabilities to binary mask
        device (str): Device to run prediction on (cpu or cuda)
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = [f for f in os.listdir(image_dir) if any(f.lower().endswith(ext) for ext in image_extensions)]
    
    if not image_files:
        print(f"No images found in {image_dir}")
        return
    
    print(f"Found {len(image_files)} images to process")
    
    # Set device
    device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
    print(f"Using device: {device}")
    
    # Load the model (do this just once)
    model = create_model()
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model from {model_path} (trained for {checkpoint['epoch']+1} epochs)")
    
    # Set model to evaluation mode
    model.to(device)
    model.eval()
    
    # Process each image
    for i, image_file in enumerate(image_files):
        image_path = os.path.join(image_dir, image_file)
        print(f"[{i+1}/{len(image_files)}] Processing {image_file}...")
        
        # Preprocess the image
        image_tensor, original_image = preprocess_image(image_path)
        image_tensor = image_tensor.to(device)
        
        # Run prediction
        with torch.no_grad():
            output = model(image_tensor)
        
        # Convert to numpy array
        pred_mask = output.cpu().squeeze().numpy()
        
        # Apply threshold to get binary mask
        binary_mask = (pred_mask > threshold).astype(np.uint8) * 255
        
        # Save the binary mask
        mask_path = os.path.join(output_dir, os.path.basename(image_path).replace(".", "_mask."))
        cv2.imwrite(mask_path, binary_mask)
        
        # Create overlay visualization
        overlay_img = original_image.copy()
        colored_mask = np.zeros_like(overlay_img)
        colored_mask[..., 1] = binary_mask  # Green channel
        
        plt.figure(figsize=(8, 8))
        plt.imshow(original_image)
        plt.imshow(colored_mask, alpha=0.5)
        plt.title(f"Segmentation: {image_file}")
        plt.axis("off")
        
        # Save the overlay
        overlay_path = os.path.join(output_dir, os.path.basename(image_path).replace(".", "_overlay."))
        plt.savefig(overlay_path)
        plt.close()
    
    print(f"Processed {len(image_files)} images. Results saved to {output_dir}")

if __name__ == "__main__":
    import sys
    sys.exit(main()) 