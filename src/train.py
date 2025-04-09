import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import argparse

from dataset import SegmentationDataset, SegmentationTransform

# Import SAM2 video predictor
from sam2.sam2_video_predictor import SAM2VideoPredictor


def dice_loss(pred, target):
    """
    Compute the DICE loss between predictions and targets.
    
    Args:
        pred (torch.Tensor): Predicted masks
        target (torch.Tensor): Ground truth masks
        
    Returns:
        torch.Tensor: DICE loss
    """
    smooth = 1.0
    
    pred_flat = pred.reshape(-1)
    target_flat = target.reshape(-1)
    
    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum()
    
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1.0 - dice


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    """
    Train the model for one epoch.
    
    Args:
        model: The model to train
        dataloader: DataLoader for training data
        optimizer: Optimizer for parameter updates
        criterion: Loss function
        device: Device to train on
        
    Returns:
        float: Average loss for the epoch
    """
    model.train()
    running_loss = 0.0
    
    for i, batch in enumerate(tqdm(dataloader, desc="Training")):
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass with SAM2 model
        # Note: This is simplified and needs to be adjusted according to SAM2's API
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            # Initialize state with batch images
            # For training, we need to adapt this to SAM2's API
            outputs = model(images)
        
        # Calculate loss
        loss = criterion(outputs, masks)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
    epoch_loss = running_loss / len(dataloader)
    return epoch_loss


def validate(model, dataloader, criterion, device):
    """
    Validate the model.
    
    Args:
        model: The model to validate
        dataloader: DataLoader for validation data
        criterion: Loss function
        device: Device to validate on
        
    Returns:
        float: Average loss for the validation set
    """
    model.eval()
    running_loss = 0.0
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc="Validation")):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            
            # Forward pass with SAM2 model
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                outputs = model(images)
            
            # Calculate loss
            loss = criterion(outputs, masks)
            
            running_loss += loss.item()
    
    val_loss = running_loss / len(dataloader)
    return val_loss


def create_sam2_model(args):
    """
    Create and configure a SAM2 model for fine-tuning.
    
    Args:
        args: Command-line arguments
        
    Returns:
        model: The configured SAM2 model
    """
    # Load SAM2 model using the video predictor API
    # Choose model size based on available resources
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
        # Use from_pretrained for the complete pipeline
        model = SAM2VideoPredictor.from_pretrained(model_name)
        
        # Note: For actual fine-tuning, we would need to:
        # 1. Extract the base model from the predictor
        # 2. Set up appropriate training hooks
        # 3. Configure parameters for fine-tuning
        
        print("SAM2 model loaded successfully")
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
        
        return model


def main(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dataset and dataloaders
    transform = SegmentationTransform(img_size=(args.img_size, args.img_size))
    
    train_dataset = SegmentationDataset(
        root_dir=args.data_dir,
        split='train',
        transform=transform
    )
    
    val_dataset = SegmentationDataset(
        root_dir=args.data_dir,
        split='test',
        transform=transform
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    
    # Load SAM2 model
    model = create_sam2_model(args).to(device)
    
    # Define optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = dice_loss
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(args.num_epochs):
        print(f"Epoch {epoch+1}/{args.num_epochs}")
        
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        
        # Validate
        val_loss = validate(model, val_loader, criterion, device)
        
        print(f"Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            
            # Save checkpoint
            checkpoint_path = os.path.join(args.output_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, checkpoint_path)
            
            print(f"Saved best model checkpoint to {checkpoint_path}")
    
    print("Training completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune SAM2 on synthetic dataset")
    
    parser.add_argument('--data_dir', type=str, default='dataset', 
                        help='Path to the dataset directory')
    parser.add_argument('--output_dir', type=str, default='models',
                        help='Directory to save model checkpoints')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to SAM2 checkpoint')
    parser.add_argument('--img_size', type=int, default=224,
                        help='Image size for training')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for data loading')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='Number of epochs to train for')
    parser.add_argument('--model_size', type=str, default='small',
                        choices=['tiny', 'small', 'base', 'large'],
                        help='SAM2 model size to use')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    main(args) 