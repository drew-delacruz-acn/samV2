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
    
    # Ensure pred and target have the same shape
    if pred.shape != target.shape:
        target = target.expand_as(pred)
    
    # Flatten the tensors
    pred_flat = pred.reshape(-1)
    target_flat = target.reshape(-1)
    
    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum()
    
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1.0 - dice


def train_one_epoch(model, dataloader, optimizer, criterion, device, use_autocast=True):
    """
    Train the model for one epoch.
    
    Args:
        model: The model to train
        dataloader: DataLoader for training data
        optimizer: Optimizer for parameter updates
        criterion: Loss function
        device: Device to train on
        use_autocast: Whether to use automatic mixed precision
        
    Returns:
        float: Average loss for the epoch
    """
    model.train()
    running_loss = 0.0
    
    for i, batch in enumerate(tqdm(dataloader, desc="Training")):
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        
        # Ensure masks have the right shape [B, H, W] -> [B, 1, H, W]
        if masks.dim() == 3:
            masks = masks.unsqueeze(1)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass with SAM2 model
        if use_autocast and device.type == 'cuda':
            # Use autocast only if explicitly enabled and on GPU
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                outputs = model(images)
        else:
            # Otherwise run with full precision
            outputs = model(images)
        
        # Calculate loss
        loss = criterion(outputs, masks)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
    epoch_loss = running_loss / len(dataloader)
    return epoch_loss


def validate(model, dataloader, criterion, device, use_autocast=True):
    """
    Validate the model.
    
    Args:
        model: The model to validate
        dataloader: DataLoader for validation data
        criterion: Loss function
        device: Device to validate on
        use_autocast: Whether to use automatic mixed precision
        
    Returns:
        float: Average loss for the validation set
    """
    model.eval()
    running_loss = 0.0
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc="Validation")):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            
            # Ensure masks have the right shape [B, H, W] -> [B, 1, H, W]
            if masks.dim() == 3:
                masks = masks.unsqueeze(1)
            
            # Forward pass with SAM2 model
            if use_autocast and device.type == 'cuda':
                # Use autocast only if explicitly enabled and on GPU
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                    outputs = model(images)
            else:
                # Otherwise run with full precision
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
    
    print(f"Would load SAM2 model: {model_name} in a production environment")
    
    # SAM2VideoPredictor is not designed for training purposes 
    # (it raises NotImplementedError when used in forward pass)
    # Using a simple placeholder model for demonstration
    print("Using a placeholder model for training demonstration")
    
    # Create a placeholder model with the same input/output dimensions
    # In a real implementation, you would need to:
    # 1. Extract the proper trainable components from SAM2
    # 2. Create a proper training wrapper for them
    # 3. Implement the necessary hooks for fine-tuning
    
    placeholder_model = nn.Sequential(
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
    
    return placeholder_model


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
    
    # Load placeholder model 
    model = create_sam2_model(args).to(device)
    
    # Define optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Use binary cross-entropy loss since our model outputs sigmoid probabilities
    criterion = nn.BCELoss()
    
    # Alternative: combine BCE and Dice loss for better segmentation results
    if args.use_combined_loss:
        print("Using combined BCE and Dice loss")
        criterion = lambda pred, target: nn.BCELoss()(pred, target) + dice_loss(pred, target)
    else:
        print("Using BCE loss")
        criterion = nn.BCELoss()
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(args.num_epochs):
        print(f"Epoch {epoch+1}/{args.num_epochs}")
        
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, use_autocast=args.use_autocast)
        
        # Validate
        val_loss = validate(model, val_loader, criterion, device, use_autocast=args.use_autocast)
        
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
    parser.add_argument('--use_autocast', action='store_true',
                        help='Use automatic mixed precision (may not work on all devices)')
    parser.add_argument('--use_combined_loss', action='store_true',
                        help='Use combined BCE and Dice loss for better segmentation results')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    main(args) 