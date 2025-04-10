#!/usr/bin/env python
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import time
import datetime

# Import our dataset
from dataset import SegmentationDataset, SegmentationTransform

def dice_loss(pred, target):
    """
    Compute the DICE loss between predictions and targets.
    
    Args:
        pred (torch.Tensor): Predicted masks (B, 1, H, W)
        target (torch.Tensor): Ground truth masks (B, 1, H, W)
        
    Returns:
        torch.Tensor: DICE loss
    """
    smooth = 1.0
    
    # Flatten the tensors
    pred_flat = pred.view(pred.size(0), -1)
    target_flat = target.view(target.size(0), -1)
    
    intersection = (pred_flat * target_flat).sum(1)
    union = pred_flat.sum(1) + target_flat.sum(1)
    
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1.0 - dice.mean()


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
        
        # Ensure masks have the right shape [B, H, W] -> [B, 1, H, W]
        if masks.dim() == 3:
            masks = masks.unsqueeze(1)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
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
            
            # Ensure masks have the right shape [B, H, W] -> [B, 1, H, W]
            if masks.dim() == 3:
                masks = masks.unsqueeze(1)
            
            # Forward pass
            outputs = model(images)
            
            # Calculate loss
            loss = criterion(outputs, masks)
            
            running_loss += loss.item()
    
    val_loss = running_loss / len(dataloader)
    return val_loss


def create_model(img_size=640):
    """
    Create a simple U-Net style segmentation model.
    
    Returns:
        model: Segmentation model
    """
    # A simplified U-Net style architecture
    model = nn.Sequential(
        # Encoder
        nn.Conv2d(3, 32, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),
        
        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),
        
        nn.Conv2d(64, 128, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),
        
        # Bottleneck
        nn.Conv2d(128, 256, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        
        # Decoder
        nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
        nn.ReLU(inplace=True),
        
        nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
        nn.ReLU(inplace=True),
        
        nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
        nn.ReLU(inplace=True),
        
        # Output layer
        nn.Conv2d(32, 1, kernel_size=1),
        nn.Sigmoid()  # Output probabilities between 0 and 1
    )
    
    print(f"Created a model with input size: {img_size}x{img_size}")
    
    return model


def main(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create dataset and dataloaders
    img_size = args.img_size
    transform = SegmentationTransform(img_size=(img_size, img_size))
    
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
    
    # Create model
    model = create_model(img_size=img_size).to(device)
    
    # Define optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Use combined loss (BCE + Dice loss) for better segmentation results
    criterion = lambda pred, target: nn.BCELoss()(pred, target) + dice_loss(pred, target)
    
    # Training loop
    best_val_loss = float('inf')
    start_time = time.time()
    
    print(f"Starting training for {args.num_epochs} epochs...")
    
    for epoch in range(args.num_epochs):
        print(f"Epoch {epoch+1}/{args.num_epochs}")
        
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        
        # Validate
        val_loss = validate(model, val_loader, criterion, device)
        
        # Calculate elapsed time
        elapsed = time.time() - start_time
        elapsed_str = str(datetime.timedelta(seconds=int(elapsed)))
        
        print(f"Time elapsed: {elapsed_str}")
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
    
    # Save final model
    final_checkpoint_path = os.path.join(args.output_dir, 'final_model.pth')
    torch.save({
        'epoch': args.num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
    }, final_checkpoint_path)
    
    print(f"Saved final model checkpoint to {final_checkpoint_path}")
    print("Training completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a segmentation model on dummy dataset")
    
    parser.add_argument('--data_dir', type=str, default='dataset', 
                        help='Path to the dataset directory')
    parser.add_argument('--output_dir', type=str, default='models/direct',
                        help='Directory to save model checkpoints')
    parser.add_argument('--img_size', type=int, default=640,
                        help='Image size for training')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for data loading')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=20,
                        help='Number of epochs to train for')
    
    args = parser.parse_args()
    
    # Check if the dataset directory exists
    if not os.path.exists(args.data_dir):
        raise ValueError(f"Dataset directory {args.data_dir} does not exist")
    
    # Start training
    main(args) 