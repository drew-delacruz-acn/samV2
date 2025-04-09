import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms


class SegmentationDataset(Dataset):
    """
    Dataset class for loading image and mask pairs from the synthetic dataset.
    """
    def __init__(self, root_dir, split='train', transform=None):
        """
        Args:
            root_dir (str): Root directory of the dataset.
            split (str): 'train' or 'test' split.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        
        self.videos = []
        self.image_paths = []
        self.mask_paths = []
        
        # Get all video directories in the split
        split_dir = os.path.join(root_dir, split)
        self.videos = sorted([d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))])
        
        # Collect all image and mask pairs
        for video in self.videos:
            video_dir = os.path.join(split_dir, video)
            images_dir = os.path.join(video_dir, 'images')
            masks_dir = os.path.join(video_dir, 'masks')
            
            image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.jpg')])
            
            for img_file in image_files:
                img_path = os.path.join(images_dir, img_file)
                mask_path = os.path.join(masks_dir, img_file.replace('.jpg', '.png'))
                
                if os.path.exists(mask_path):
                    self.image_paths.append(img_path)
                    self.mask_paths.append(mask_path)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        
        # Read image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Read mask (binary)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Convert mask to binary (0 or 1)
        mask = (mask > 0).astype(np.float32)
        
        sample = {'image': image, 'mask': mask}
        
        if self.transform:
            sample = self.transform(sample)
        else:
            # Default transformations
            sample['image'] = transforms.ToTensor()(image)
            sample['mask'] = torch.from_numpy(mask)
        
        return sample


class SegmentationTransform:
    """
    Transformations for the segmentation dataset.
    """
    def __init__(self, img_size=(224, 224)):
        self.img_size = img_size
    
    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        
        # Resize
        image = cv2.resize(image, self.img_size)
        mask = cv2.resize(mask, self.img_size, interpolation=cv2.INTER_NEAREST)
        
        # Convert to PyTorch tensors
        image = transforms.ToTensor()(image)
        mask = torch.from_numpy(mask).float()
        
        # Add normalization if needed for SAM2
        # image = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(image)
        
        return {'image': image, 'mask': mask} 