import os
import shutil
from tqdm import tqdm

def main():
    """
    Reorganize the existing dataset to match the DAVIS-style format required by SAM2 training.
    
    Current structure:
    dataset/
    ├── train/
    │   ├── video1/
    │   │   ├── images/
    │   │   │   ├── 0000.jpg
    │   │   │   └── ...
    │   │   └── masks/
    │   │       ├── 0000.png
    │   │       └── ...
    │   ├── video2/
    │   │   └── ...
    │   └── video3/
    │       └── ...
    └── test/
        ├── video1/
        │   └── ...
        └── video2/
            └── ...
    
    Target structure (DAVIS-style):
    sam2_dataset/
    ├── JPEGImages/
    │   ├── video1/
    │   │   ├── 00000.jpg
    │   │   └── ...
    │   ├── video2/
    │   │   └── ...
    │   └── ...
    └── Annotations/
        ├── video1/
        │   ├── 00000.png
        │   └── ...
        ├── video2/
        │   └── ...
        └── ...
    """
    src_dir = os.path.join(os.getcwd(), 'dataset')
    target_dir = os.path.join(os.getcwd(), 'sam2_dataset')
    
    # Create target directories
    os.makedirs(os.path.join(target_dir, 'JPEGImages'), exist_ok=True)
    os.makedirs(os.path.join(target_dir, 'Annotations'), exist_ok=True)
    
    # Process train and test sets
    for split in ['train', 'test']:
        split_dir = os.path.join(src_dir, split)
        
        # Skip if directory doesn't exist
        if not os.path.exists(split_dir):
            print(f"Warning: {split_dir} does not exist, skipping...")
            continue
        
        # Get all video directories
        video_dirs = [d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))]
        
        for video in tqdm(video_dirs, desc=f"Processing {split} videos"):
            # Source paths
            src_images_dir = os.path.join(split_dir, video, 'images')
            src_masks_dir = os.path.join(split_dir, video, 'masks')
            
            # Target paths - for SAM2, we keep the same folder name for train and test
            # but you might want to add a prefix for test videos to differentiate
            target_video_name = f"{split}_{video}" if split == 'test' else video
            target_images_dir = os.path.join(target_dir, 'JPEGImages', target_video_name)
            target_masks_dir = os.path.join(target_dir, 'Annotations', target_video_name)
            
            # Create target directories
            os.makedirs(target_images_dir, exist_ok=True)
            os.makedirs(target_masks_dir, exist_ok=True)
            
            # Copy images
            if os.path.exists(src_images_dir):
                image_files = sorted([f for f in os.listdir(src_images_dir) if f.endswith('.jpg')])
                
                for img_file in image_files:
                    # SAM2 expects 5-digit frame numbers (00000.jpg)
                    # Our current format is 4-digit (0000.jpg)
                    # Add an extra leading zero
                    new_img_name = '0' + img_file
                    
                    src_img_path = os.path.join(src_images_dir, img_file)
                    target_img_path = os.path.join(target_images_dir, new_img_name)
                    
                    shutil.copy2(src_img_path, target_img_path)
            else:
                print(f"Warning: {src_images_dir} does not exist, skipping...")
            
            # Copy masks
            if os.path.exists(src_masks_dir):
                mask_files = sorted([f for f in os.listdir(src_masks_dir) if f.endswith('.png')])
                
                for mask_file in mask_files:
                    # SAM2 expects 5-digit frame numbers (00000.png)
                    # Our current format is 4-digit (0000.png)
                    # Add an extra leading zero
                    new_mask_name = '0' + mask_file
                    
                    src_mask_path = os.path.join(src_masks_dir, mask_file)
                    target_mask_path = os.path.join(target_masks_dir, new_mask_name)
                    
                    shutil.copy2(src_mask_path, target_mask_path)
            else:
                print(f"Warning: {src_masks_dir} does not exist, skipping...")
    
    # Create a filelist for training videos
    train_videos = [d for d in os.listdir(os.path.join(target_dir, 'JPEGImages')) 
                  if not d.startswith('test_')]
    
    with open(os.path.join(target_dir, 'train_list.txt'), 'w') as f:
        f.write('\n'.join(train_videos))
    
    print(f"Dataset restructured successfully to {target_dir}")
    print(f"Created train_list.txt with {len(train_videos)} videos")

if __name__ == "__main__":
    main() 