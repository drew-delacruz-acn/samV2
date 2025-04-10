#!/usr/bin/env python
import os
import argparse
import subprocess
import sys
import shutil
from pathlib import Path

# Get the absolute path to the project root directory
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def prepare_dataset():
    """
    Run the dataset preparation script if it hasn't been run yet.
    """
    dataset_dir = os.path.join(ROOT_DIR, 'sam2_dataset')
    
    if not os.path.exists(dataset_dir):
        print("Preparing dataset in DAVIS format...")
        
        # Run the dataset preparation script
        prep_script = os.path.join(ROOT_DIR, 'src', 'prepare_sam2_dataset.py')
        subprocess.run([sys.executable, prep_script], check=True)
    else:
        print(f"Dataset directory {dataset_dir} already exists. Skipping preparation.")
    
    return dataset_dir

def check_sam2_setup():
    """
    Check if SAM2 is installed properly and return the path to the SAM2 directory.
    """
    sam2_dir = os.path.join(ROOT_DIR, 'sam2')
    
    if not os.path.exists(sam2_dir):
        raise FileNotFoundError(f"SAM2 directory not found at {sam2_dir}. "
                               "Please clone the SAM2 repository first.")
    
    # Check if the training module exists
    training_dir = os.path.join(sam2_dir, 'training')
    if not os.path.exists(training_dir):
        raise FileNotFoundError(f"SAM2 training directory not found at {training_dir}. "
                               "Please make sure the SAM2 repository is properly cloned.")
    
    return sam2_dir

def setup_hydra_configs(sam2_dir):
    """
    Set up the SAM2 configs directory for Hydra.
    """
    # Create the configs directory in SAM2 if it doesn't exist
    configs_dir = os.path.join(sam2_dir, 'configs')
    os.makedirs(configs_dir, exist_ok=True)
    
    # Create a custom directory for our configs
    custom_configs_dir = os.path.join(configs_dir, 'custom')
    os.makedirs(custom_configs_dir, exist_ok=True)
    
    return configs_dir, custom_configs_dir

def create_config(dataset_dir, custom_configs_dir, args):
    """
    Create or update the config file with the correct paths.
    """
    # Source config path in our project
    src_config_path = os.path.join(ROOT_DIR, 'sam2_configs', 'custom_finetune.yaml')
    
    # If the source config doesn't exist yet, create it
    if not os.path.exists(os.path.dirname(src_config_path)):
        os.makedirs(os.path.dirname(src_config_path), exist_ok=True)
    
    # If we don't have the config yet, create a basic one
    if not os.path.exists(src_config_path):
        print(f"Creating basic config file at {src_config_path}")
        with open(src_config_path, 'w') as f:
            f.write("""# @package _global_

scratch:
  resolution: 640  # Resolution to resize images to
  train_batch_size: 1  # Start with a small batch size for stability
  num_train_workers: 4  # Adjust based on your CPU cores
  num_frames: 4  # Number of frames to process at once
  max_num_objects: 1  # Our dataset has single objects per video
  base_lr: 5.0e-6  # Start with a low learning rate
  vision_lr: 3.0e-6  # Lower learning rate for vision encoder
  phases_per_epoch: 1
  num_epochs: 10  # Start with fewer epochs for testing

dataset:
  # PATHS to Dataset - these will be set in the training script
  img_folder: null  # PATH to JPEGImages folder
  gt_folder: null   # PATH to Annotations folder
  file_list_txt: null  # PATH to file list containing videos to be used for training
  multiplier: 1

# Video transforms
vos:
  train_transforms:
    - _target_: training.dataset.transforms.ComposeAPI
      transforms:
        - _target_: training.dataset.transforms.RandomHorizontalFlip
          consistent_transform: True
        - _target_: training.dataset.transforms.RandomAffine
          degrees: 20
          shear: 10
          image_interpolation: bilinear
          consistent_transform: True
        - _target_: training.dataset.transforms.RandomResizeAPI
          sizes: ${scratch.resolution}
          square: true
          consistent_transform: True
        - _target_: training.dataset.transforms.ColorJitter
          consistent_transform: True
          brightness: 0.1
          contrast: 0.03
          saturation: 0.03
          hue: null
        - _target_: training.dataset.transforms.RandomGrayscale
          p: 0.05
          consistent_transform: True
        - _target_: training.dataset.transforms.ColorJitter
          consistent_transform: False
          brightness: 0.1
          contrast: 0.05
          saturation: 0.05
          hue: null
        - _target_: training.dataset.transforms.ToTensorAPI
        - _target_: training.dataset.transforms.NormalizeAPI
          mean: [0.485, 0.456, 0.406]
          std: [0.229, 0.224, 0.225]

trainer:
  _target_: training.trainer.Trainer
  mode: train_only
  max_epochs: ${times:${scratch.num_epochs},${scratch.phases_per_epoch}}
  accelerator: cuda
  seed_value: 123

  model:
    _target_: training.model.sam2.SAM2Train
    # We'll use the base model for fine-tuning
    image_encoder_name: "facebook/sam2-hiera-base-plus"
    # Use higher resolution features in the SAM mask decoder
    use_high_res_features_in_sam: true
    # Output 3 masks on the first click on initial conditioning frames
    multimask_output_in_sam: true
    # SAM heads
    iou_prediction_use_sigmoid: True
    # Set image size based on our dataset
    image_size: ${scratch.resolution}

  # Optimizer configuration
  optim:
    _target_: training.optimizer.AdamWMultiGroup
    lr: ${scratch.base_lr}
    weight_decay: 0.01
    param_group_modifiers:
      - _target_: training.optimizer.ImageEncoderParamGroupModifier
        weight_decay: 0.001
        lr_mult: ${div:${scratch.vision_lr},${scratch.base_lr}}

  # Learning rate scheduler
  lr_scheduler:
    _target_: training.optimizer.CosineAnnealingLR
    T_max: ${minus:${trainer.max_epochs},1}
    eta_min: 1e-8

# Dataloaders configuration
data:
  train:
    _target_: training.dataset.sam2_datasets.TorchTrainMixedDataset
    phases_per_epoch: ${phases_per_epoch}
    batch_sizes:
    - ${scratch.train_batch_size}
    datasets:
    # For Video Dataset (DAVIS-style)
    - _target_: training.dataset.vos_dataset.VOSDataset
      training: true
      video_dataset:
        _target_: training.dataset.vos_raw_dataset.DAVISRawDataset
        img_folder: ${dataset.img_folder}
        gt_folder: ${dataset.gt_folder}
        file_list_txt: ${dataset.file_list_txt}
        ann_every: 1  # Use every frame with annotation
      sampler:
        _target_: training.dataset.vos_sampler.RandomUniformSampler
        num_frames: ${scratch.num_frames}  # Number of frames per video sample
        max_num_objects: ${scratch.max_num_objects}
        reverse_time_prob: 0.5  # probability to reverse video
      transforms:
        _target_: training.dataset.transforms.VideoTransforms
        transform_params:
          min_resize_res: 640
          max_resize_res: 640  # Fixed resolution for our small dataset
          jitter_scale: [0.8, 1.2]
    shuffle: True
    num_workers: ${scratch.num_train_workers}
    pin_memory: True
    drop_last: True
    collate_fn:
      _target_: training.utils.data_utils.collate_fn
      _partial_: true
      dict_key: all

# Experiment logging
experiment_log_dir: "./sam2_logs/custom_finetune"
""")
    
    # Read the source config
    with open(src_config_path, 'r') as f:
        config_content = f.read()
    
    # Update paths in the config content
    img_folder = os.path.join(dataset_dir, 'JPEGImages')
    gt_folder = os.path.join(dataset_dir, 'Annotations')
    file_list_txt = os.path.join(dataset_dir, 'train_list.txt')
    
    # Replace the null values with actual paths
    config_content = config_content.replace('img_folder: null', f'img_folder: "{img_folder}"')
    config_content = config_content.replace('gt_folder: null', f'gt_folder: "{gt_folder}"')
    config_content = config_content.replace('file_list_txt: null', f'file_list_txt: "{file_list_txt}"')
    
    # Update batch size if specified
    if args.batch_size:
        config_content = config_content.replace('train_batch_size: 1', f'train_batch_size: {args.batch_size}')
    
    # Update number of epochs if specified
    if args.epochs:
        config_content = config_content.replace('num_epochs: 10', f'num_epochs: {args.epochs}')
    
    # Update model size if specified
    if args.model_size:
        if args.model_size == 'tiny':
            model_name = 'facebook/sam2-hiera-tiny'
        elif args.model_size == 'small':
            model_name = 'facebook/sam2-hiera-small'
        elif args.model_size == 'base':
            model_name = 'facebook/sam2-hiera-base-plus'
        elif args.model_size == 'large':
            model_name = 'facebook/sam2-hiera-large'
        else:
            model_name = 'facebook/sam2-hiera-base-plus'  # Default
        
        config_content = config_content.replace(
            'image_encoder_name: "facebook/sam2-hiera-base-plus"', 
            f'image_encoder_name: "{model_name}"'
        )
    
    # Target config path in SAM2 configs directory
    target_config_path = os.path.join(custom_configs_dir, 'custom_finetune.yaml')
    
    # Write the updated config to the target path
    with open(target_config_path, 'w') as f:
        f.write(config_content)
    
    print(f"Created configuration at {target_config_path}")
    
    return 'custom/custom_finetune'  # Return relative config path for Hydra

def run_training(sam2_dir, config_name, args):
    """
    Run the SAM2 training script with the provided configuration.
    """
    # Navigate to the SAM2 directory
    os.chdir(sam2_dir)
    
    # Build the command to run the training script
    train_script = os.path.join(sam2_dir, 'training', 'train.py')
    
    cmd = [
        sys.executable,
        train_script,
        '-c', config_name,
        '--use-cluster', '0',
        '--num-gpus', str(args.gpus)
    ]
    
    print(f"Running training with command: {' '.join(cmd)}")
    
    # Run the training script
    try:
        subprocess.run(cmd, check=True)
        print("Training completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Training failed with error: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Train SAM2 on custom dataset using official code")
    parser.add_argument('--batch-size', type=int, help='Batch size for training')
    parser.add_argument('--epochs', type=int, help='Number of epochs to train for')
    parser.add_argument('--gpus', type=int, default=1, help='Number of GPUs to use for training')
    parser.add_argument('--model-size', type=str, choices=['tiny', 'small', 'base', 'large'],
                        help='SAM2 model size to use')
    
    args = parser.parse_args()
    
    # Prepare dataset
    dataset_dir = prepare_dataset()
    
    # Check SAM2 setup
    sam2_dir = check_sam2_setup()
    
    # Setup Hydra configs directory
    configs_dir, custom_configs_dir = setup_hydra_configs(sam2_dir)
    
    # Create or update the config file
    config_name = create_config(dataset_dir, custom_configs_dir, args)
    
    # Run the training
    run_training(sam2_dir, config_name, args)

if __name__ == "__main__":
    main() 