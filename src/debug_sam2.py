import torch
import traceback
import numpy as np
import os
import argparse
import logging
import time
import sys
from pathlib import Path
from PIL import Image
from create_video import create_sample_video

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import both image and video predictors
try:
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from sam2.sam2_video_predictor import SAM2VideoPredictor
    sam2_imports_successful = True
except ImportError as e:
    logger.error(f"Error importing SAM2: {e}")
    sam2_imports_successful = False

def check_environment():
    """Check the environment for dependencies and system configuration."""
    logger.info("\n=== Environment Check ===")
    logger.info(f'PyTorch version: {torch.__version__}')
    logger.info(f'Is CUDA available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        logger.info(f'CUDA version: {torch.version.cuda}')
        logger.info(f'Current device: {torch.cuda.current_device()}')
        logger.info(f'Device name: {torch.cuda.get_device_name(0)}')
    logger.info(f'Device available: {torch.device("cuda" if torch.cuda.is_available() else "cpu")}')
    
    # Check for important dependencies
    dependencies = [
        ('hydra', 'hydra'),
        ('huggingface_hub', 'huggingface_hub'),
        ('PIL', 'PIL'),
        ('numpy', 'numpy'),
        ('cv2', 'opencv-python')
    ]
    
    for module_name, package_name in dependencies:
        try:
            module = __import__(module_name)
            if hasattr(module, '__version__'):
                logger.info(f'{package_name} version: {module.__version__}')
            else:
                logger.info(f'{package_name} installed')
        except ImportError:
            logger.info(f'{package_name} not installed')

def debug_image_prediction(model_id='facebook/sam2-hiera-tiny', device="cpu"):
    """Debug the SAM2 image predictor."""
    if not sam2_imports_successful:
        logger.error("Cannot test image prediction because SAM2 import failed.")
        return False
        
    logger.info("\n=== Testing SAM2 Image Predictor ===")
    
    try:
        start_time = time.time()
        logger.info(f'Loading image predictor model {model_id} on {device}...')
        predictor = SAM2ImagePredictor.from_pretrained(model_id, device=device)
        logger.info('Model loaded successfully')
        
        # Create a simple test image (black square on white background)
        logger.info('Creating test image...')
        test_image = np.ones((512, 512, 3), dtype=np.uint8) * 255
        test_image[200:300, 200:300] = 0  # Black square
        pil_image = Image.fromarray(test_image)
        
        # Save test image for reference
        test_image_path = 'debug_test_image.png'
        pil_image.save(test_image_path)
        logger.info(f'Saved test image to {test_image_path}')
        
        # Try to use the model
        logger.info('Setting image...')
        predictor.set_image(pil_image)
        logger.info('Image set successfully')
        
        # Try a simple prediction with a point
        logger.info('Making a prediction...')
        input_point = np.array([[250, 250]])  # Center of the black square
        input_label = np.array([1])  # Foreground
        
        masks, scores, _ = predictor.predict(
            point_coords=input_point,
            point_labels=input_label
        )
        
        logger.info(f'Prediction successful! Mask shape: {masks.shape}')
        logger.info(f'Scores: {scores}')
        
        # Save the mask for visualization
        mask_image = Image.fromarray((masks[0] * 255).astype(np.uint8))
        mask_image.save('debug_mask_result.png')
        logger.info('Saved mask to debug_mask_result.png')
        
        end_time = time.time()
        logger.info(f'Image prediction test PASSED in {end_time - start_time:.2f} seconds!')
        return True
        
    except Exception as e:
        logger.error(f'Error in image prediction: {e}')
        logger.error(f'Error type: {type(e).__name__}')
        traceback.print_exc()
        return False

def debug_video_prediction(model_id='facebook/sam2-hiera-tiny', video_path=None, device="cpu"):
    """Debug the SAM2 video predictor."""
    if not sam2_imports_successful:
        logger.error("Cannot test video prediction because SAM2 import failed.")
        return False
        
    logger.info("\n=== Testing SAM2 Video Predictor ===")
    
    # Ensure we have a video file to work with
    if video_path is None or not os.path.exists(video_path):
        logger.info(f"No valid video found at {video_path}, creating a sample video...")
        video_path = 'debug_sample_video.mp4'
        create_sample_video(video_path)
        logger.info(f"Created sample video at {video_path}")
    
    try:
        start_time = time.time()
        logger.info(f'Loading video predictor model {model_id} on {device}...')
        predictor = SAM2VideoPredictor.from_pretrained(model_id, device=device)
        logger.info('Model loaded successfully')
        
        # Try to initialize state with the video
        logger.info('Initializing state with video...')
        state = predictor.init_state(video_path, offload_video_to_cpu=True)
        logger.info('State initialized successfully')
        
        # Print some information about the video
        logger.info(f'Number of frames: {state["num_frames"]}')
        logger.info(f'Video dimensions: {state["video_width"]}x{state["video_height"]}')
        
        # Try adding a point to track an object
        logger.info('Adding point to track...')
        frame_idx = 0  # First frame
        points = np.array([[[state["video_width"] // 2, state["video_height"] // 2]]])  # Center
        point_labels = np.array([[1]])  # Foreground point
        
        frame_idx, obj_ids, masks = predictor.add_new_points_or_box(
            inference_state=state,
            obj_id=0,  # First object
            frame_idx=frame_idx,
            points=points,
            labels=point_labels,
        )
        
        logger.info(f'Added points to frame {frame_idx}')
        logger.info(f'Object IDs: {obj_ids}')
        logger.info(f'Mask shape: {masks.shape if masks is not None else "None"}')
        
        # Try propagating through a few frames
        logger.info('Propagating through video...')
        for i, (frame_idx, obj_ids, masks) in enumerate(predictor.propagate_in_video(state)):
            logger.info(f'Processed frame {frame_idx}, objects: {obj_ids}')
            if i >= 5:  # Just test a few frames
                break
        
        end_time = time.time()
        logger.info(f'Video prediction test PASSED in {end_time - start_time:.2f} seconds!')
        return True
        
    except Exception as e:
        logger.error(f'Error in video prediction: {e}')
        logger.error(f'Error type: {type(e).__name__}')
        traceback.print_exc()
        return False

def test_model_variants():
    """Test multiple model variants and configurations."""
    logger.info("\n=== Testing Multiple SAM2 Model Variants ===")
    
    if not sam2_imports_successful:
        logger.error("Cannot test model variants because SAM2 import failed.")
        return False
    
    # Models to test
    model_configs = [
        # Model type, model ID, device
        ("image", "facebook/sam2-hiera-tiny", "cpu"),
        ("video", "facebook/sam2-hiera-tiny", "cpu"),
    ]
    
    # If CUDA is available, also test with GPU
    if torch.cuda.is_available():
        model_configs.extend([
            ("image", "facebook/sam2-hiera-tiny", "cuda"),
            ("video", "facebook/sam2-hiera-tiny", "cuda"),
            # Uncomment to test larger models
            # ("image", "facebook/sam2-hiera-small", "cuda"),
            # ("image", "facebook/sam2-hiera-base-plus", "cuda"),
            # ("image", "facebook/sam2-hiera-large", "cuda"),
        ])
    
    results = {}
    
    # Run tests
    for model_type, model_id, device in model_configs:
        key = f"{model_type}-{model_id}-{device}"
        logger.info(f"\nTesting {key}...")
        
        try:
            start_time = time.time()
            success = False
            
            if model_type == "image":
                success = debug_image_prediction(model_id, device)
            elif model_type == "video":
                success = debug_video_prediction(model_id, None, device)
            
            end_time = time.time()
            status = "✅ PASS" if success else "❌ FAIL"
            logger.info(f"{status} - {key} in {end_time - start_time:.2f} seconds")
            results[key] = success
            
        except Exception as e:
            logger.error(f"Error testing {key}: {e}")
            results[key] = False
    
    # Print summary
    logger.info("\n=== TEST SUMMARY ===")
    for key, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        logger.info(f"{status} - {key}")
    
    return all(results.values())

def main():
    parser = argparse.ArgumentParser(description="Debug and test SAM2 models")
    parser.add_argument('--mode', type=str, default='env',
                       choices=['image', 'video', 'both', 'env', 'variants'],
                       help='Which mode to debug')
    parser.add_argument('--model', type=str, default='facebook/sam2-hiera-tiny',
                       help='Model ID to use for testing')
    parser.add_argument('--video', type=str, default=None,
                       help='Path to video file for testing')
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cpu', 'cuda'], 
                        help='Device to run tests on')
    
    args = parser.parse_args()
    
    # Always check the environment
    check_environment()
    
    # Track overall success
    success = True
    
    # Run the requested tests
    if args.mode in ['image', 'both']:
        image_result = debug_image_prediction(args.model, args.device)
        logger.info(f"Image test result: {'PASSED' if image_result else 'FAILED'}")
        success = success and image_result
        
    if args.mode in ['video', 'both']:
        video_result = debug_video_prediction(args.model, args.video, args.device)
        logger.info(f"Video test result: {'PASSED' if video_result else 'FAILED'}")
        success = success and video_result
    
    if args.mode == 'variants':
        variants_result = test_model_variants()
        logger.info(f"Model variants test result: {'PASSED' if variants_result else 'FAILED'}")
        success = success and variants_result
    
    logger.info("\n=== Debug Complete ===")
    
    # Return non-zero exit code if any tests failed
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main()) 