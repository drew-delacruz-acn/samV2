#!/usr/bin/env python
import os
import sys
import torch
import logging
import time
import traceback
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_model_variant(model_type, model_id, device):
    """Test loading and basic operation of a SAM2 model variant"""
    logger.info(f"Testing {model_type} with {model_id} on device={device}")
    
    try:
        start_time = time.time()
        
        if model_type == "image":
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            predictor = SAM2ImagePredictor.from_pretrained(model_id, device=device)
            logger.info(f"✅ Successfully loaded {model_type} model {model_id}")
            
            # Test image creation
            import numpy as np
            from PIL import Image
            test_image = np.ones((512, 512, 3), dtype=np.uint8) * 255
            test_image[200:300, 200:300] = 0  # Black square
            pil_image = Image.fromarray(test_image)
            
            # Set image
            predictor.set_image(pil_image)
            logger.info(f"✅ Successfully set image")
            
            # Make prediction
            input_point = np.array([[250, 250]])  # Center of the black square
            input_label = np.array([1])  # Foreground
            masks, _, _ = predictor.predict(
                point_coords=input_point,
                point_labels=input_label
            )
            logger.info(f"✅ Prediction successful! Mask shape: {masks.shape}")
            
        elif model_type == "video":
            # Check for a test video
            test_video = os.path.join(os.getcwd(), "sample_video.mp4")
            if not os.path.exists(test_video):
                logger.warning(f"Test video not found at {test_video}. Skipping video inference tests.")
                return
                
            from sam2.sam2_video_predictor import SAM2VideoPredictor
            predictor = SAM2VideoPredictor.from_pretrained(model_id, device=device)
            logger.info(f"✅ Successfully loaded {model_type} model {model_id}")
            
            # Initialize state with the video
            state = predictor.init_state(test_video, offload_video_to_cpu=True)
            logger.info(f"✅ Successfully initialized state with video")
            logger.info(f"   Number of frames: {state['num_frames']}")
            logger.info(f"   Video dimensions: {state['video_width']}x{state['video_height']}")
            
            # Add a point to track
            import numpy as np
            frame_idx = 0  # First frame
            points = np.array([[state["video_width"] // 2, state["video_height"] // 2]])  # Center
            point_labels = np.array([1])  # Foreground point
            
            frame_idx, obj_ids, masks = predictor.add_new_points_or_box(
                inference_state=state,
                obj_id=0,  # First object
                frame_idx=frame_idx,
                points=points,
                labels=point_labels,
            )
            logger.info(f"✅ Successfully added tracking points")
            logger.info(f"   Mask shape: {masks.shape}")
        
        end_time = time.time()
        logger.info(f"✅ All tests passed for {model_type} model {model_id} in {end_time - start_time:.2f} seconds")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error with {model_type} model {model_id} on device={device}: {e}")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error("Traceback:")
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        logger.info(f"CUDA device count: {torch.cuda.device_count()}")
        logger.info(f"CUDA device name: {torch.cuda.get_device_name(0)}")
    
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
            # Can add more models if desired
            # ("image", "facebook/sam2-hiera-small", "cuda"),
            # ("image", "facebook/sam2-hiera-base-plus", "cuda"),
            # ("image", "facebook/sam2-hiera-large", "cuda"),
        ])
    
    results = {}
    
    # Run tests
    for model_type, model_id, device in model_configs:
        key = f"{model_type}-{model_id}-{device}"
        results[key] = test_model_variant(model_type, model_id, device)
    
    # Print summary
    logger.info("\n=== TEST SUMMARY ===")
    for key, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        logger.info(f"{status} - {key}")
    
    # Return non-zero exit code if any tests failed
    return 0 if all(results.values()) else 1

if __name__ == "__main__":
    sys.exit(main()) 