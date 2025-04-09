import torch
from sam2 import build_sam2_video_predictor
import logging
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def get_device():
    """Determine the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def test_model_loading():
    """Test loading the SAM2 model with detailed error reporting."""
    try:
        logger.info("Starting model loading test...")
        
        # Get the appropriate device
        device = get_device()
        logger.info(f"Using device: {device}")
        
        # Attempt to build the model
        logger.info("Attempting to build SAM2 video predictor...")
        model = build_sam2_video_predictor(
            device=device,
            mode="eval"
        )
        
        logger.info("Model loaded successfully!")
        
        # Print model summary
        logger.info("\nModel Configuration:")
        logger.info(f"Model device: {next(model.parameters()).device}")
        logger.info(f"Model type: {type(model).__name__}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        logger.error("Traceback:", exc_info=True)
        return False

def main():
    logger.info("=== SAM2 Model Loading Test ===")
    
    # Ensure the models directory exists
    models_dir = Path(__file__).parent.parent / "models"
    models_dir.mkdir(exist_ok=True)
    
    success = test_model_loading()
    
    if success:
        logger.info("\n✓ Model loading test completed successfully!")
    else:
        logger.error("\n✗ Model loading test failed. Please check the error messages above.")
        sys.exit(1)

if __name__ == "__main__":
    main() 