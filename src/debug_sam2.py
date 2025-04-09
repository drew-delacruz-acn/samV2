import torch
import traceback
import numpy as np
from PIL import Image
from sam2.sam2_image_predictor import SAM2ImagePredictor

print('Is CUDA available:', torch.cuda.is_available())
print('Device available:', torch.device('cpu'))
print('PyTorch version:', torch.__version__)

# Check for important dependencies
try:
    import hydra
    print('Hydra version installed:', hydra.__version__)
except ImportError:
    print('Hydra not installed')

try:
    import huggingface_hub
    print('Huggingface Hub version:', huggingface_hub.__version__)
except ImportError:
    print('Huggingface Hub not installed')

try:
    print('Attempting to load model...')
    # Use SAM2ImagePredictor for static images
    predictor = SAM2ImagePredictor.from_pretrained('facebook/sam2-hiera-tiny', device="cpu")
    print('Model loaded successfully')
    
    # Create a simple test image (black square on white background)
    print('Creating test image...')
    test_image = np.ones((512, 512, 3), dtype=np.uint8) * 255
    test_image[200:300, 200:300] = 0  # Black square
    pil_image = Image.fromarray(test_image)
    
    # Try to use the model
    print('Setting image...')
    try:
        predictor.set_image(pil_image)
        print('Image set successfully')
        
        # Try a simple prediction with a point
        print('Making a prediction...')
        input_point = np.array([[250, 250]])  # Center of the black square
        input_label = np.array([1])  # Foreground
        masks, _, _ = predictor.predict(
            point_coords=input_point,
            point_labels=input_label
        )
        print(f'Prediction successful! Mask shape: {masks.shape}')
        print('Model is working!')
        
    except Exception as e:
        print(f'Error using model: {e}')
        print(f'Error type: {type(e).__name__}')
        traceback.print_exc()
        
except Exception as e:
    print(f'Error loading model: {e}')
    print(f'Error type: {type(e).__name__}')
    print('Traceback:')
    traceback.print_exc() 