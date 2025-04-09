import torch
import traceback
import numpy as np
import os
from sam2.sam2_video_predictor import SAM2VideoPredictor

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

# Check if a video file exists
video_path = 'sample_video.mp4'
if not os.path.exists(video_path):
    print(f"Warning: {video_path} doesn't exist. Please download a sample MP4 video file.")
    print("You can create one or download one and name it 'sample_video.mp4' in the project root.")
    exit(1)

try:
    print('Attempting to load model...')
    predictor = SAM2VideoPredictor.from_pretrained('facebook/sam2-hiera-tiny', device="cpu")
    print('Model loaded successfully')
    
    # Try to initialize state with the video
    print('Initializing state with video...')
    try:
        state = predictor.init_state(video_path, offload_video_to_cpu=True)
        print('State initialized successfully')
        
        # Print some information about the video
        print(f'Number of frames: {state["num_frames"]}')
        print(f'Video dimensions: {state["video_width"]}x{state["video_height"]}')
        
        # Try adding a point to track an object
        print('Adding point to track...')
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
        
        print(f'Added points to frame {frame_idx}')
        print(f'Object IDs: {obj_ids}')
        print(f'Mask shape: {masks.shape}')
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