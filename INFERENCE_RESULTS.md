# SAM2 Inference Results

This document summarizes the results of running inference with the Segment Anything Model 2 (SAM2) using both image and video predictors on sample data.

## Environment Setup

The inference was run on:
- macOS (darwin 23.5.0)
- Python with PyTorch 2.6.0
- CPU inference (CUDA not available)
- SAM2 'facebook/sam2-hiera-tiny' model

## Image Predictor Results

We created a script (`src/run_image_inference.py`) to run the SAM2 image predictor on a test image:

```python
# Example code snippet
from sam2.sam2_image_predictor import SAM2ImagePredictor
# Initialize predictor
predictor = SAM2ImagePredictor.from_pretrained('facebook/sam2-hiera-tiny', device="cpu")
# Set image
predictor.set_image(image)
# Make predictions with point prompts
masks, scores, _ = predictor.predict(
    point_coords=point_coords,
    point_labels=point_labels,
    multimask_output=True
)
```

### Image Inference Details

- We used a grid of 9 points (3×3) across the test image to generate segmentation masks
- For each point, we:
  - Generated a mask with the point as a foreground prompt
  - Saved the mask as a PNG file
  - Scored the prediction quality
- The results were combined into a visualization showing all 9 masks with their corresponding points

## Video Predictor Results

We created a script (`src/run_video_inference.py`) to run the SAM2 video predictor on a sample video:

```python
# Example code snippet
from sam2.sam2_video_predictor import SAM2VideoPredictor
# Initialize predictor
predictor = SAM2VideoPredictor.from_pretrained('facebook/sam2-hiera-tiny', device="cpu")
# Initialize state with video
state = predictor.init_state(video_path, offload_video_to_cpu=True)
# Add points to track in the first frame
predictor.add_new_points_or_box(
    inference_state=state,
    frame_idx=0,
    obj_id=0,
    points=points,
    labels=labels
)
# Propagate tracking through the video
for frame_idx, obj_ids, masks in predictor.propagate_in_video(state):
    # Process each frame...
```

### Video Inference Details

- Used a grid of points at the center of the first frame to identify the object to track
- Tracked the segmentation masks through all frames of the video
- Saved:
  - Original video frames
  - Binary segmentation masks
  - Visualizations with the mask overlay
  - A composite video showing the tracking results

## Observations

1. **Image Predictor Performance**:
   - Successfully loaded and ran on CPU
   - Generated multiple mask predictions per image
   - Processing time was reasonable for a CPU environment

2. **Video Predictor Performance**:
   - Successfully loaded and initialized with the sample video
   - Processed frames sequentially
   - Generated masks that track objects through the video
   - Working despite a non-critical warning about missing C++ extensions

3. **CPU vs. GPU Considerations**:
   - While the models work on CPU, processing is slower than GPU would be
   - For real-time applications, GPU is recommended
   - The 'tiny' model variant is most suitable for CPU inference

## Recommendations

1. **For Image Segmentation**:
   - SAM2 works well with point prompts
   - For more precise masks, use multiple points or box prompts
   - Consider using the multimask_output option to get multiple candidates

2. **For Video Tracking**:
   - Initialize with clear, distinctive objects
   - Use multiple points to define the object of interest
   - Consider the offload_video_to_cpu option for memory management

3. **General Tips**:
   - Always explicitly specify the device ("cpu" or "cuda")
   - For larger videos or real-time applications, use GPU acceleration
   - Consider model size tradeoffs (tiny vs. small vs. base vs. large)

## Conclusion

The SAM2 model provides powerful segmentation capabilities for both images and videos. It can run on CPU hardware, though with appropriate performance expectations. The API is consistent between model sizes, allowing for easy scaling based on performance requirements. 