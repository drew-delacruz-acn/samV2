#!/usr/bin/env python
import os
import argparse
from pathlib import Path
from run_image_inference import run_image_inference
from run_video_inference import run_video_inference
from evaluate import evaluate_image_segmentation, evaluate_video_tracking

def evaluate_inference(image_path=None, video_path=None, output_dir="inference_results", 
                      model_id="facebook/sam2-hiera-tiny", device="cpu"):
    """
    Run and evaluate both image and video inference if paths are provided.
    """
    base_output_dir = Path(output_dir)
    results = {}

    # Image inference and evaluation
    if image_path:
        print("\n=== Running Image Inference and Evaluation ===")
        image_output_dir = base_output_dir / "image_results"
        
        # Run inference
        run_image_inference(
            image_path=image_path,
            output_dir=str(image_output_dir),
            model_id=model_id,
            device=device
        )
        
        # Evaluate the results
        # We'll evaluate the first mask as the primary prediction
        mask_path = image_output_dir / "mask_point_1.png"
        if mask_path.exists():
            try:
                image_metrics = evaluate_image_segmentation(
                    image_path=image_path,
                    mask_path=str(mask_path)
                )
                results['image_metrics'] = image_metrics
                print("\nImage Evaluation Results:")
                for metric, value in image_metrics.items():
                    print(f"{metric}: {value:.4f}")
            except Exception as e:
                print(f"Error during image evaluation: {str(e)}")
        else:
            print(f"Warning: No mask found at {mask_path}")

    # Video inference and evaluation
    if video_path:
        print("\n=== Running Video Inference and Evaluation ===")
        video_output_dir = base_output_dir / "video_results"
        
        # Run inference
        run_video_inference(
            video_path=video_path,
            output_dir=str(video_output_dir),
            model_id=model_id,
            device=device
        )
        
        # Evaluate the results
        masks_dir = video_output_dir / "masks"
        if masks_dir.exists():
            try:
                video_metrics = evaluate_video_tracking(
                    video_path=video_path,
                    masks_path=str(masks_dir)
                )
                results['video_metrics'] = video_metrics
                print("\nVideo Evaluation Results:")
                for metric, value in video_metrics.items():
                    print(f"{metric}: {value:.4f}")
            except Exception as e:
                print(f"Error during video evaluation: {str(e)}")
        else:
            print(f"Warning: No masks directory found at {masks_dir}")

    return results

def main():
    parser = argparse.ArgumentParser(description="Run and evaluate SAM2 inference")
    parser.add_argument("--image", type=str, help="Path to test image")
    parser.add_argument("--video", type=str, help="Path to test video")
    parser.add_argument("--output", type=str, default="inference_results",
                      help="Directory to save results")
    parser.add_argument("--model", type=str, default="facebook/sam2-hiera-tiny",
                      help="SAM2 model ID")
    parser.add_argument("--device", type=str, default="cpu",
                      choices=["cpu", "cuda"], help="Device to run inference on")
    
    args = parser.parse_args()
    
    if not args.image and not args.video:
        print("Error: Please provide at least one of --image or --video")
        return 1
    
    # Convert paths to absolute
    output_dir = os.path.abspath(args.output)
    image_path = os.path.abspath(args.image) if args.image else None
    video_path = os.path.abspath(args.video) if args.video else None
    
    # Verify input files exist
    if image_path and not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        return 1
    if video_path and not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return 1
    
    # Run evaluation
    evaluate_inference(
        image_path=image_path,
        video_path=video_path,
        output_dir=output_dir,
        model_id=args.model,
        device=args.device
    )
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main()) 