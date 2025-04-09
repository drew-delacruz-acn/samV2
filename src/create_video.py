import cv2
import os
import numpy as np
from pathlib import Path
from glob import glob
from tqdm import tqdm

def create_video_from_frames(frames_dir, output_path, fps=30):
    """
    Create a video file from a sequence of frames.
    
    Args:
        frames_dir (str): Directory containing the frame images
        output_path (str): Path where to save the video
        fps (int): Frames per second for the output video
        
    Returns:
        bool: True if video was created successfully, False otherwise
    """
    # Get all frames
    frames = sorted(glob(os.path.join(frames_dir, "*.jpg")))
    if not frames:
        print(f"No frames found in {frames_dir}")
        return False
        
    # Read first frame to get dimensions
    first_frame = cv2.imread(frames[0])
    height, width = first_frame.shape[:2]
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"Creating video from {len(frames)} frames...")
    for frame_path in tqdm(frames, desc="Processing frames"):
        frame = cv2.imread(frame_path)
        out.write(frame)
    
    out.release()
    print(f"Video saved to: {output_path}")
    return True

def create_sample_video(output_path='sample_video.mp4', resolution=(512, 512), duration=5, fps=30):
    """
    Create a simple test video with a moving black circle on a white background.
    
    Args:
        output_path (str): Path to save the output video.
        resolution (tuple): (width, height) tuple for video resolution.
        duration (int): Duration of the video in seconds.
        fps (int): Frames per second.
        
    Returns:
        bool: True if video was created successfully
    """
    # Video parameters
    width, height = resolution
    total_frames = duration * fps
    
    # Create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use mp4v codec
    out = cv2.VideoWriter(output_path, fourcc, fps, resolution)
    
    # Generate frames with a moving circle
    for i in tqdm(range(total_frames), desc="Generating frames"):
        # Create white background
        frame = np.ones((height, width, 3), dtype=np.uint8) * 255
        
        # Calculate circle position (moving in a circular path)
        angle = (i / total_frames) * 2 * np.pi
        center_x = int(width/2 + width/4 * np.cos(angle))
        center_y = int(height/2 + height/4 * np.sin(angle))
        radius = 50
        
        # Draw black circle
        cv2.circle(frame, (center_x, center_y), radius, (0, 0, 0), -1)
        
        # Write the frame to the video
        out.write(frame)
    
    # Release the VideoWriter
    out.release()
    print(f"Sample video saved to {output_path}")
    return True

def generate_test_image(output_path=None, size=480, square_size=100):
    """
    Generate a test image with a black square on a white background.
    
    Args:
        output_path (str, optional): Path to save the image. If None, uses default location.
        size (int): Size of the image (width and height)
        square_size (int): Size of the black square
        
    Returns:
        str: Path to the saved image
    """
    # Create a white background
    width, height = size, size
    image = np.ones((height, width), dtype=np.uint8) * 255
    
    # Calculate center and square dimensions
    center_x = width // 2
    center_y = height // 2
    top_left = (center_x - square_size // 2, center_y - square_size // 2)
    bottom_right = (center_x + square_size // 2, center_y + square_size // 2)
    
    # Draw black square
    cv2.rectangle(image, top_left, bottom_right, 0, -1)  # 0 for black, -1 for filled
    
    # Determine output path
    if output_path is None:
        output_path = str(Path(__file__).parent.parent / 'test_image.jpg')
    
    # Save the image
    cv2.imwrite(output_path, image)
    print(f"Test image saved as: {output_path}")
    return output_path

def create_dataset_videos(base_dir=None):
    """
    Create videos for both test sequences in the dataset.
    
    Args:
        base_dir (str, optional): Base directory of the dataset.
                                 If None, use default location.
    """
    if base_dir is None:
        base_dir = Path(__file__).parent.parent / 'dataset'
    else:
        base_dir = Path(base_dir)
    
    # Process test videos
    for video_num in [1, 2]:
        frames_dir = base_dir / 'test' / f'video{video_num}' / 'images'
        output_path = base_dir / 'test' / f'video{video_num}.mp4'
        
        print(f"\nProcessing test video {video_num}...")
        if create_video_from_frames(str(frames_dir), str(output_path)):
            print(f"Successfully created test video {video_num}")
    
    print("\nVideo creation complete!")

def main():
    """Main function that provides command-line interface."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Video and image generation utilities")
    
    subparsers = parser.add_subparsers(dest='command', help='Command')
    
    # Parser for creating dataset videos
    dataset_parser = subparsers.add_parser('dataset', help='Create videos from dataset frames')
    dataset_parser.add_argument('--base_dir', type=str, default=None,
                              help='Base directory of the dataset')
    
    # Parser for creating sample video
    sample_parser = subparsers.add_parser('sample', help='Create a sample video with moving circle')
    sample_parser.add_argument('--output', type=str, default='sample_video.mp4',
                             help='Output path for the sample video')
    sample_parser.add_argument('--width', type=int, default=512,
                             help='Width of the video')
    sample_parser.add_argument('--height', type=int, default=512,
                             help='Height of the video')
    sample_parser.add_argument('--duration', type=int, default=5,
                             help='Duration of the video in seconds')
    sample_parser.add_argument('--fps', type=int, default=30,
                             help='Frames per second')
    
    # Parser for creating custom video from frames
    frames_parser = subparsers.add_parser('frames', help='Create video from frames')
    frames_parser.add_argument('--frames_dir', type=str, required=True,
                             help='Directory containing frame images')
    frames_parser.add_argument('--output', type=str, required=True,
                             help='Output path for the video')
    frames_parser.add_argument('--fps', type=int, default=30,
                             help='Frames per second')
    
    # Parser for generating test image
    image_parser = subparsers.add_parser('image', help='Generate a test image with a black square')
    image_parser.add_argument('--output', type=str, default=None,
                            help='Output path for the test image')
    image_parser.add_argument('--size', type=int, default=480,
                            help='Size of the image (width and height)')
    image_parser.add_argument('--square_size', type=int, default=100,
                            help='Size of the black square')
    
    args = parser.parse_args()
    
    if args.command == 'dataset':
        create_dataset_videos(args.base_dir)
    elif args.command == 'sample':
        create_sample_video(
            output_path=args.output,
            resolution=(args.width, args.height),
            duration=args.duration,
            fps=args.fps
        )
    elif args.command == 'frames':
        create_video_from_frames(
            frames_dir=args.frames_dir,
            output_path=args.output,
            fps=args.fps
        )
    elif args.command == 'image':
        generate_test_image(
            output_path=args.output,
            size=args.size,
            square_size=args.square_size
        )
    else:
        # If no command provided, print help
        parser.print_help()

if __name__ == "__main__":
    main() 