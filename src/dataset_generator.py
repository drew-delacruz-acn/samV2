import cv2
import numpy as np
import os
import random
from tqdm import tqdm
from pathlib import Path

def generate_video_sequence(video_dir, num_frames, width, height, movement_funcs):
    """
    Generates a sequence of frames and corresponding masks for a video with moving objects.
    
    Args:
        video_dir (str): Directory to save the video frames and masks.
        num_frames (int): Number of frames in the video.
        width (int): Width of the frames.
        height (int): Height of the frames.
        movement_funcs (list): List of functions defining the movement of each object.
    """
    images_dir = os.path.join(video_dir, 'images')
    masks_dir = os.path.join(video_dir, 'masks')
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)

    # Also save a sample frame in root directory for quick testing
    if 'video1' in video_dir and 'train' in video_dir:
        sample_path = Path(__file__).parent.parent / 'sample_frame.jpg'
    
    for i in tqdm(range(num_frames), desc=f"Generating {os.path.basename(video_dir)}"):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        mask = np.zeros((height, width), dtype=np.uint8)

        for func in movement_funcs:
            if len(func(i)) == 2:  # Compatibility with both function signatures
                x, y = func(i)
                r = radius  # Use global radius
            else:
                x, y, r = func(i)
                
            cv2.circle(frame, (x, y), r, (255, 255, 255), -1)  # White circle
            cv2.circle(mask, (x, y), r, 255, -1)               # Foreground in mask

        cv2.imwrite(os.path.join(images_dir, f'{i:04d}.jpg'), frame)
        cv2.imwrite(os.path.join(masks_dir, f'{i:04d}.png'), mask)
        
        # Save the first frame of train/video1 as sample
        if i == 0 and 'video1' in video_dir and 'train' in video_dir:
            cv2.imwrite(str(sample_path), frame)

# Movement Functions
def horizontal_movement(i, width=640, height=480, radius=30):
    """Circle moves horizontally from left to right."""
    x = (50 + i * 5) % (width - radius * 2) + radius
    y = height // 2
    return int(x), y, radius

def vertical_movement(i, width=640, height=480, radius=30):
    """Circle moves vertically from top to bottom."""
    x = width // 2
    y = (50 + i * 5) % (height - radius * 2) + radius
    return x, int(y), radius

def diagonal_movement1(i, width=640, height=480, radius=30):
    """Circle moves diagonally from top-left to bottom-right."""
    x = (50 + i * 3) % (width - radius * 2) + radius
    y = (50 + i * 3) % (height - radius * 2) + radius
    return int(x), int(y), radius

def diagonal_movement2(i, width=640, height=480, radius=30):
    """Circle moves diagonally from top-right to bottom-left."""
    x = (width - 50 - i * 3) % (width - radius * 2) + radius
    y = (50 + i * 3) % (height - radius * 2) + radius
    return int(x), int(y), radius

def circular_movement(i, width=640, height=480, radius=30):
    """Circle moves in a circular path around the center."""
    center_x, center_y = width // 2, height // 2
    circle_radius = 100
    angle_speed = 0.1
    
    x = center_x + circle_radius * np.cos(i * angle_speed)
    y = center_y + circle_radius * np.sin(i * angle_speed)
    return int(x), int(y), radius

def generate_random_path(num_frames, width=640, height=480, radius=30, step=10):
    """Generates a random path for an object within the frame."""
    path = []
    x, y = random.randint(radius, width-radius), random.randint(radius, height-radius)
    for _ in range(num_frames):
        dx, dy = random.randint(-step, step), random.randint(-step, step)
        x = np.clip(x + dx, radius, width - radius)
        y = np.clip(y + dy, radius, height - radius)
        path.append((int(x), int(y), radius))
    return path

def generate_dataset(base_dir=None, width=640, height=480, num_frames=100, radius=30):
    """
    Generate the synthetic dataset with all video types.
    
    Args:
        base_dir (str, optional): Base directory to save the dataset.
                                 If None, uses default location.
        width (int): Frame width.
        height (int): Frame height.
        num_frames (int): Number of frames per video.
        radius (int): Radius of the circles.
    """
    if base_dir is None:
        base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'dataset')
    else:
        base_dir = Path(base_dir)
        
    train_dir = os.path.join(base_dir, 'train')
    test_dir = os.path.join(base_dir, 'test')

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    print("Generating training videos...")
    # Video 1: One circle moving horizontally
    train_video1_dir = os.path.join(train_dir, 'video1')
    movement_funcs_train1 = [lambda i: horizontal_movement(i, width, height, radius)]
    generate_video_sequence(train_video1_dir, num_frames, width, height, movement_funcs_train1)

    # Video 2: One circle moving vertically
    train_video2_dir = os.path.join(train_dir, 'video2')
    movement_funcs_train2 = [lambda i: vertical_movement(i, width, height, radius)]
    generate_video_sequence(train_video2_dir, num_frames, width, height, movement_funcs_train2)

    # Video 3: Two circles moving diagonally
    train_video3_dir = os.path.join(train_dir, 'video3')
    movement_funcs_train3 = [
        lambda i: diagonal_movement1(i, width, height, radius),
        lambda i: diagonal_movement2(i, width, height, radius)
    ]
    generate_video_sequence(train_video3_dir, num_frames, width, height, movement_funcs_train3)

    print("Generating test videos...")
    # Video 1: One circle moving in a circular path
    test_video1_dir = os.path.join(test_dir, 'video1')
    movement_funcs_test1 = [lambda i: circular_movement(i, width, height, radius)]
    generate_video_sequence(test_video1_dir, num_frames, width, height, movement_funcs_test1)

    # Video 2: Two circles moving randomly
    test_video2_dir = os.path.join(test_dir, 'video2')
    random_path1 = generate_random_path(num_frames, width, height, radius)
    random_path2 = generate_random_path(num_frames, width, height, radius)
    movement_funcs_test2 = [
        lambda i: random_path1[i],
        lambda i: random_path2[i]
    ]
    generate_video_sequence(test_video2_dir, num_frames, width, height, movement_funcs_test2)

    print("Dummy dataset with three training videos and two test videos generated successfully.")
    print(f"Dataset location: {base_dir}")
    print("Sample frame saved as 'sample_frame.jpg' in project root (if training video 1 was generated)")

def main():
    """Generate the synthetic dataset using default parameters."""
    # Parameters
    width = 640
    height = 480
    num_frames = 100
    radius = 30

    # Generate dataset
    generate_dataset(width=width, height=height, num_frames=num_frames, radius=radius)

if __name__ == "__main__":
    main() 