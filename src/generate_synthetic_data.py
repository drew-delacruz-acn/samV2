import cv2
import numpy as np
import os
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
    
    for i in range(num_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        mask = np.zeros((height, width), dtype=np.uint8)

        for func in movement_funcs:
            x, y = func(i)
            cv2.circle(frame, (x, y), radius, (255, 255, 255), -1)  # White circle
            cv2.circle(mask, (x, y), radius, 255, -1)  # Foreground in mask

        cv2.imwrite(os.path.join(images_dir, f'{i:04d}.jpg'), frame)
        cv2.imwrite(os.path.join(masks_dir, f'{i:04d}.png'), mask)
        
        # Save the first frame of train/video1 as sample
        if i == 0 and 'video1' in video_dir and 'train' in video_dir:
            cv2.imwrite(str(sample_path), frame)

# Parameters
width = 640
height = 480
num_frames = 100
radius = 30

base_dir = Path(__file__).parent.parent / 'dummy_dataset'
train_dir = base_dir / 'train'
test_dir = base_dir / 'test'

train_dir.mkdir(parents=True, exist_ok=True)
test_dir.mkdir(parents=True, exist_ok=True)

# Movement Functions
def horizontal_movement(i):
    """Circle moves horizontally from left to right."""
    x = (50 + i * 5) % width
    y = height // 2
    return int(x), y

def vertical_movement(i):
    """Circle moves vertically from top to bottom."""
    x = width // 2
    y = (50 + i * 5) % height
    return x, int(y)

def diagonal_movement1(i):
    """Circle moves diagonally from top-left to bottom-right."""
    x = (50 + i * 3) % width
    y = (50 + i * 3) % height
    return int(x), int(y)

def diagonal_movement2(i):
    """Circle moves diagonally from top-right to bottom-left."""
    x = (width - 50 - i * 3) % width
    y = (50 + i * 3) % height
    return int(x), int(y)

# Circular movement for test set
center_x, center_y = width // 2, height // 2
circle_radius = 100
angle_speed = 0.1

def circular_movement(i):
    """Circle moves in a circular path around the center."""
    x = center_x + circle_radius * np.cos(i * angle_speed)
    y = center_y + circle_radius * np.sin(i * angle_speed)
    return int(x), int(y)

def generate_random_path(num_frames, width, height, step=10):
    """Generates a random path for an object within the frame."""
    path = []
    x, y = np.random.randint(0, width-1), np.random.randint(0, height-1)
    for _ in range(num_frames):
        dx, dy = np.random.randint(-step, step), np.random.randint(-step, step)
        x = np.clip(x + dx, 0, width - 1)
        y = np.clip(y + dy, 0, height - 1)
        path.append((int(x), int(y)))
    return path

def main():
    print("Generating synthetic dataset...")
    
    # Generate Training Videos
    print("\nGenerating training videos...")
    
    # Video 1: One circle moving horizontally
    print("- Generating training video 1: Horizontal movement")
    train_video1_dir = train_dir / 'video1'
    movement_funcs_train1 = [horizontal_movement]
    generate_video_sequence(str(train_video1_dir), num_frames, width, height, movement_funcs_train1)

    # Video 2: One circle moving vertically
    print("- Generating training video 2: Vertical movement")
    train_video2_dir = train_dir / 'video2'
    movement_funcs_train2 = [vertical_movement]
    generate_video_sequence(str(train_video2_dir), num_frames, width, height, movement_funcs_train2)

    # Video 3: Two circles moving diagonally
    print("- Generating training video 3: Diagonal movements")
    train_video3_dir = train_dir / 'video3'
    movement_funcs_train3 = [diagonal_movement1, diagonal_movement2]
    generate_video_sequence(str(train_video3_dir), num_frames, width, height, movement_funcs_train3)

    # Generate Test Videos
    print("\nGenerating test videos...")
    
    # Video 1: One circle moving in a circular path
    print("- Generating test video 1: Circular movement")
    test_video1_dir = test_dir / 'video1'
    movement_funcs_test1 = [circular_movement]
    generate_video_sequence(str(test_video1_dir), num_frames, width, height, movement_funcs_test1)

    # Video 2: Two circles moving randomly
    print("- Generating test video 2: Random movements")
    test_video2_dir = test_dir / 'video2'
    path1 = generate_random_path(num_frames, width, height)
    path2 = generate_random_path(num_frames, width, height)
    movement_funcs_test2 = [lambda i: path1[i], lambda i: path2[i]]
    generate_video_sequence(str(test_video2_dir), num_frames, width, height, movement_funcs_test2)

    print("\nDataset generation complete!")
    print(f"Dataset location: {base_dir}")
    print("Sample frame saved as 'sample_frame.jpg' in project root")

if __name__ == "__main__":
    main() 