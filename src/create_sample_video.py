import numpy as np
import cv2

def create_sample_video(output_path='sample_video.mp4', resolution=(512, 512), duration=5, fps=30):
    """
    Create a simple test video with a moving black circle on a white background.
    
    Args:
        output_path: Path to save the output video.
        resolution: (width, height) tuple for video resolution.
        duration: Duration of the video in seconds.
        fps: Frames per second.
    """
    # Video parameters
    width, height = resolution
    total_frames = duration * fps
    
    # Create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use mp4v codec
    out = cv2.VideoWriter(output_path, fourcc, fps, resolution)
    
    # Generate frames with a moving circle
    for i in range(total_frames):
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
        
        if i % fps == 0:
            print(f"Generated {i+1}/{total_frames} frames...")
    
    # Release the VideoWriter
    out.release()
    print(f"Video saved to {output_path}")

if __name__ == "__main__":
    create_sample_video()
    print("Sample video created successfully!") 