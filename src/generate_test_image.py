import cv2
import numpy as np
from pathlib import Path

def generate_test_square():
    # Create a white background
    width, height = 480, 480
    image = np.ones((height, width), dtype=np.uint8) * 255
    
    # Calculate center and square dimensions
    square_size = 100
    center_x = width // 2
    center_y = height // 2
    top_left = (center_x - square_size // 2, center_y - square_size // 2)
    bottom_right = (center_x + square_size // 2, center_y + square_size // 2)
    
    # Draw black square
    cv2.rectangle(image, top_left, bottom_right, 0, -1)  # 0 for black, -1 for filled
    
    # Save the image
    output_path = Path(__file__).parent.parent / 'test_image.jpg'
    cv2.imwrite(str(output_path), image)
    print(f"Test image saved as: {output_path}")

if __name__ == "__main__":
    generate_test_square() 