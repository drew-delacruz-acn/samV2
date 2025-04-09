import cv2
import numpy as np
from pathlib import Path
from sam2 import SAM2ImagePredictor
from evaluate_performance import evaluate_image_segmentation

def create_test_data():
    # Create test directories if they don't exist
    test_img_dir = Path('data/test_images')
    test_mask_dir = Path('data/test_masks')
    test_img_dir.mkdir(parents=True, exist_ok=True)
    test_mask_dir.mkdir(parents=True, exist_ok=True)

    # Create a sample test image (480x480)
    test_image = np.zeros((480, 480, 3), dtype=np.uint8)
    # Add a white rectangle in the middle
    cv2.rectangle(test_image, (180, 180), (300, 300), (255, 255, 255), -1)
    cv2.imwrite(str(test_img_dir / 'test1.jpg'), test_image)

    # Create a corresponding ground truth mask (480x480)
    test_mask = np.zeros((480, 480), dtype=np.uint8)
    # Add a white rectangle slightly offset from the image
    cv2.rectangle(test_mask, (190, 190), (310, 310), 255, -1)
    cv2.imwrite(str(test_mask_dir / 'test1.png'), test_mask)

    return str(test_img_dir / 'test1.jpg'), str(test_mask_dir / 'test1.png')

def main():
    print("Creating test data...")
    test_image_path, test_mask_path = create_test_data()
    
    print("\nInitializing SAM2 predictor...")
    try:
        predictor = SAM2ImagePredictor.from_pretrained("mobile_sam")
        
        print("\nRunning evaluation...")
        results = evaluate_image_segmentation(
            predictor=predictor,
            image_path=test_image_path,
            mask_path=test_mask_path,
            target_size=(480, 480)
        )
        
        print("\nEvaluation Results:")
        print(f"Average IoU: {results['average_iou']:.4f}")
        print(f"Average Dice: {results['average_dice']:.4f}")
        print(f"Best Point IoU: {results['best_point']['iou']:.4f} at {results['best_point']['location']}")
        print(f"Worst Point IoU: {results['worst_point']['iou']:.4f} at {results['worst_point']['location']}")
        
        print("\nPoint-wise Results:")
        for i, point_metrics in enumerate(results['points']):
            print(f"Point {i+1} at {point_metrics['location']}:")
            print(f"  IoU: {point_metrics['iou']:.4f}")
            print(f"  Dice: {point_metrics['dice']:.4f}")
            
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")

if __name__ == "__main__":
    main() 