# Prediction and Deployment Guide

This guide explains how to use your trained segmentation model for inference, both locally and in production environments. It covers deployment strategies, API integration, and performance optimization.

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Model Inference Options](#model-inference-options)
3. [Local Inference](#local-inference)
4. [Production Deployment](#production-deployment)
5. [API Integration](#api-integration)
6. [Performance Optimization](#performance-optimization)
7. [Batch Processing](#batch-processing)
8. [Model Evaluation](#model-evaluation)
9. [Troubleshooting](#troubleshooting)

## Prerequisites

- Trained model checkpoint (best_model.pth)
- Python 3.8+ installed
- Required dependencies installed:
  ```bash
  pip install torch torchvision opencv-python numpy matplotlib Pillow tqdm flask
  ```

## Model Inference Options

The project provides two main scripts for inference:

1. **predict.py** - For single image inference
2. **predict_on_dataset.py** - For batch evaluation on test datasets

## Local Inference

### Single Image Prediction

Use `predict.py` to generate a segmentation mask for a single image:

```bash
python src/predict.py --image path/to/your/image.jpg --model models/best_model.pth --output predictions
```

#### Arguments:

| Argument | Description | Default |
|----------|-------------|---------|
| `--image` | Path to input image | Required |
| `--model` | Path to model checkpoint | `models/best_model.pth` |
| `--output` | Directory to save results | `predictions` |
| `--threshold` | Threshold for binary mask | `0.5` |
| `--device` | Device to run on (cpu/cuda) | `cpu` |

#### Outputs:

- Binary mask (`image_mask.jpg`)
- Visualization with probability map (`image_viz.jpg`)
- Segmentation overlay (`image_overlay.jpg`)

### Batch Processing

For processing multiple images at once, use the `predict_batch` function from `predict.py`:

```python
from predict import predict_batch

predict_batch(
    image_dir="path/to/images/",
    model_path="models/best_model.pth",
    output_dir="predictions/batch_results",
    threshold=0.5,
    device="cuda"  # Use "cpu" if no GPU is available
)
```

### Evaluating Model Performance

To evaluate your model's performance on the test dataset:

```bash
python src/predict_on_dataset.py --model models/best_model.pth --dataset dataset --output evaluations
```

This will generate:
- Predictions for all test images
- Visualizations comparing ground truth and predictions
- IoU and accuracy metrics for each video and overall

## Production Deployment

### Option 1: Flask REST API

A simple Flask API can be implemented as follows:

```python
# app.py
import os
import cv2
import base64
import numpy as np
from io import BytesIO
from PIL import Image
import torch
from flask import Flask, request, jsonify

from predict import create_model, preprocess_image

app = Flask(__name__)

# Global variables
model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    """Load the model at server startup"""
    global model
    model_path = os.environ.get("MODEL_PATH", "models/best_model.pth")
    
    # Create and load model
    model = create_model()
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Model loaded from {model_path} on {device}")

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        # Check if the post request has the file part
        if "image" not in request.files:
            return jsonify({"error": "No image provided"}), 400
        
        file = request.files["image"]
        
        # Convert to PIL image
        img = Image.open(BytesIO(file.read())).convert("RGB")
        
        # Set threshold from request or use default
        threshold = float(request.form.get("threshold", 0.5))
        
        # Preprocess image
        img_tensor = preprocess_image(img)[0].to(device)
        
        # Generate prediction
        with torch.no_grad():
            output = model(img_tensor)
        
        # Convert to numpy array
        pred_mask = output.cpu().squeeze().numpy()
        
        # Apply threshold
        binary_mask = (pred_mask > threshold).astype(np.uint8) * 255
        
        # Encode mask as base64 string
        _, buffer = cv2.imencode(".png", binary_mask)
        mask_base64 = base64.b64encode(buffer).decode("utf-8")
        
        return jsonify({
            "success": True,
            "mask_base64": mask_base64,
            "threshold": threshold
        })

if __name__ == "__main__":
    load_model()
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
```

To run the Flask server:

```bash
# Set environment variables
export MODEL_PATH=models/best_model.pth
export PORT=5000

# Start the server
python app.py
```

### Option 2: FastAPI Deployment

For a more modern and high-performance API:

```python
# app_fastapi.py
import os
import cv2
import base64
import numpy as np
from io import BytesIO
from PIL import Image
import torch
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse

from predict import create_model, preprocess_image

app = FastAPI(title="Segmentation API")

# Global variables
model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@app.on_event("startup")
def startup_event():
    """Load model on startup"""
    global model
    model_path = os.environ.get("MODEL_PATH", "models/best_model.pth")
    
    # Create and load model
    model = create_model()
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Model loaded from {model_path} on {device}")

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    threshold: float = Form(0.5)
):
    # Read image
    contents = await file.read()
    img = Image.open(BytesIO(contents)).convert("RGB")
    
    # Preprocess image
    img_tensor = preprocess_image(img)[0].to(device)
    
    # Generate prediction
    with torch.no_grad():
        output = model(img_tensor)
    
    # Convert to numpy array
    pred_mask = output.cpu().squeeze().numpy()
    
    # Apply threshold
    binary_mask = (pred_mask > threshold).astype(np.uint8) * 255
    
    # Encode mask as base64 string
    _, buffer = cv2.imencode(".png", binary_mask)
    mask_base64 = base64.b64encode(buffer).decode("utf-8")
    
    return JSONResponse({
        "success": True,
        "mask_base64": mask_base64,
        "threshold": threshold
    })

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app_fastapi:app", host="0.0.0.0", port=port, reload=True)
```

To run the FastAPI server:

```bash
pip install fastapi uvicorn
python app_fastapi.py
```

### Option 3: Containerization with Docker

For portable deployment across environments:

```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model and code
COPY models/best_model.pth /app/models/
COPY src/predict.py /app/src/
COPY app_fastapi.py .

# Set environment variables
ENV MODEL_PATH=/app/models/best_model.pth
ENV PORT=8000

# Expose the port
EXPOSE 8000

# Start the FastAPI server
CMD ["python", "app_fastapi.py"]
```

Build and run the Docker container:

```bash
docker build -t segmentation-api .
docker run -p 8000:8000 segmentation-api
```

## API Integration

### Client-Side Example (Python)

```python
import requests
import base64
import cv2
import numpy as np
from PIL import Image
import io

def send_image_for_prediction(image_path, api_url, threshold=0.5):
    """
    Send an image to the prediction API and get the segmentation mask.
    
    Args:
        image_path: Path to the input image
        api_url: URL of the prediction API
        threshold: Threshold for binary mask (0-1)
        
    Returns:
        Binary mask as numpy array
    """
    # Prepare the image
    with open(image_path, "rb") as f:
        files = {"image": f}
        data = {"threshold": str(threshold)}
        
        # Make the request
        response = requests.post(api_url, files=files, data=data)
    
    if response.status_code == 200:
        # Decode the mask
        result = response.json()
        mask_base64 = result["mask_base64"]
        mask_bytes = base64.b64decode(mask_base64)
        
        # Convert to numpy array
        mask_array = np.frombuffer(mask_bytes, dtype=np.uint8)
        mask = cv2.imdecode(mask_array, cv2.IMREAD_GRAYSCALE)
        
        return mask
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return None

# Example usage
mask = send_image_for_prediction(
    "test_image.jpg",
    "http://localhost:8000/predict",
    threshold=0.5
)

# Save or display the mask
if mask is not None:
    cv2.imwrite("result_mask.png", mask)
    
    # Display the mask
    from matplotlib import pyplot as plt
    plt.imshow(mask, cmap="gray")
    plt.axis("off")
    plt.show()
```

### Client-Side Example (JavaScript)

```javascript
async function getPrediction(imageFile, threshold = 0.5) {
  // Create form data
  const formData = new FormData();
  formData.append('image', imageFile);
  formData.append('threshold', threshold);
  
  try {
    // Send request to API
    const response = await fetch('http://your-api-url:8000/predict', {
      method: 'POST',
      body: formData,
    });
    
    if (!response.ok) {
      throw new Error(`HTTP error ${response.status}`);
    }
    
    // Parse response
    const result = await response.json();
    
    if (result.success) {
      // Convert base64 to image
      const maskImage = new Image();
      maskImage.src = `data:image/png;base64,${result.mask_base64}`;
      return maskImage;
    } else {
      throw new Error('Prediction failed');
    }
  } catch (error) {
    console.error('Error getting prediction:', error);
    return null;
  }
}

// Example usage with file input
document.getElementById('imageInput').addEventListener('change', async (event) => {
  const file = event.target.files[0];
  if (file) {
    const maskImage = await getPrediction(file, 0.5);
    if (maskImage) {
      // Display the mask
      maskImage.onload = () => {
        const canvas = document.getElementById('resultCanvas');
        const ctx = canvas.getContext('2d');
        canvas.width = maskImage.width;
        canvas.height = maskImage.height;
        ctx.drawImage(maskImage, 0, 0);
      };
    }
  }
});
```

## Performance Optimization

### 1. Model Optimization

- **Quantization**: Reduce model size and improve inference speed
  ```python
  # Example of post-training quantization
  import torch
  
  # Load the model
  model = create_model()
  checkpoint = torch.load("models/best_model.pth", map_location="cpu")
  model.load_state_dict(checkpoint['model_state_dict'])
  model.eval()
  
  # Quantize the model
  quantized_model = torch.quantization.quantize_dynamic(
      model, {torch.nn.Conv2d, torch.nn.ConvTranspose2d}, dtype=torch.qint8
  )
  
  # Save the quantized model
  torch.save({
      'model_state_dict': quantized_model.state_dict(),
      'epoch': checkpoint['epoch'],
  }, "models/quantized_model.pth")
  ```

- **Model Pruning**: Remove unnecessary weights to reduce model size
- **ONNX Conversion**: Convert to ONNX for deployment in various environments

### 2. Inference Optimization

- **Batch Processing**: Process multiple images at once when possible
- **Input Size Reduction**: Resize input images to smaller dimensions
- **Mixed Precision**: Use FP16 computation on supported GPUs
- **TorchScript**: Convert model to TorchScript for optimized inference

### 3. Server Optimization

- Use WSGI/ASGI servers like Gunicorn or Uvicorn for production
- Implement caching for frequent requests
- Use non-blocking I/O for image processing
- Consider horizontal scaling with multiple workers

## Batch Processing

For processing large datasets or video frames:

```python
# batch_processor.py
import os
import argparse
import torch
from tqdm import tqdm
from predict import create_model, preprocess_image

def process_directory(input_dir, model_path, output_dir, threshold=0.5, batch_size=8, device="cpu"):
    """
    Process all images in a directory in batches.
    
    Args:
        input_dir: Directory containing input images
        model_path: Path to model checkpoint
        output_dir: Directory to save results
        threshold: Threshold for binary mask
        batch_size: Number of images to process at once
        device: Device to run inference on
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = [f for f in os.listdir(input_dir) if any(f.lower().endswith(ext) for ext in image_extensions)]
    
    if not image_files:
        print(f"No images found in {input_dir}")
        return
    
    print(f"Found {len(image_files)} images")
    
    # Set device
    device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    model = create_model()
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Process images in batches
    for i in range(0, len(image_files), batch_size):
        batch_files = image_files[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(image_files)-1)//batch_size + 1}...")
        
        # Prepare batch
        batch_tensors = []
        for image_file in batch_files:
            image_path = os.path.join(input_dir, image_file)
            tensor, _ = preprocess_image(image_path)
            batch_tensors.append(tensor)
        
        batch_input = torch.cat(batch_tensors, dim=0).to(device)
        
        # Run inference
        with torch.no_grad():
            batch_output = model(batch_input)
        
        # Process results
        for j, image_file in enumerate(batch_files):
            mask = batch_output[j].cpu().squeeze().numpy()
            binary_mask = (mask > threshold).astype(np.uint8) * 255
            
            # Save mask
            output_path = os.path.join(output_dir, f"{os.path.splitext(image_file)[0]}_mask.png")
            cv2.imwrite(output_path, binary_mask)
    
    print(f"Processed {len(image_files)} images. Results saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch process images with segmentation model")
    parser.add_argument("--input", required=True, help="Input directory containing images")
    parser.add_argument("--model", default="models/best_model.pth", help="Model checkpoint path")
    parser.add_argument("--output", default="batch_results", help="Output directory")
    parser.add_argument("--threshold", type=float, default=0.5, help="Mask threshold")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu", help="Device")
    
    args = parser.parse_args()
    
    # Import here to avoid circular imports
    import cv2
    import numpy as np
    
    process_directory(
        args.input, args.model, args.output, 
        args.threshold, args.batch_size, args.device
    )
```

## Model Evaluation

To evaluate the model's performance on a dataset:

```bash
python src/predict_on_dataset.py --model models/best_model.pth --dataset dataset --output evaluations
```

This will:

1. Run the model on all test videos in the dataset
2. Calculate IoU and accuracy metrics for each video
3. Generate visualizations comparing predictions to ground truth
4. Save overall metrics to a summary file

The evaluation script produces these metrics:
- **IoU (Intersection over Union)**: Measures overlap between prediction and ground truth
- **Accuracy**: Percentage of correctly classified pixels

## Troubleshooting

### Common Issues During Inference

1. **Out of Memory (OOM) errors**:
   - Reduce batch size
   - Process images at a lower resolution
   - Use CPU instead of GPU for very large images

2. **Slow inference**:
   - Use GPU if available
   - Implement batching for multiple images
   - Consider model optimization techniques

3. **Poor quality predictions**:
   - Try different threshold values
   - Ensure input images are properly preprocessed
   - Check if the model was trained on similar data

4. **API deployment issues**:
   - Check for correct CORS settings in production
   - Ensure proper error handling for all API endpoints
   - Monitor memory usage during peak loads 