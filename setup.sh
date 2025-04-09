#!/bin/bash

# Setup script for SAM2 Fine-Tuning Project

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is not installed. Please install Python 3 and try again."
    exit 1
fi

# Create a virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

# Activate the virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Create output directories if they don't exist
echo "Creating output directories..."
mkdir -p models evaluation_results

echo "Setup complete! Activate the virtual environment with 'source venv/bin/activate'"
echo "Run 'python src/dataset_generator.py' to generate the synthetic dataset" 