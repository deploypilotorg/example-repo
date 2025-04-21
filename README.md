# Fashion MNIST Classifier with Streamlit

This repository contains a simple PyTorch implementation of a Fashion MNIST classifier with a Streamlit web interface.

## Overview

The application provides a user-friendly interface to:
- View the model architecture
- Train the model with adjustable parameters
- Test the model on sample images from the test set
- Upload custom images for prediction

## Requirements

- Python 3.7+
- PyTorch and torchvision
- Streamlit
- matplotlib
- PIL
- numpy

All dependencies are listed in the `requirements.txt` file.

## Setup and Installation

1. Clone this repository:
   ```
   git clone https://github.com/deploypilotorg/example-repo.git
   cd example-repo
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Create a virtual environment (recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

## Running the Application

1. First, train the model:
   ```
   python main.py
   ```

2. Then run the Streamlit application:
   ```
   streamlit run streamlit_app.py
   ```

This will start a local server and open the application in your web browser.

## Features

### Model Overview
- View the neural network architecture used for classification
- Learn about the Fashion MNIST dataset and its classes

### Train Model
- Adjust hyperparameters like epochs, batch size, and learning rate
- Track training progress in real-time
- View training loss and final test accuracy

### Test on Sample Images
- View predictions on randomly selected test images
- Compare predictions with actual labels
- See probability distributions for top predictions

### Upload Your Own Image
- Upload and test custom images
- View probability distribution across all classes
- Images will be automatically resized and converted to grayscale

## Model Architecture

The model is a simple feed-forward neural network with:
- Input layer: 28×28 = 784 input features (flattened image)
- Hidden layer 1: 128 neurons with ReLU activation
- Hidden layer 2: 64 neurons with ReLU activation
- Output layer: 10 neurons (one for each class)

## Implementation Details

The implementation consists of two main files:
- `main.py`: Contains the model definition, training code, and saves the trained model
- `streamlit_app.py`: Provides a web interface to use the trained model for image classification
# Test branch change
# Test modification for test10 branch
