import streamlit as st
import torch
import joblib
import numpy as np
from PIL import Image
from torchvision import transforms

# Fashion MNIST classes
classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Load the trained model
@st.cache_resource
def load_model():
    try:
        model = joblib.load('model.joblib')
        model.eval()  # Set to evaluation mode
        return model
    except FileNotFoundError:
        st.error("Model file not found. Please run 'python main.py' first to train and save the model.")
        return None

st.title('Fashion MNIST Classifier')

st.write('This app uses a neural network to classify Fashion MNIST images.')
st.write('The model can identify 10 different clothing items:')
st.write(', '.join(classes))

# Create tabs for different functionalities
tab1, tab2 = st.tabs(["Upload Image", "Model Info"])

with tab1:
    st.header("Upload an image")
    uploaded_file = st.file_uploader("Choose a grayscale image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        model = load_model()
        if model:
            # Display the uploaded image
            image = Image.open(uploaded_file).convert('L')  # Convert to grayscale
            st.image(image, caption='Uploaded Image', width=150)
            
            # Preprocess the image
            transform = transforms.Compose([
                transforms.Resize((28, 28)),
                transforms.ToTensor()
            ])
            input_tensor = transform(image).unsqueeze(0)  # Add batch dimension
            
            # Make prediction
            with torch.no_grad():
                output = model(input_tensor)
                probabilities = torch.softmax(output, dim=1)[0]
                predicted_class = torch.argmax(probabilities).item()
            
            # Display results
            st.success(f"Prediction: {classes[predicted_class]}")
            
            # Show probability distribution
            probs = probabilities.numpy()
            st.bar_chart({classes[i]: float(probs[i]) for i in range(10)})

with tab2:
    st.header("Model Architecture")
    st.code("""
    FashionNN(
      (model): Sequential(
        (0): Flatten()
        (1): Linear(in_features=784, out_features=128)
        (2): ReLU()
        (3): Linear(in_features=128, out_features=64)
        (4): ReLU()
        (5): Linear(in_features=64, out_features=10)
      )
    )
    """)
    
    st.write("This neural network was trained on the Fashion MNIST dataset, which consists of 60,000 training images and 10,000 test images.")
    st.write("Each image is a 28x28 grayscale image of a clothing item.")

st.sidebar.info("""
### Instructions
1. Run `python main.py` to train the model (if not already done)
2. Use this app to classify clothing images
3. For best results, use images with a plain background
""")