import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import io
from PIL import Image

# Define the Neural Network (same as in main.py)
class FashionNN(nn.Module):
    def __init__(self):
        super(FashionNN, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.model(x)

# Fashion MNIST class labels
class_labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Load or train the model
@st.cache_resource
def get_model(epochs, batch_size, learning_rate):
    # Data loading
    transform = transforms.ToTensor()
    train_data = datasets.FashionMNIST(root='data', train=True, download=True, transform=transform)
    test_data = datasets.FashionMNIST(root='data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=1000)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FashionNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training progress
    progress_bar = st.progress(0)
    loss_placeholder = st.empty()

    # Training loop
    for epoch in range(epochs):
        total_loss = 0
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            
            # Update progress bar during each batch
            progress = (epoch * len(train_loader) + i + 1) / (epochs * len(train_loader))
            progress_bar.progress(progress)
            
        loss_placeholder.text(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

    # Evaluation
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

    accuracy = 100 * correct / total
    st.success(f"Training completed! Accuracy on test set: {accuracy:.2f}%")

    return model, test_data, device, accuracy

# Function to make prediction on a single image
def predict_image(model, image_tensor, device):
    image_tensor = image_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image_tensor)
        _, prediction = torch.max(output, 1)
        probabilities = torch.nn.functional.softmax(output, dim=1)
    return prediction.item(), probabilities[0].tolist()

# Process uploaded image
def process_uploaded_image(upload):
    image = Image.open(upload).convert('L')  # Convert to grayscale
    image = image.resize((28, 28))
    
    # Convert to tensor
    transform = transforms.ToTensor()
    image_tensor = transform(image)
    
    return image_tensor, image

# Main Streamlit app
def main():
    st.title("Fashion MNIST Classifier")
    st.sidebar.title("Options")
    
    page = st.sidebar.selectbox(
        "Choose a page", 
        ["Model Overview", "Train Model", "Test on Sample Images", "Upload Your Own Image"]
    )
    
    if page == "Model Overview":
        st.header("Model Architecture")
        st.code("""
class FashionNN(nn.Module):
    def __init__(self):
        super(FashionNN, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.model(x)
        """)
        
        st.header("Fashion MNIST Dataset")
        st.write("The Fashion MNIST dataset consists of 70,000 grayscale images of fashion items (60,000 training, 10,000 testing)")
        st.write("Each image is 28x28 pixels")
        st.write("There are 10 categories:")
        
        for i, label in enumerate(class_labels):
            st.write(f"{i}: {label}")
            
    elif page == "Train Model":
        st.header("Train the Fashion MNIST Classifier")

        # Hyperparameters
        epochs = st.slider("Number of Epochs", min_value=1, max_value=10, value=3)
        batch_size = st.select_slider("Batch Size", options=[32, 64, 128, 256], value=64)
        learning_rate = st.select_slider(
            "Learning Rate", 
            options=[0.0001, 0.0005, 0.001, 0.005, 0.01], 
            value=0.001,
            format_func=lambda x: f"{x:.4f}"
        )
        
        if st.button("Train Model"):
            # Training will start and cache the model
            model, test_data, device, accuracy = get_model(epochs, batch_size, learning_rate)
    
    elif page == "Test on Sample Images":
        st.header("Test with Sample Images")
        
        # Training must have occurred first
        if 'model' not in st.session_state:
            # Default settings for quick model training if not yet available
            st.info("Model needs to be trained first. Training with default settings...")
            model, test_data, device, accuracy = get_model(epochs=2, batch_size=64, learning_rate=0.001)
            st.session_state.model = model
            st.session_state.test_data = test_data
            st.session_state.device = device
        else:
            model = st.session_state.model
            test_data = st.session_state.test_data
            device = st.session_state.device
        
        # Get random test images
        num_images = st.slider("Number of test images to display", 1, 9, 4)
        st.write("Randomly selected test images:")
        
        # Create a grid for displaying images
        cols = min(3, num_images)
        rows = (num_images + cols - 1) // cols
        grid = st.columns(cols)
        
        # Get random indices for the test data
        indices = torch.randperm(len(test_data))[:num_images]
        
        # Display each image with prediction
        for i, idx in enumerate(indices):
            image, label = test_data[idx]
            true_label = class_labels[label]
            
            # Make prediction
            pred_label_idx, probabilities = predict_image(model, image, device)
            pred_label = class_labels[pred_label_idx]
            
            # Display in grid
            with grid[i % cols]:
                plt.figure(figsize=(3, 3))
                plt.imshow(image.squeeze().numpy(), cmap='gray')
                plt.axis('off')
                
                # Save figure to buffer
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                
                # Display image and prediction
                st.image(buf, caption=f"Prediction: {pred_label}")
                st.write(f"True: {true_label}")
                
                # Display top 3 predictions with probabilities
                top3_probs, top3_idx = torch.tensor(probabilities).topk(3)
                st.write("Top predictions:")
                for j in range(3):
                    st.write(f"{class_labels[top3_idx[j]]}: {top3_probs[j]:.2%}")
                
                plt.close()
    
    elif page == "Upload Your Own Image":
        st.header("Upload Your Own Image")
        
        # Training must have occurred first
        if 'model' not in st.session_state:
            # Default settings for quick model training if not yet available
            st.info("Model needs to be trained first. Training with default settings...")
            model, test_data, device, accuracy = get_model(epochs=2, batch_size=64, learning_rate=0.001)
            st.session_state.model = model
            st.session_state.test_data = test_data
            st.session_state.device = device
        else:
            model = st.session_state.model
            test_data = st.session_state.test_data
            device = st.session_state.device
        
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            # Process the uploaded image
            image_tensor, displayed_image = process_uploaded_image(uploaded_file)
            
            # Display the processed image
            st.image(displayed_image, caption="Uploaded Image (Resized to 28x28)", width=200)
            
            # Make prediction
            prediction, probabilities = predict_image(model, image_tensor, device)
            
            # Display prediction results
            st.subheader(f"Prediction: {class_labels[prediction]}")
            
            # Display probability distribution
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.bar(class_labels, probabilities)
            ax.set_ylabel('Probability')
            ax.set_xlabel('Class')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig)

if __name__ == "__main__":
    main()
