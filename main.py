# main.py

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import joblib

# Define a simple Neural Network
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

# Training setup
def train():
    # Data loading
    transform = transforms.ToTensor()
    train_data = datasets.FashionMNIST(root='data', train=True, download=True, transform=transform)
    test_data = datasets.FashionMNIST(root='data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=1000)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FashionNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(3):
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

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

    print(f"Accuracy on test set: {100 * correct / total:.2f}%")
    
    # Save the trained model
    joblib.dump(model, 'model.joblib')
    print("Model saved as model.joblib")

if __name__ == "__main__":
    train()
