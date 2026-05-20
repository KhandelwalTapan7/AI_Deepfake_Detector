"""Quick training with smaller dataset - Only 1000 images per class"""
import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import random

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Quick dataset (only 1000 images per class)
class QuickDataset(Dataset):
    def __init__(self, root_dir, num_samples=1000):
        self.images = []
        self.labels = []
        
        for label, category in enumerate(['real', 'fake']):
            category_path = os.path.join(root_dir, category)
            if os.path.exists(category_path):
                files = [f for f in os.listdir(category_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                files = files[:num_samples]  # Take only first 1000
                for f in files:
                    self.images.append(os.path.join(category_path, f))
                    self.labels.append(label)
        
        print(f"Loaded {len(self.images)} images")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        return transform(img), self.labels[idx]

# Simple model (much smaller than EfficientNet)
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 2)
        )
    
    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

print("📂 Loading datasets (1000 images each)...")
train_dir = os.path.join('datasets', 'organized', 'train')
train_dataset = QuickDataset(train_dir, num_samples=1000)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

print("🏗️ Creating model...")
model = SimpleModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("🚀 Training (this will take 2-3 minutes)...")
model.train()
for epoch in range(5):
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f"Epoch {epoch+1}/5 - Loss: {running_loss/len(train_loader):.4f}, Accuracy: {accuracy:.2f}%")

# Save model
os.makedirs('models', exist_ok=True)
torch.save(model.state_dict(), 'models/best_model.pth')
print("\n✅ Model saved to models/best_model.pth")
print("🎯 Restart your web app and it will use this model!")