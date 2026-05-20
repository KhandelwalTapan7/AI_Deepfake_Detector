"""
Train Deepfake Detection Model
This script trains a model on your organized dataset
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
import numpy as np
from tqdm import tqdm
import json
from datetime import datetime

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class ImageDataset(Dataset):
    """Custom dataset for loading images"""
    
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []
        
        # Load real images (label 0)
        real_dir = os.path.join(root_dir, 'real')
        if os.path.exists(real_dir):
            for img_name in os.listdir(real_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.images.append(os.path.join(real_dir, img_name))
                    self.labels.append(0)  # Real = 0
        
        # Load fake images (label 1)
        fake_dir = os.path.join(root_dir, 'fake')
        if os.path.exists(fake_dir):
            for img_name in os.listdir(fake_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.images.append(os.path.join(fake_dir, img_name))
                    self.labels.append(1)  # Fake = 1
        
        print(f"Loaded {len(self.images)} images from {root_dir}")
        print(f"  Real: {self.labels.count(0)}")
        print(f"  Fake: {self.labels.count(1)}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

class DeepfakeModel(nn.Module):
    """Deepfake Detection Model using EfficientNet"""
    
    def __init__(self, num_classes=2):
        super(DeepfakeModel, self).__init__()
        # Use pre-trained EfficientNet
        self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        
        # Get the number of features
        num_features = self.backbone.classifier[1].in_features
        
        # Replace classifier for binary classification
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

def train_model(train_dir, val_dir, epochs=30, batch_size=32, learning_rate=0.001):
    """Train the deepfake detection model"""
    
    print("\n" + "="*60)
    print("🚀 STARTING MODEL TRAINING")
    print("="*60)
    
    # Data transforms with augmentation
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    print("\n📂 Loading datasets...")
    train_dataset = ImageDataset(train_dir, transform=train_transform)
    val_dataset = ImageDataset(val_dir, transform=val_transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    print(f"\n📊 Training batches: {len(train_loader)}")
    print(f"📊 Validation batches: {len(val_loader)}")
    
    # Initialize model
    model = DeepfakeModel(num_classes=2).to(device)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_val_acc = 0
    
    print("\n🎯 Starting training...")
    print("-"*60)
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]')
        for images, labels in train_pbar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            train_pbar.set_postfix({'loss': loss.item(), 'acc': 100.*train_correct/train_total})
        
        train_acc = 100. * train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{epochs} [Val]')
            for images, labels in val_pbar:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                
                val_pbar.set_postfix({'loss': loss.item(), 'acc': 100.*val_correct/val_total})
        
        val_acc = 100. * val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)
        
        # Update learning rate
        scheduler.step(avg_val_loss)
        
        # Save history
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_acc)
        
        # Print epoch summary
        print(f"\n📊 Epoch {epoch+1}/{epochs} Summary:")
        print(f"   Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"   Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"   Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs('models', exist_ok=True)
            model_path = os.path.join(project_root, 'models', 'best_model.pth')
            torch.save(model.state_dict(), model_path)
            print(f"   💾 Saved best model with accuracy: {val_acc:.2f}%")
        
        print("-"*60)
    
    # Save training history
    history_path = os.path.join(project_root, 'models', 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    print("\n" + "="*60)
    print("✅ TRAINING COMPLETE!")
    print("="*60)
    print(f"📊 Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"💾 Model saved to: {project_root}/models/best_model.pth")
    print(f"📈 Training history saved to: {history_path}")
    
    return model, history

def main():
    """Main training function"""
    
    # Set paths
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    train_dir = os.path.join(project_root, 'datasets', 'organized', 'train')
    val_dir = os.path.join(project_root, 'datasets', 'organized', 'test')
    
    # Check if directories exist
    if not os.path.exists(train_dir):
        print(f"❌ Training directory not found: {train_dir}")
        print("   Run 'python main.py --organize' first")
        return
    
    if not os.path.exists(val_dir):
        print(f"❌ Validation directory not found: {val_dir}")
        return
    
    # Training parameters
    epochs = 30
    batch_size = 64 if torch.cuda.is_available() else 32
    learning_rate = 0.001
    
    print(f"\n📋 Training Configuration:")
    print(f"   Epochs: {epochs}")
    print(f"   Batch Size: {batch_size}")
    print(f"   Learning Rate: {learning_rate}")
    print(f"   Device: {device}")
    
    # Train model
    model, history = train_model(
        train_dir=train_dir,
        val_dir=val_dir,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate
    )

if __name__ == "__main__":
    main()