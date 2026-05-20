"""
Proper Deepfake Detection Model Training
Using your organized dataset
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
import numpy as np
from tqdm import tqdm
import json
from sklearn.metrics import accuracy_score, classification_report

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Dataset class
class DeepfakeDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.images = []
        self.labels = []
        
        # real = 0, fake = 1
        for label, category in enumerate(['real', 'fake']):
            category_path = os.path.join(root_dir, category)
            if os.path.exists(category_path):
                files = [f for f in os.listdir(category_path) 
                        if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                for f in files:
                    self.images.append(os.path.join(category_path, f))
                    self.labels.append(label)
        
        print(f"Loaded {len(self.images)} images from {root_dir}")
        print(f"  Real: {self.labels.count(0)}")
        print(f"  Fake: {self.labels.count(1)}")
        
        self.transform = transform
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# Model architecture
class DeepfakeModel(nn.Module):
    def __init__(self, num_classes=2):
        super(DeepfakeModel, self).__init__()
        # Use pre-trained ResNet50 (more stable than EfficientNet for this task)
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        
        # Replace final layer
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 512),
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

def train_model():
    """Main training function"""
    
    # Paths
    project_root = r"E:\AI deepfake detector\AI-DEEPFAKE-DETECTOR"
    train_dir = os.path.join(project_root, 'datasets', 'organized', 'train')
    test_dir = os.path.join(project_root, 'datasets', 'organized', 'test')
    
    if not os.path.exists(train_dir):
        print(f"❌ Training directory not found!")
        print("Run: python main.py --organize first")
        return
    
    # Data transforms with augmentation (helps with generalization)
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load datasets
    print("\n📂 Loading training data...")
    train_dataset = DeepfakeDataset(train_dir, transform=train_transform)
    test_dataset = DeepfakeDataset(test_dir, transform=test_transform)
    
    # Use subset for faster training (adjust as needed)
    use_subset = True
    subset_size = 10000  # Use 10k images for training
    
    if use_subset and len(train_dataset) > subset_size:
        indices = np.random.choice(len(train_dataset), subset_size, replace=False)
        train_dataset = torch.utils.data.Subset(train_dataset, indices)
        print(f"Using subset of {subset_size} images for training")
    
    # Create data loaders
    batch_size = 64 if torch.cuda.is_available() else 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    print(f"\n📊 Training batches: {len(train_loader)}")
    print(f"📊 Test batches: {len(test_loader)}")
    
    # Initialize model
    model = DeepfakeModel(num_classes=2).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
    
    # Training parameters
    epochs = 5
    best_test_acc = 0
    
    print("\n🚀 Starting training...")
    print("="*60)
    
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
            
            train_pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*train_correct/train_total:.2f}%'
            })
        
        train_acc = 100. * train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        test_loss = 0
        test_correct = 0
        test_total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            test_pbar = tqdm(test_loader, desc=f'Epoch {epoch+1}/{epochs} [Test]')
            for images, labels in test_pbar:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                test_total += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                test_pbar.set_postfix({'acc': f'{100.*test_correct/test_total:.2f}%'})
        
        test_acc = 100. * test_correct / test_total
        avg_test_loss = test_loss / len(test_loader)
        
        # Update learning rate
        scheduler.step(avg_test_loss)
        
        # Print summary
        print(f"\n📊 Epoch {epoch+1}/{epochs} Summary:")
        print(f"   Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"   Test Loss: {avg_test_loss:.4f}, Test Acc: {test_acc:.2f}%")
        print(f"   Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            os.makedirs('models', exist_ok=True)
            model_path = os.path.join(project_root, 'models', 'best_model.pth')
            torch.save({
                'model_state_dict': model.state_dict(),
                'test_acc': test_acc,
                'epoch': epoch
            }, model_path)
            print(f"   💾 Saved best model! (Accuracy: {test_acc:.2f}%)")
        
        print("-"*60)
    
    print("\n" + "="*60)
    print("✅ TRAINING COMPLETE!")
    print("="*60)
    print(f"📊 Best Test Accuracy: {best_test_acc:.2f}%")
    print(f"💾 Model saved to: models/best_model.pth")
    print("\n🎯 Restart your web app to use the trained model!")
    
    return model

if __name__ == "__main__":
    train_model()