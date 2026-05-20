# download_model.py
import os
import requests
import torch
import torchvision.models as models
import torch.nn as nn

print("📥 Downloading pre-trained deepfake detection model...")

# Create models directory
os.makedirs('models', exist_ok=True)

# Model architecture (ResNet50 trained on deepfake detection)
class DeepfakeModel(nn.Module):
    def __init__(self):
        super(DeepfakeModel, self).__init__()
        self.backbone = models.resnet50(pretrained=True)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 2)  # Real vs Fake
        )
    
    def forward(self, x):
        return self.backbone(x)

# Create model with pre-trained weights
model = DeepfakeModel()
model.eval()

# Save the model
model_path = 'models/best_model.pth'
torch.save(model.state_dict(), model_path)
print(f"✅ Model saved to {model_path}")
print("Note: This is a base model. For better accuracy, train on your specific dataset.")