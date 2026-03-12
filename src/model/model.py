import torch
import torch.nn as nn
import torchvision.models as models

class DeepfakeDetector(nn.Module):
    """
    Custom model for detecting real, AI-generated, and deepfake images
    Uses ResNet50 backbone with attention mechanism
    """
    def __init__(self, num_classes=3):
        super(DeepfakeDetector, self).__init__()
        
        # Load pre-trained ResNet50
        # Updated to use weights parameter instead of pretrained
        try:
            # For newer PyTorch versions
            self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        except:
            # Fallback for older versions
            self.backbone = models.resnet50(pretrained=True)
        
        # Get the feature dimension
        self.feature_dim = self.backbone.fc.in_features
        
        # Remove the original classifier
        self.backbone.fc = nn.Identity()
        
        # Attention mechanism for focusing on suspicious regions
        self.attention = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Custom classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize the weights of the custom layers"""
        for m in self.attention.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Extract features from backbone
        # ResNet50 forward pass until layer4
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        features = self.backbone.layer4(x)  # Shape: [batch, 2048, H, W]
        
        # Apply attention
        attention_weights = self.attention(features)  # Shape: [batch, 1, H, W]
        attended_features = features * attention_weights
        
        # Global average pooling
        pooled = attended_features.mean([2, 3])  # Shape: [batch, 2048]
        
        # Classification
        output = self.classifier(pooled)
        
        return output
    
    def predict_proba(self, x):
        """
        Get prediction probabilities
        """
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = torch.softmax(logits, dim=1)
        return probabilities


class DeepfakeDetectorEfficientNet(nn.Module):
    """
    Alternative model using EfficientNet for better performance
    """
    def __init__(self, num_classes=3):
        super(DeepfakeDetectorEfficientNet, self).__init__()
        
        # Load pre-trained EfficientNet
        try:
            # For newer PyTorch versions
            self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        except:
            # Fallback for older versions
            self.backbone = models.efficientnet_b0(pretrained=True)
        
        # Get the feature dimension
        self.feature_dim = self.backbone.classifier[1].in_features
        
        # Remove the original classifier
        self.backbone.classifier = nn.Identity()
        
        # Get the number of channels from the last convolutional layer
        # For EfficientNet-B0, the last features have 1280 channels
        self.last_channels = 1280
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Conv2d(self.last_channels, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Custom classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize the weights of the custom layers"""
        for m in self.attention.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Extract features
        features = self.backbone.features(x)  # Shape: [batch, 1280, H, W]
        
        # Apply attention
        attention_weights = self.attention(features)
        attended_features = features * attention_weights
        
        # Global average pooling
        pooled = attended_features.mean([2, 3])
        
        # Classification
        output = self.classifier(pooled)
        
        return output
    
    def predict_proba(self, x):
        """
        Get prediction probabilities
        """
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = torch.softmax(logits, dim=1)
        return probabilities


class DeepfakeDetectorLight(nn.Module):
    """
    Lightweight model for CPU training and faster inference
    """
    def __init__(self, num_classes=3, input_size=224):
        super(DeepfakeDetectorLight, self).__init__()
        
        # Simple but effective CNN architecture
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        
        # Calculate the size after convolutions
        # For 224x224 input: after 4 pools -> 14x14
        self.feature_size = 256 * 14 * 14
        
        # Classifier with attention-like mechanism
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.feature_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize the weights using He initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
    def predict_proba(self, x):
        """
        Get prediction probabilities
        """
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = torch.softmax(logits, dim=1)
        return probabilities


class DeepfakeDetectorEnsemble(nn.Module):
    """
    Ensemble of multiple models for better accuracy
    """
    def __init__(self, num_classes=3):
        super(DeepfakeDetectorEnsemble, self).__init__()
        
        # Create multiple models
        self.model1 = DeepfakeDetector(num_classes)
        self.model2 = DeepfakeDetectorEfficientNet(num_classes)
        self.model3 = DeepfakeDetectorLight(num_classes)
        
        # Fusion layer
        self.fusion = nn.Linear(num_classes * 3, num_classes)
        
    def forward(self, x):
        # Get predictions from each model
        out1 = self.model1(x)
        out2 = self.model2(x)
        out3 = self.model3(x)
        
        # Concatenate and fuse
        combined = torch.cat([out1, out2, out3], dim=1)
        output = self.fusion(combined)
        
        return output
    
    def predict_proba(self, x):
        """
        Get prediction probabilities
        """
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = torch.softmax(logits, dim=1)
        return probabilities


# Factory function to easily create models
def create_model(model_type='resnet50', num_classes=3, pretrained=True):
    """
    Factory function to create models
    
    Args:
        model_type: 'resnet50', 'efficientnet', 'light', or 'ensemble'
        num_classes: number of output classes
        pretrained: whether to use pretrained weights (for resnet50 and efficientnet)
    
    Returns:
        model: PyTorch model
    """
    if model_type == 'resnet50':
        return DeepfakeDetector(num_classes)
    elif model_type == 'efficientnet':
        return DeepfakeDetectorEfficientNet(num_classes)
    elif model_type == 'light':
        return DeepfakeDetectorLight(num_classes)
    elif model_type == 'ensemble':
        return DeepfakeDetectorEnsemble(num_classes)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

# For backward compatibility
DeepfakeDetectorResNet = DeepfakeDetector