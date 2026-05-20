import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class DeepfakeDetector(nn.Module):
    """
    ResNet50 backbone with 3-class head.
    Matches the architecture used during Colab training.
    0 = Real Image | 1 = AI Generated | 2 = Deepfake
    """
    def __init__(self, num_classes=3, dropout_rate=0.4, pretrained=False):
        super().__init__()
        self.backbone = models.resnet50(weights=None)

        # Replace FC with same structure used in Colab training
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate / 2),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

        self.num_classes = num_classes
        self.class_names = ['Real Image', 'AI Generated', 'Deepfake']

    def forward(self, x):
        return self.backbone(x)


def get_model(num_classes=3, pretrained=True):
    model = DeepfakeDetector(num_classes=num_classes)
    return model


def load_model(model_path, device='cpu'):
    model = DeepfakeDetector(num_classes=3, pretrained=False)
    checkpoint = torch.load(model_path, map_location=device)

    # Handle both raw state_dict and wrapped checkpoint
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def save_model(model, path, optimizer=None, epoch=None, val_acc=None):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'num_classes': model.num_classes,
        'class_names': model.class_names,
    }
    if optimizer: checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    if epoch is not None: checkpoint['epoch'] = epoch
    if val_acc is not None: checkpoint['val_accuracy'] = val_acc
    torch.save(checkpoint, path)
    print(f'Model saved to {path}')