# test_model.py - Simple test for your deepfake detection model
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import sys

print("="*60)
print("🔍 AURORA MODEL TESTER")
print("="*60)

# ============================================
# MODEL ARCHITECTURE (Must match training)
# ============================================
class DeepfakeDetector(nn.Module):
    def __init__(self, num_classes=3, dropout=0.4):
        super().__init__()
        self.backbone = models.efficientnet_b4(weights=None)
        in_feat = self.backbone.classifier[1].in_features  # 1792
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=False),
            nn.Linear(in_feat, 512),
            nn.ReLU(),
            nn.Dropout(p=dropout / 2, inplace=False),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )
        self.num_classes = num_classes

    def forward(self, x):
        return self.backbone(x)


# ============================================
# FIND MODEL FILE
# ============================================
def find_model_file():
    """Search for model file in multiple locations"""
    possible_paths = [
        'models/best_model.pth',
        './models/best_model.pth',
        'E:/AI deepfake detector/AI-DEEPFAKE-DETECTOR/models/best_model.pth',
        'E:/AI deepfake detector/ai-deepfake-detector/models/best_model.pth',
        '../models/best_model.pth',
        'best_model.pth',
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    return None


# ============================================
# LOAD MODEL
# ============================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n📱 Device: {device}")

model_path = find_model_file()

if model_path is None:
    print("\n❌ MODEL NOT FOUND!")
    print("\nPlease place your trained model in one of these locations:")
    print("   - models/best_model.pth")
    print("   - ./models/best_model.pth")
    print("   - E:/AI deepfake detector/AI-DEEPFAKE-DETECTOR/models/best_model.pth")
    print("\n📥 If you haven't trained yet, run the Colab notebook first.")
    sys.exit(1)

print(f"\n✅ Found model: {model_path}")
file_size = os.path.getsize(model_path) / 1e6
print(f"   Size: {file_size:.1f} MB")

# Load model
print("\n📦 Loading model...")
model = DeepfakeDetector(num_classes=3).to(device)

try:
    checkpoint = torch.load(model_path, map_location=device)
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint.get('epoch', '?')
        accuracy = checkpoint.get('val_accuracy', 0)
        print(f"   Checkpoint info: Epoch {epoch}, Val Acc: {accuracy*100:.1f}%")
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    print("✅ Model loaded successfully!")

except Exception as e:
    print(f"❌ Error loading model: {e}")
    sys.exit(1)


# ============================================
# PREPROCESSING
# ============================================
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

CLASS_NAMES = ['Real Image', 'AI Generated', 'Deepfake']


# ============================================
# TEST WITH DUMMY INPUT
# ============================================
print("\n" + "="*60)
print("📊 TESTING WITH RANDOM INPUT")
print("="*60)

test_input = torch.randn(1, 3, 224, 224).to(device)
with torch.no_grad():
    output = model(test_input)
    probs = torch.softmax(output, dim=1).cpu().numpy()[0]

print("\nModel prediction on random noise:")
for i, name in enumerate(CLASS_NAMES):
    bar = '█' * int(probs[i] * 40)
    print(f"   {name:14s}: {probs[i]*100:6.2f}%  {bar}")

if max(probs) > 0.5:
    print(f"\n✅ Model is working! Best prediction: {CLASS_NAMES[probs.argmax()]}")
else:
    print("\n⚠️ Model predictions are weak. This may be normal for random input.")


# ============================================
# OPTIONAL: TEST WITH ACTUAL IMAGE
# ============================================
print("\n" + "="*60)
print("🖼️ TEST WITH ACTUAL IMAGE (Optional)")
print("="*60)

# Ask user if they want to test with an image
test_image = input("\nEnter path to an image to test (or press Enter to skip): ").strip()

if test_image and os.path.exists(test_image):
    try:
        print(f"\n📸 Analyzing: {test_image}")
        img = Image.open(test_image).convert('RGB')
        input_tensor = transform(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.softmax(output, dim=1).cpu().numpy()[0]
        
        print("\n📊 Results:")
        for i, name in enumerate(CLASS_NAMES):
            bar = '█' * int(probs[i] * 40)
            print(f"   {name:14s}: {probs[i]*100:6.2f}%  {bar}")
        
        predicted = CLASS_NAMES[probs.argmax()]
        confidence = max(probs) * 100
        print(f"\n🎯 Prediction: {predicted} ({confidence:.1f}% confidence)")
        
    except Exception as e:
        print(f"Error analyzing image: {e}")
elif test_image:
    print(f"File not found: {test_image}")

print("\n" + "="*60)
print("✅ Test complete!")
print("="*60)