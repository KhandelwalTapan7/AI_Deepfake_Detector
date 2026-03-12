import os
import sys
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(src_dir)
sys.path.insert(0, project_root)
sys.path.insert(0, src_dir)

print(f"📂 Project root: {project_root}")
print(f"📂 Current dir: {current_dir}")

# Try multiple import approaches
try:
    # Method 1: Direct import from src
    from src.model.model import DeepfakeDetectorLight
    print("✅ Imported DeepfakeDetectorLight from src.model")
except ImportError as e:
    print(f"⚠️ Import error: {e}")
    try:
        # Method 2: Import from model
        sys.path.insert(0, os.path.join(src_dir, 'model'))
        from model import DeepfakeDetectorLight
        print("✅ Imported DeepfakeDetectorLight from model")
    except ImportError as e2:
        print(f"⚠️ Second import error: {e2}")
        # Method 3: Define a simple model here as fallback
        print("⚠️ Using fallback model definition")
        
        class DeepfakeDetectorLight(nn.Module):
            def __init__(self, num_classes=3):
                super(DeepfakeDetectorLight, self).__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(3, 32, 3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2, 2),
                    nn.Conv2d(32, 64, 3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2, 2),
                    nn.Conv2d(64, 128, 3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2, 2),
                )
                self.classifier = nn.Sequential(
                    nn.Linear(128 * 28 * 28, 256),
                    nn.ReLU(),
                    nn.Linear(256, num_classes)
                )
            
            def forward(self, x):
                x = self.features(x)
                x = x.view(x.size(0), -1)
                x = self.classifier(x)
                return x

# Fixed preprocessing function - ensures float32 type
def simple_preprocess(image_path, img_size=224):
    """Fixed image preprocessing with correct data type"""
    try:
        # Open image
        if isinstance(image_path, str):
            img = Image.open(image_path).convert('RGB')
        else:
            img = image_path.convert('RGB')
        
        # Resize
        img = img.resize((img_size, img_size))
        
        # Convert to numpy array and ensure float32
        img_array = np.array(img).astype(np.float32) / 255.0
        
        # Normalize
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_array = (img_array - mean) / std
        
        # Convert to tensor (float32 by default in recent PyTorch)
        img_tensor = torch.from_numpy(img_array)
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)
        
        # Explicitly ensure float32
        img_tensor = img_tensor.to(torch.float32)
        
        return img_tensor
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

# Fixed prediction function
def simple_predict(model, image_tensor, device):
    """Fixed prediction with proper type handling"""
    class_names = ['Real Image', 'AI Generated', 'Deepfake']
    colors = ['#4CAF50', '#FF9800', '#F44336']
    
    try:
        with torch.no_grad():
            # Ensure image is on correct device and type
            image_tensor = image_tensor.to(device, dtype=torch.float32)
            
            # Get model output
            outputs = model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            
            # Convert to numpy with proper types
            predicted_class = int(torch.argmax(probabilities, dim=1).item())
            confidence = float(probabilities[0][predicted_class].item())
            all_probs = probabilities[0].cpu().numpy().astype(float)
            
            result = {
                'class': class_names[predicted_class],
                'confidence': round(confidence * 100, 2),
                'color': colors[predicted_class],
                'probabilities': {
                    class_names[i]: round(float(prob) * 100, 2) 
                    for i, prob in enumerate(all_probs)
                }
            }
        
        return result
    except Exception as e:
        print(f"Error in prediction: {e}")
        return None

def load_my_model(model_path, device):
    """Load the trained model"""
    try:
        print(f"📂 Loading model from: {model_path}")
        
        # Check if file exists
        if not os.path.exists(model_path):
            print(f"❌ Model file not found: {model_path}")
            return None
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        
        # Create model
        model = DeepfakeDetectorLight(num_classes=3)
        
        # Load weights
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        # Ensure model is in eval mode and float32
        model.to(device, dtype=torch.float32)
        model.eval()
        
        print(f"✅ Model loaded successfully!")
        return model
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return None

def analyze_predictions(image_folder, model_path=None):
    """Analyze predictions on a folder of images"""
    print(f"\n🔍 Analyzing predictions on: {image_folder}")
    
    # Initialize variables
    predictions = []
    confidences = []
    details = []
    counter = Counter()
    
    # Check if folder exists
    if not os.path.exists(image_folder):
        print(f"❌ Folder not found: {image_folder}")
        return counter, details
    
    # Set default model path if not provided
    if model_path is None:
        model_path = os.path.join(project_root, 'models', 'best_model.pth')
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️ Using device: {device}")
    
    model = load_my_model(model_path, device)
    if model is None:
        print("❌ Failed to load model. Exiting.")
        return counter, details
    
    # Get all images
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.jfif')
    images = [f for f in os.listdir(image_folder) 
              if f.lower().endswith(image_extensions)]
    
    if len(images) == 0:
        print(f"❌ No images found in {image_folder}")
        return counter, details
    
    print(f"📸 Found {len(images)} images")
    analyze_count = min(20, len(images))
    print(f"🔬 Analyzing first {analyze_count} images...")
    
    successful_predictions = 0
    
    for i, img_file in enumerate(images[:analyze_count]):
        img_path = os.path.join(image_folder, img_file)
        
        try:
            print(f"\n🖼️  Processing {i+1}/{analyze_count}: {img_file}")
            
            # Preprocess
            image_tensor = simple_preprocess(img_path)
            if image_tensor is None:
                print(f"   ⚠️ Failed to preprocess")
                continue
            
            print(f"   Tensor shape: {image_tensor.shape}, dtype: {image_tensor.dtype}")
            
            # Predict
            result = simple_predict(model, image_tensor, device)
            if result is None:
                print(f"   ⚠️ Failed to predict")
                continue
            
            predictions.append(result['class'])
            confidences.append(result['confidence'])
            details.append({
                'file': img_file,
                'result': result
            })
            
            # Print result with nice formatting
            print(f"   ✅ Prediction: {result['class']}")
            print(f"   📊 Confidence: {result['confidence']}%")
            print(f"   📈 Probabilities:")
            for class_name, prob in result['probabilities'].items():
                bar = "█" * int(prob/5)
                print(f"      {class_name:15}: {prob:5.1f}% {bar}")
            
            successful_predictions += 1
            
        except Exception as e:
            print(f"   ❌ Error on {img_file}: {e}")
            import traceback
            traceback.print_exc()
    
    # Show statistics
    print("\n" + "="*60)
    print("📊 PREDICTION STATISTICS")
    print("="*60)
    
    if predictions:
        counter = Counter(predictions)
        for pred, count in counter.items():
            percentage = (count / len(predictions)) * 100
            print(f"   {pred:15}: {count:2d} images ({percentage:5.1f}%)")
        
        avg_confidence = np.mean(confidences) if confidences else 0
        print(f"\n   Average confidence: {avg_confidence:.1f}%")
        print(f"   Successful predictions: {successful_predictions}/{analyze_count}")
        
        # Try to create a chart
        try:
            plt.figure(figsize=(10, 6))
            plt.bar(counter.keys(), counter.values())
            plt.title('Prediction Distribution')
            plt.xlabel('Class')
            plt.ylabel('Count')
            plt.xticks(rotation=45)
            plt.tight_layout()
            chart_path = os.path.join(project_root, 'prediction_analysis.png')
            plt.savefig(chart_path)
            print(f"\n📊 Chart saved to: {chart_path}")
            plt.show()
        except Exception as e:
            print(f"Could not create chart: {e}")
    else:
        print("   ❌ No predictions were made successfully")
        print("   Possible issues:")
        print("   - Images might be corrupted")
        print("   - Model might be incompatible")
        print("   - Data type mismatch (fixed in this version)")
    
    return counter, details

def main():
    print("="*60)
    print("🔍 DEEPFAKE DETECTOR DIAGNOSTIC TOOL")
    print("="*60)
    print(f"Project root: {project_root}")
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print("="*60)
    
    # Check for model
    default_model = os.path.join(project_root, 'models', 'best_model.pth')
    if os.path.exists(default_model):
        print(f"✅ Model found: {default_model}")
        model_size = os.path.getsize(default_model) / (1024*1024)
        print(f"   Size: {model_size:.2f} MB")
    else:
        print(f"⚠️ Default model not found at: {default_model}")
    
    # Get folder path
    print("\n📁 Enter path to folder with test images:")
    print("   (You can use absolute path or relative to project root)")
    print("   Examples:")
    print("   - real_img")
    print("   - C:\\Users\\YourName\\Pictures")
    print("   - ..\\..\\some\\folder")
    
    folder = input("\n📁 Path: ").strip()
    
    # If relative path, make it absolute from project root
    if not os.path.isabs(folder):
        folder = os.path.join(project_root, folder)
    
    # Optional: specify model path
    model_path = input("\n📁 Enter model path (or press Enter for default): ").strip()
    if model_path and not os.path.isabs(model_path):
        model_path = os.path.join(project_root, model_path)
    elif not model_path:
        model_path = default_model
    
    # Run analysis
    analyze_predictions(folder, model_path)

if __name__ == "__main__":
    main()