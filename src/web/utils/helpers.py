import os
import sys
import torch
from PIL import Image
import numpy as np
import random
import json

# Fix path calculation
current_dir = os.path.dirname(os.path.abspath(__file__))
web_dir = os.path.dirname(current_dir)
src_dir = os.path.dirname(web_dir)
project_root = os.path.dirname(src_dir)

# Add paths
sys.path.insert(0, project_root)
sys.path.insert(0, src_dir)

# Import the correct model based on what you trained
from src.model.model import DeepfakeDetectorLight

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

class WebAppHelper:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.model_path = os.path.join(project_root, 'models', 'best_model.pth')
        self.class_names = ['Real Image', 'AI Generated', 'Deepfake']
        self.colors = ['#4CAF50', '#FF9800', '#F44336']
        
        # Image preprocessing settings
        self.img_size = 224
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        
        self.load_model()
    
    def load_model(self):
        """Load the trained model"""
        try:
            if os.path.exists(self.model_path):
                print(f"📂 Loading model from: {self.model_path}")
                
                # Load checkpoint
                checkpoint = torch.load(self.model_path, map_location=self.device)
                
                # Create the light model
                self.model = DeepfakeDetectorLight(num_classes=3)
                
                # Load weights based on checkpoint format
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    self.model.load_state_dict(checkpoint)
                
                self.model.to(self.device)
                self.model.eval()
                print(f"✅ Model loaded successfully!")
                print(f"   Model type: DeepfakeDetectorLight")
                print(f"   Device: {self.device}")
            else:
                print(f"⚠️ Model not found at {self.model_path}")
                print("   Using mock results for demonstration")
                self.model = None
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("   Using mock results for demonstration")
            self.model = None
    
    def preprocess_image(self, image):
        """Preprocess image for model inference"""
        try:
            from torchvision import transforms
            
            # Define transforms
            transform = transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std)
            ])
            
            # Apply transforms
            image_tensor = transform(image)
            return image_tensor.unsqueeze(0)  # Add batch dimension
            
        except Exception as e:
            print(f"Error preprocessing image: {e}")
            return None
    
    def predict(self, image_tensor):
        """Make prediction on preprocessed image"""
        if self.model is None:
            return self.get_mock_result()
        
        with torch.no_grad():
            image_tensor = image_tensor.to(self.device)
            outputs = self.model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            
            # Get prediction - convert to Python native types
            predicted_class = int(torch.argmax(probabilities, dim=1).item())
            confidence = float(probabilities[0][predicted_class].item())
            
            # Get all probabilities - convert to Python float
            all_probs = probabilities[0].cpu().numpy()
            
            # Convert numpy values to Python native types
            result = {
                'class': self.class_names[predicted_class],
                'confidence': round(float(confidence) * 100, 2),
                'color': self.colors[predicted_class],
                'probabilities': {
                    self.class_names[i]: round(float(prob) * 100, 2) 
                    for i, prob in enumerate(all_probs)
                }
            }
        
        return result
    
    def analyze_image(self, image_file):
        """
        Analyze an uploaded image
        """
        try:
            # Open image
            if isinstance(image_file, str):
                image = Image.open(image_file).convert('RGB')
            else:
                # Handle file upload object
                image = Image.open(image_file).convert('RGB')
            
            # Preprocess
            image_tensor = self.preprocess_image(image)
            
            if image_tensor is None:
                return {
                    'error': 'Failed to preprocess image',
                    'class': 'Error',
                    'confidence': 0,
                    'color': '#9E9E9E'
                }
            
            # Make prediction
            result = self.predict(image_tensor)
            
            # Add image info - ensure all values are JSON serializable
            result['image_size'] = str(f"{image.size[0]}x{image.size[1]}")
            result['image_format'] = str(image.format) if image.format else 'Unknown'
            
            return result
            
        except Exception as e:
            print(f"Error in analyze_image: {e}")
            return {
                'error': str(e),
                'class': 'Error',
                'confidence': 0,
                'color': '#9E9E9E'
            }
    
    def get_mock_result(self):
        """Return mock result for demonstration with proper JSON serializable types"""
        idx = random.randint(0, 2)
        
        # Generate realistic mock probabilities - ensure Python float
        probs = [float(random.uniform(0, 100)) for _ in range(3)]
        # Normalize to sum to 100
        total = sum(probs)
        probs = [round(p * 100 / total, 2) for p in probs]
        
        return {
            'class': self.class_names[idx],
            'confidence': float(probs[idx]),
            'color': self.colors[idx],
            'probabilities': {
                self.class_names[i]: float(probs[i]) for i in range(3)
            },
            'image_size': '224x224',
            'image_format': 'PNG',
            'note': '⚠️ Using mock results - Train the model for actual predictions!'
        }
    
    def batch_analyze(self, image_files):
        """
        Analyze multiple images
        """
        results = []
        for image_file in image_files:
            result = self.analyze_image(image_file)
            result['filename'] = str(image_file.filename)
            results.append(result)
        
        return results

# Create singleton instance
webapp_helper = WebAppHelper()

# Helper function to get JSON serializable dict
def make_json_serializable(obj):
    """Convert any object to JSON serializable format"""
    if isinstance(obj, dict):
        return {key: make_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, torch.Tensor):
        return obj.cpu().detach().numpy().tolist()
    else:
        return obj