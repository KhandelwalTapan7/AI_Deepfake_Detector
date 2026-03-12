import os
import sys
import json
import torch
import numpy as np
from PIL import Image
import base64
from io import BytesIO
import uuid

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model.model import DeepfakeDetector

def load_model(model_path, device='cpu'):
    """
    Load trained model
    """
    model = DeepfakeDetector(num_classes=3)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model

def preprocess_image(image, target_size=(224, 224)):
    """
    Preprocess image for model inference
    """
    from torchvision import transforms
    
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Convert to PIL Image if needed
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    elif isinstance(image, str):
        image = Image.open(image).convert('RGB')
    elif not isinstance(image, Image.Image):
        image = Image.open(image).convert('RGB')
    
    # Apply transforms
    image_tensor = transform(image)
    return image_tensor.unsqueeze(0)

def predict_image(model, image_tensor, device='cpu'):
    """
    Make prediction on single image
    """
    class_names = ['Real Image', 'AI Generated', 'Deepfake']
    colors = ['#4CAF50', '#FF9800', '#F44336']  # Green, Orange, Red
    
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        
        # Get prediction
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
        
        # Get all probabilities
        all_probs = probabilities[0].cpu().numpy()
        
        result = {
            'class': class_names[predicted_class],
            'confidence': round(confidence * 100, 2),
            'color': colors[predicted_class],
            'probabilities': {
                class_names[i]: round(prob * 100, 2) 
                for i, prob in enumerate(all_probs)
            }
        }
    
    return result

def image_to_base64(image):
    """
    Convert PIL Image to base64 string
    """
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

def base64_to_image(base64_string):
    """
    Convert base64 string to PIL Image
    """
    img_data = base64.b64decode(base64_string)
    img = Image.open(BytesIO(img_data))
    return img

def save_uploaded_file(uploaded_file, save_dir='uploads'):
    """
    Save uploaded file to disk
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Generate unique filename
    filename = str(uuid.uuid4()) + '_' + uploaded_file.filename
    filepath = os.path.join(save_dir, filename)
    
    # Save file
    uploaded_file.save(filepath)
    
    return filepath

def get_model_info():
    """
    Get information about the model
    """
    return {
        'name': 'Deepfake Detector v1.0',
        'classes': ['Real Image', 'AI Generated', 'Deepfake'],
        'input_size': '224x224',
        'framework': 'PyTorch',
        'architecture': 'ResNet50 with Attention'
    }

def calculate_confidence_color(confidence):
    """
    Calculate color based on confidence level
    """
    if confidence >= 80:
        return '#4CAF50'  # Green
    elif confidence >= 60:
        return '#FFC107'  # Yellow
    else:
        return '#F44336'  # Red