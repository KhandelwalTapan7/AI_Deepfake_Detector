import os
import sys
import torch
from PIL import Image
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.helpers import load_model, preprocess_image, predict_image

def test_single_image(image_path):
    """Test a single image"""
    print(f"\n🔍 Testing image: {image_path}")
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = os.path.join('models', 'best_model.pth')
    model = load_model(model_path, device)
    
    # Preprocess and predict
    image_tensor = preprocess_image(image_path)
    result = predict_image(model, image_tensor, device)
    
    # Display results
    print("\n📊 Results:")
    print(f"   Class: {result['class']}")
    print(f"   Confidence: {result['confidence']}%")
    print("\n   Probabilities:")
    for class_name, prob in result['probabilities'].items():
        print(f"      {class_name}: {prob}%")
    
    # Show image
    img = Image.open(image_path)
    plt.imshow(img)
    plt.title(f"Prediction: {result['class']} ({result['confidence']}%)")
    plt.axis('off')
    plt.show()
    
    return result

if __name__ == "__main__":
    # Test with your own image
    image_path = input("Enter path to image: ")
    test_single_image(image_path)