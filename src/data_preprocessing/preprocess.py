import os
import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Move ImageDataset to top level (outside any function)
class ImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        try:
            image = Image.open(self.image_paths[idx]).convert('RGB')
            label = self.labels[idx]
            
            if self.transform:
                image = self.transform(image)
            
            return image, label
        except Exception as e:
            print(f"Error loading image {self.image_paths[idx]}: {e}")
            # Return a dummy tensor and label (you might want to handle this differently)
            return torch.zeros(3, 224, 224), self.labels[idx]

class ImagePreprocessor:
    def __init__(self, img_size=224):
        self.img_size = img_size
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def preprocess_image(self, image_path):
        """
        Preprocess a single image for model inference
        """
        try:
            # Load image
            image = Image.open(image_path).convert('RGB')
            # Apply transforms
            image_tensor = self.transform(image)
            return image_tensor.unsqueeze(0)  # Add batch dimension
        except Exception as e:
            print(f"Error preprocessing {image_path}: {e}")
            return None
    
    def preprocess_batch(self, image_paths):
        """
        Preprocess multiple images
        """
        batch_tensors = []
        valid_paths = []
        
        for path in tqdm(image_paths, desc="Preprocessing images"):
            tensor = self.preprocess_image(path)
            if tensor is not None:
                batch_tensors.append(tensor.squeeze(0))
                valid_paths.append(path)
        
        if batch_tensors:
            return torch.stack(batch_tensors), valid_paths
        return None, []
    
    def extract_face(self, image_path):
        """
        Extract face from image using OpenCV (useful for deepfake detection)
        """
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                return None
            
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Load face cascade classifier
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 
                                                'haarcascade_frontalface_default.xml')
            
            # Detect faces
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) > 0:
                # Extract the first face
                x, y, w, h = faces[0]
                face = image[y:y+h, x:x+w]
                face = cv2.resize(face, (self.img_size, self.img_size))
                return Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
            
            return None
        except Exception as e:
            print(f"Error extracting face: {e}")
            return None

def create_data_loaders(data_dir, batch_size=32, img_size=224, num_workers=0):
    """
    Create data loaders for training and validation
    Set num_workers=0 for Windows to avoid multiprocessing issues
    """
    
    # Collect all images and labels
    image_paths = []
    labels = []
    class_names = ['real', 'ai_generated', 'deepfake']
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    
    print("Scanning for images...")
    for class_name in class_names:
        class_dir = os.path.join(data_dir, class_name)
        if os.path.exists(class_dir):
            class_images = []
            for img_file in os.listdir(class_dir):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    class_images.append(os.path.join(class_dir, img_file))
            
            print(f"   Found {len(class_images)} {class_name} images")
            image_paths.extend(class_images)
            labels.extend([class_to_idx[class_name]] * len(class_images))
    
    if len(image_paths) == 0:
        raise ValueError(f"No images found in {data_dir}")
    
    print(f"Total images: {len(image_paths)}")
    
    # Split into train and validation
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    print(f"Training samples: {len(train_paths)}")
    print(f"Validation samples: {len(val_paths)}")
    
    # Define transforms
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = ImageDataset(train_paths, train_labels, train_transform)
    val_dataset = ImageDataset(val_paths, val_labels, val_transform)
    
    # Create data loaders with num_workers=0 for Windows compatibility
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=0,  # Set to 0 for Windows
        pin_memory=False  # Set to False for CPU
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0,  # Set to 0 for Windows
        pin_memory=False  # Set to False for CPU
    )
    
    return train_loader, val_loader, class_names