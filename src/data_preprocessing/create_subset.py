import os
import shutil
import random
from tqdm import tqdm

def create_small_subset(original_dir, subset_dir, samples_per_class=1000):
    """
    Create a smaller subset of the dataset for testing
    """
    print(f"\n📁 Creating small subset with {samples_per_class} samples per class...")
    
    # Create subset directory
    os.makedirs(subset_dir, exist_ok=True)
    
    class_names = ['real', 'ai_generated', 'deepfake']
    total_copied = 0
    
    for class_name in class_names:
        class_dir = os.path.join(original_dir, class_name)
        subset_class_dir = os.path.join(subset_dir, class_name)
        os.makedirs(subset_class_dir, exist_ok=True)
        
        if os.path.exists(class_dir):
            # Get all images
            images = [f for f in os.listdir(class_dir) 
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            # Randomly select samples
            num_samples = min(samples_per_class, len(images))
            selected = random.sample(images, num_samples)
            
            # Copy selected images
            for img in tqdm(selected, desc=f"Copying {class_name}"):
                src = os.path.join(class_dir, img)
                dst = os.path.join(subset_class_dir, img)
                shutil.copy2(src, dst)
            
            print(f"   ✅ Copied {len(selected)} {class_name} images")
            total_copied += len(selected)
    
    print(f"\n✅ Subset created successfully!")
    print(f"   Location: {subset_dir}")
    print(f"   Total images: {total_copied}")
    print(f"   Classes: {', '.join(class_names)}")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    original_dir = os.path.join(base_dir, "unified_dataset", "train")
    subset_dir = os.path.join(base_dir, "unified_dataset", "train_subset")
    
    create_small_subset(original_dir, subset_dir, samples_per_class=1000)