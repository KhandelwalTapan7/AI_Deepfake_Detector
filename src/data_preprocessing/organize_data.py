import os
import shutil
import random
from tqdm import tqdm

def organize_datasets():
    """
    Organize both AI Generated and Deepfake datasets into a unified structure
    """
    # Define paths
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ai_train_fake = os.path.join(base_dir, "AI Generated dataset", "train", "FAKE")
    ai_train_real = os.path.join(base_dir, "AI Generated dataset", "train", "REAL")
    ai_test_fake = os.path.join(base_dir, "AI Generated dataset", "test", "FAKE")
    ai_test_real = os.path.join(base_dir, "AI Generated dataset", "test", "REAL")
    
    deepfake_dir = os.path.join(base_dir, "Deepfake dataset", "cropped_images")
    
    # Create unified dataset structure
    unified_dir = os.path.join(base_dir, "unified_dataset")
    train_dir = os.path.join(unified_dir, "train")
    test_dir = os.path.join(unified_dir, "test")
    
    # Create class folders
    for split in [train_dir, test_dir]:
        for class_name in ['real', 'ai_generated', 'deepfake']:
            os.makedirs(os.path.join(split, class_name), exist_ok=True)
    
    print("📁 Organizing datasets...")
    
    # 1. Copy REAL images from AI dataset
    print("Copying REAL images from AI dataset...")
    for split_name, source_dir in [('train', ai_train_real), ('test', ai_test_real)]:
        dest_dir = os.path.join(unified_dir, split_name, 'real')
        if os.path.exists(source_dir):
            for img_file in tqdm(os.listdir(source_dir)):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    src = os.path.join(source_dir, img_file)
                    dst = os.path.join(dest_dir, f"real_{split_name}_{img_file}")
                    shutil.copy2(src, dst)
    
    # 2. Copy AI-generated FAKE images
    print("Copying AI-generated FAKE images...")
    for split_name, source_dir in [('train', ai_train_fake), ('test', ai_test_fake)]:
        dest_dir = os.path.join(unified_dir, split_name, 'ai_generated')
        if os.path.exists(source_dir):
            for img_file in tqdm(os.listdir(source_dir)):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    src = os.path.join(source_dir, img_file)
                    dst = os.path.join(dest_dir, f"ai_{split_name}_{img_file}")
                    shutil.copy2(src, dst)
    
    # 3. Copy Deepfake images (split into train/test)
    print("Copying Deepfake images...")
    if os.path.exists(deepfake_dir):
        all_deepfake_images = []
        for folder in os.listdir(deepfake_dir):
            folder_path = os.path.join(deepfake_dir, folder)
            if os.path.isdir(folder_path):
                for img_file in os.listdir(folder_path):
                    if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        all_deepfake_images.append((folder, img_file))
        
        # Shuffle and split
        random.shuffle(all_deepfake_images)
        split_idx = int(0.8 * len(all_deepfake_images))
        train_images = all_deepfake_images[:split_idx]
        test_images = all_deepfake_images[split_idx:]
        
        # Copy training deepfakes
        for folder, img_file in tqdm(train_images):
            src = os.path.join(deepfake_dir, folder, img_file)
            dst = os.path.join(train_dir, 'deepfake', f"deepfake_train_{folder}_{img_file}")
            shutil.copy2(src, dst)
        
        # Copy testing deepfakes
        for folder, img_file in tqdm(test_images):
            src = os.path.join(deepfake_dir, folder, img_file)
            dst = os.path.join(test_dir, 'deepfake', f"deepfake_test_{folder}_{img_file}")
            shutil.copy2(src, dst)
    
    print(f"✅ Dataset organization complete!")
    print(f"   Training set location: {train_dir}")
    print(f"   Testing set location: {test_dir}")
    
    # Print statistics
    print("\n📊 Dataset Statistics:")
    for split in ['train', 'test']:
        print(f"\n{split.upper()}:")
        for class_name in ['real', 'ai_generated', 'deepfake']:
            class_path = os.path.join(unified_dir, split, class_name)
            if os.path.exists(class_path):
                num_images = len(os.listdir(class_path))
                print(f"   {class_name}: {num_images} images")

if __name__ == "__main__":
    organize_datasets()