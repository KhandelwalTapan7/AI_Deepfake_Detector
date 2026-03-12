import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import seaborn as sns

# Add the project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

# Now import using absolute paths
from src.model.model import create_model
from src.data_preprocessing.preprocess import create_data_loaders

class ModelTrainer:
    def __init__(self, model, device, save_dir='models'):
        self.model = model
        self.device = device
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        
        # Fix: Remove 'verbose' parameter completely
        # Different PyTorch versions have different parameter names
        try:
            # Try without verbose first (newer versions)
            self.scheduler = ReduceLROnPlateau(
                self.optimizer, 
                mode='min', 
                factor=0.5, 
                patience=3
            )
        except TypeError:
            try:
                # Try with print_lr (some versions)
                self.scheduler = ReduceLROnPlateau(
                    self.optimizer, 
                    mode='min', 
                    factor=0.5, 
                    patience=3,
                    print_lr=True
                )
            except TypeError:
                # Fallback to verbose (older versions)
                self.scheduler = ReduceLROnPlateau(
                    self.optimizer, 
                    mode='min', 
                    factor=0.5, 
                    patience=3,
                    verbose=True
                )
        
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        
        print(f"   Using scheduler: {type(self.scheduler).__name__}")
    
    def train_epoch(self, train_loader):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc='Training')
        for batch_idx, (images, labels) in enumerate(pbar):
            try:
                images, labels = images.to(self.device), labels.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                # Update progress bar
                pbar.set_postfix({
                    'Loss': f'{running_loss/(batch_idx+1):.4f}', 
                    'Acc': f'{100.*correct/total:.2f}%'
                })
                
            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                continue
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        return epoch_loss, epoch_acc
    
    def validate(self, val_loader):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc='Validation')
            for batch_idx, (images, labels) in enumerate(pbar):
                try:
                    images, labels = images.to(self.device), labels.to(self.device)
                    
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                    
                    running_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()
                    
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    
                    pbar.set_postfix({
                        'Loss': f'{running_loss/(batch_idx+1):.4f}', 
                        'Acc': f'{100.*correct/total:.2f}%'
                    })
                    
                except Exception as e:
                    print(f"Error in validation batch {batch_idx}: {e}")
                    continue
        
        epoch_loss = running_loss / len(val_loader)
        epoch_acc = 100. * correct / total
        return epoch_loss, epoch_acc, all_preds, all_labels
    
    def train(self, train_loader, val_loader, epochs=20, class_names=None):
        print(f"🚀 Starting training on {self.device}")
        print(f"   Training samples: {len(train_loader.dataset)}")
        print(f"   Validation samples: {len(val_loader.dataset)}")
        print(f"   Number of epochs: {epochs}")
        
        best_val_acc = 0.0
        
        for epoch in range(epochs):
            print(f"\n📅 Epoch {epoch+1}/{epochs}")
            print("-" * 50)
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)
            
            # Validate
            val_loss, val_acc, preds, labels = self.validate(val_loader)
            self.val_losses.append(val_loss)
            self.val_accs.append(val_acc)
            
            # Scheduler step
            self.scheduler.step(val_loss)
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            print(f"\n📊 Results:")
            print(f"   Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"   Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            print(f"   Learning Rate: {current_lr:.6f}")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save_model('best_model.pth')
                print(f"   💾 Saved best model with accuracy: {val_acc:.2f}%")
                
                # Generate classification report for best model
                if class_names and len(preds) > 0 and len(labels) > 0:
                    try:
                        print("\n📋 Classification Report:")
                        print(classification_report(labels, preds, 
                                                  target_names=class_names,
                                                  zero_division=0))
                    except Exception as e:
                        print(f"Could not generate classification report: {e}")
        
        # Save final model
        self.save_model('final_model.pth')
        self.plot_training_history()
        
        return self.model
    
    def save_model(self, filename):
        path = os.path.join(self.save_dir, filename)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accs': self.train_accs,
            'val_accs': self.val_accs,
        }, path)
        print(f"   💾 Model saved to {path}")
    
    def plot_training_history(self):
        """Plot training and validation curves"""
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            
            # Plot losses
            ax1.plot(self.train_losses, label='Train Loss')
            ax1.plot(self.val_losses, label='Val Loss')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.set_title('Training and Validation Loss')
            ax1.legend()
            ax1.grid(True)
            
            # Plot accuracies
            ax2.plot(self.train_accs, label='Train Acc')
            ax2.plot(self.val_accs, label='Val Acc')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Accuracy (%)')
            ax2.set_title('Training and Validation Accuracy')
            ax2.legend()
            ax2.grid(True)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.save_dir, 'training_history.png'))
            plt.show()
        except Exception as e:
            print(f"Could not plot training history: {e}")

def main():
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Paths
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Check if we should use the full dataset or a subset
    use_subset = True  # Set to False when you want to train on full dataset
    if use_subset:
        unified_data_dir = os.path.join(base_dir, "unified_dataset", "train_subset")
        # Create subset if it doesn't exist
        if not os.path.exists(unified_data_dir):
            print("📁 Creating subset dataset first...")
            try:
                from src.data_preprocessing.create_subset import create_small_subset
                original_dir = os.path.join(base_dir, "unified_dataset", "train")
                create_small_subset(original_dir, unified_data_dir, samples_per_class=1000)
            except ImportError:
                print("⚠️  create_subset module not found. Using full dataset.")
                unified_data_dir = os.path.join(base_dir, "unified_dataset", "train")
    else:
        unified_data_dir = os.path.join(base_dir, "unified_dataset", "train")
    
    # Check if unified dataset exists
    if not os.path.exists(unified_data_dir):
        print(f"❌ Unified dataset not found at {unified_data_dir}")
        print("   Please run 'python main.py --organize' first to organize your datasets.")
        return
    
    # Create data loaders with num_workers=0 for Windows
    print("📦 Creating data loaders...")
    try:
        # Use smaller batch size for CPU training to avoid memory issues
        batch_size = 16 if device.type == 'cpu' else 32
        train_loader, val_loader, class_names = create_data_loaders(
            unified_data_dir, batch_size=batch_size, img_size=224, num_workers=0
        )
        print(f"   Successfully created data loaders with batch size {batch_size}")
    except Exception as e:
        print(f"❌ Error creating data loaders: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Initialize model - use light version for CPU
    print("🏗️ Building model...")
    try:
        # For CPU training, use the light model
        if device.type == 'cpu':
            print("   Using lightweight model for CPU training")
            model = create_model('light', num_classes=3)
        else:
            print("   Using ResNet50 with attention for GPU training")
            model = create_model('resnet50', num_classes=3)
        
        model = model.to(device)
        print(f"   Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    except Exception as e:
        print(f"❌ Error creating model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Train model
    trainer = ModelTrainer(model, device, save_dir=os.path.join(base_dir, 'models'))
    try:
        # Use fewer epochs for CPU training
        epochs = 5 if device.type == 'cpu' else 20
        print(f"\n📋 Training Configuration:")
        print(f"   Epochs: {epochs}")
        print(f"   Batch size: {batch_size}")
        print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"   Device: {device}")
        
        trained_model = trainer.train(train_loader, val_loader, epochs=epochs, 
                                     class_names=class_names)
        print("\n✅ Training complete!")
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
        trainer.save_model('interrupted_model.pth')
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()