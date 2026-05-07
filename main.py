#!/usr/bin/env python3
"""
AI Deepfake Detector - Main Entry Point
Enhanced version with perfect web app integration and all features
"""

import os
import sys
import argparse
import subprocess
import webbrowser
import time
import json
import shutil
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='AI Deepfake Detector - Detect manipulated images with AI', 
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog='''
Examples:
  python main.py --organize              # Organize your datasets
  python main.py --train                 # Train the model
  python main.py --web                   # Start web application
  python main.py --web --port 8080       # Start on custom port
  python main.py --web --open-browser    # Auto-open browser
  python main.py --setup                 # Complete setup workflow
    ''')
    
    parser.add_argument('--setup', action='store_true', help='Complete setup workflow (organize + check web)')
    parser.add_argument('--organize', action='store_true', help='Organize datasets into unified structure')
    parser.add_argument('--train', action='store_true', help='Train the deepfake detection model')
    parser.add_argument('--web', action='store_true', help='Start web application')
    parser.add_argument('--port', type=int, default=5000, help='Web app port (default: 5000)')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='Web app host (default: 127.0.0.1)')
    parser.add_argument('--open-browser', action='store_true', help='Automatically open browser when web app starts')
    parser.add_argument('--debug', action='store_true', help='Run web app in debug mode')
    parser.add_argument('--check', action='store_true', help='Check system and dependencies')
    parser.add_argument('--version', action='store_true', help='Show version information')
    
    args = parser.parse_args()
    
    # Get project root
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    # Show version
    if args.version:
        show_version()
        return
    
    # Check system
    if args.check:
        check_system(project_root)
        return
    
    # Complete setup workflow
    if args.setup:
        complete_setup(project_root)
        return
    
    # Organize datasets
    if args.organize:
        organize_datasets(project_root)
        return
    
    # Train model
    if args.train:
        train_model(project_root)
        return
    
    # Start web application
    if args.web:
        start_web_app(project_root, args)
        return
    
    # No arguments - show help and quick start menu
    show_menu(project_root)

def show_version():
    """Show version information"""
    print("""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║   🚀 AI DEEPFAKE DETECTOR v2.0                               ║
║                                                              ║
║   Version: 2.0.0                                            ║
║   Release: March 2026                                       ║
║   Framework: PyTorch / TensorFlow                           ║
║                                                              ║
║   Features:                                                 ║
║   ✓ Real vs AI-generated vs Deepfake classification        ║
║   ✓ Modern responsive web interface                         ║
║   ✓ Batch image processing                                  ║
║   ✓ History tracking & analytics                            ║
║   ✓ Export results (JSON/CSV)                              ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
""")

def check_system(project_root):
    """Check system dependencies and setup"""
    print("\n🔍 SYSTEM CHECK")
    print("="*50)
    
    # Check Python version
    python_version = sys.version_info
    print(f"✓ Python: {python_version.major}.{python_version.minor}.{python_version.micro}")
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        print("  ⚠️ Warning: Python 3.8+ recommended")
    
    # Check directories
    directories = ['datasets', 'models', 'uploads', 'history', 'web/templates', 'web/static']
    for directory in directories:
        dir_path = os.path.join(project_root, directory)
        if os.path.exists(dir_path):
            print(f"✓ {directory}/ exists")
        else:
            print(f"✗ {directory}/ missing (will be created)")
            os.makedirs(dir_path, exist_ok=True)
    
    # Check web files
    template_path = os.path.join(project_root, 'web', 'templates', 'upload_image.html')
    if os.path.exists(template_path):
        print(f"✓ upload_image.html found")
    else:
        print(f"✗ upload_image.html missing - web interface won't work")
        print(f"  Please create: {template_path}")
    
    # Check model
    model_path = os.path.join(project_root, 'models', 'best_model.pth')
    if os.path.exists(model_path):
        size = os.path.getsize(model_path) / (1024 * 1024)
        print(f"✓ Model found ({size:.1f} MB)")
    else:
        print(f"✗ Model not found - train with: python main.py --train")
    
    print("\n" + "="*50)
    print("✅ System check complete!")

def complete_setup(project_root):
    """Complete setup workflow"""
    print("""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║   🚀 COMPLETE SETUP WORKFLOW                                 ║
║                                                              ║
║   This will:                                                ║
║   1. Organize your datasets                                 ║
║   2. Check web interface files                              ║
║   3. Verify everything is ready                             ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
""")
    
    # Step 1: Organize datasets
    print("\n📁 STEP 1: Organizing datasets...")
    organize_datasets(project_root)
    
    # Step 2: Check web files
    print("\n🌐 STEP 2: Checking web interface...")
    template_path = os.path.join(project_root, 'web', 'templates', 'upload_image.html')
    if not os.path.exists(template_path):
        print("   ⚠️ Web template not found!")
        print("   Please ensure upload_image.html exists in web/templates/")
    else:
        print("   ✓ Web template found")
    
    # Step 3: Final check
    print("\n✅ SETUP COMPLETE!")
    print("\n📋 Next steps:")
    print("   1. Train the model:   python main.py --train")
    print("   2. Start web app:     python main.py --web --open-browser")
    print("")
    
    # Ask if user wants to train
    response = input("Would you like to train the model now? (y/n): ").lower()
    if response == 'y':
        train_model(project_root)

def organize_datasets(project_root):
    """Organize datasets from various sources"""
    print("\n📁 ORGANIZING DATASETS")
    print("="*50)
    
    # Check for datasets
    ai_dataset = os.path.join(project_root, "AI Generated dataset")
    deepfake_dataset = os.path.join(project_root, "Deepfake dataset")
    
    if os.path.exists(ai_dataset):
        print(f"✓ Found AI dataset: {ai_dataset}")
    else:
        print(f"✗ AI dataset not found: {ai_dataset}")
        print("   Please place your 'AI Generated dataset' folder here")
    
    if os.path.exists(deepfake_dataset):
        print(f"✓ Found Deepfake dataset: {deepfake_dataset}")
    else:
        print(f"✗ Deepfake dataset not found: {deepfake_dataset}")
        print("   Please place your 'Deepfake dataset' folder here")
    
    # Create organized structure
    target_base = os.path.join(project_root, "datasets", "organized")
    train_real = os.path.join(target_base, "train", "real")
    train_fake = os.path.join(target_base, "train", "fake")
    test_real = os.path.join(target_base, "test", "real")
    test_fake = os.path.join(target_base, "test", "fake")
    
    for dir_path in [train_real, train_fake, test_real, test_fake]:
        os.makedirs(dir_path, exist_ok=True)
    
    # Process AI dataset
    if os.path.exists(ai_dataset):
        print("\n📋 Processing AI dataset...")
        for split in ['train', 'test']:
            for category, target_cat in [('REAL', 'real'), ('FAKE', 'fake')]:
                src_path = os.path.join(ai_dataset, split, category)
                if os.path.exists(src_path):
                    target_dir = os.path.join(target_base, split, target_cat)
                    files = [f for f in os.listdir(src_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                    for f in files:
                        shutil.copy2(os.path.join(src_path, f), os.path.join(target_dir, f"ai_{split}_{category}_{f}"))
                    print(f"   Copied {len(files)} images from {split}/{category}")
    
    # Process Deepfake dataset
    deepfake_cropped = os.path.join(deepfake_dataset, "cropped_images")
    if os.path.exists(deepfake_cropped):
        print("\n📋 Processing Deepfake dataset...")
        folders = [f for f in os.listdir(deepfake_cropped) if os.path.isdir(os.path.join(deepfake_cropped, f))]
        print(f"   Found {len(folders)} video folders")
        
        fake_count = 0
        for folder in folders[:50]:  # Limit for performance
            folder_path = os.path.join(deepfake_cropped, folder)
            images = [f for f in os.listdir(folder_path) if f.endswith('.png')]
            
            split_idx = int(len(images) * 0.8)
            train_imgs = images[:split_idx]
            test_imgs = images[split_idx:]
            
            for img in train_imgs:
                src = os.path.join(folder_path, img)
                dst = os.path.join(train_fake, f"deepfake_{folder}_train_{img}")
                shutil.copy2(src, dst)
                fake_count += 1
            
            for img in test_imgs:
                src = os.path.join(folder_path, img)
                dst = os.path.join(test_fake, f"deepfake_{folder}_test_{img}")
                shutil.copy2(src, dst)
                fake_count += 1
        
        print(f"   Added {fake_count} deepfake images")
    
    # Show statistics
    print("\n📊 ORGANIZED DATASET STATISTICS:")
    for split in ['train', 'test']:
        print(f"\n{split.upper()}:")
        for category in ['real', 'fake']:
            path = os.path.join(target_base, split, category)
            if os.path.exists(path):
                count = len([f for f in os.listdir(path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                print(f"   {category}: {count} images")
    
    print("\n✅ Dataset organization complete!")

def train_model(project_root):
    """Train the deepfake detection model"""
    print("\n🏋️ TRAINING MODEL")
    print("="*50)
    
    # Check if datasets are organized
    organized_path = os.path.join(project_root, "datasets", "organized", "train")
    if not os.path.exists(organized_path):
        print("❌ Datasets not organized yet!")
        print("   Run: python main.py --organize")
        return
    
    # Count training images
    train_real = os.path.join(organized_path, "real")
    train_fake = os.path.join(organized_path, "fake")
    real_count = len([f for f in os.listdir(train_real) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]) if os.path.exists(train_real) else 0
    fake_count = len([f for f in os.listdir(train_fake) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]) if os.path.exists(train_fake) else 0
    
    print(f"📊 Training data:")
    print(f"   Real images: {real_count}")
    print(f"   Fake images: {fake_count}")
    
    if real_count == 0 or fake_count == 0:
        print("\n❌ No training data found!")
        print("   Please organize your datasets first: python main.py --organize")
        return
    
    print("\n🎯 Starting training...")
    print("   This may take several minutes...\n")
    
    try:
        # Try to import and run training
        sys.path.insert(0, project_root)
        from src.model.train_your_data import train_with_your_data
        
        train_with_your_data(
            data_dir=os.path.join(project_root, "datasets", "organized"),
            model_type='efficientnet',
            epochs=30
        )
        print("\n✅ Training complete!")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("\n💡 Make sure you have the training script:")
        print("   src/model/train_your_data.py")
        print("\n   Or use the alternative training method:")
        print("   python -c 'from src.model.train_model import main; main()'")
        
    except Exception as e:
        print(f"❌ Training error: {e}")
        import traceback
        traceback.print_exc()

def start_web_app(project_root, args):
    """Start the web application"""
    print("\n🌐 STARTING WEB APPLICATION")
    print("="*50)
    print(f"📍 URL: http://{args.host}:{args.port}")
    print(f"📁 Project: {project_root}")
    print(f"🔧 Debug: {'ON' if args.debug else 'OFF'}")
    print("="*50)
    
    # Check web files
    template_path = os.path.join(project_root, 'web', 'templates', 'upload_image.html')
    if not os.path.exists(template_path):
        print(f"❌ Template not found: {template_path}")
        print("   Please create upload_image.html in web/templates/")
        return
    
    # Check if model exists
    model_path = os.path.join(project_root, 'models', 'best_model.pth')
    if os.path.exists(model_path):
        size = os.path.getsize(model_path) / (1024 * 1024)
        print(f"✅ Model found ({size:.1f} MB)")
    else:
        print(f"⚠️ Model not found - running in demo mode")
        print(f"   Train a model with: python main.py --train")
    
    # Open browser if requested
    if args.open_browser:
        print("\n🌐 Opening browser...")
        threading = __import__('threading')
        def open_browser():
            time.sleep(1.5)
            webbrowser.open(f"http://{args.host}:{args.port}")
        threading.Thread(target=open_browser, daemon=True).start()
    
    # Start web app
    web_app_path = os.path.join(project_root, 'web', 'app.py')
    
    if not os.path.exists(web_app_path):
        print(f"❌ Web app not found: {web_app_path}")
        print("   Please create web/app.py")
        return
    
    # Set environment variables
    env = os.environ.copy()
    env['FLASK_APP'] = web_app_path
    env['FLASK_DEBUG'] = '1' if args.debug else '0'
    
    # Run the web app
    try:
        subprocess.run([sys.executable, web_app_path], env=env)
    except KeyboardInterrupt:
        print("\n\n👋 Web application stopped.")
    except Exception as e:
        print(f"❌ Error: {e}")

def show_menu(project_root):
    """Show interactive menu"""
    print("""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║   🤖 AI DEEPFAKE DETECTOR                                    ║
║                                                              ║
║   Protect yourself from manipulated and AI-generated images ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝

📋 Quick Start:

   1️⃣  Organize datasets:    python main.py --organize
   2️⃣  Train model:          python main.py --train  
   3️⃣  Launch web app:       python main.py --web --open-browser

🎯 Other Commands:

   📊 Check system:          python main.py --check
   🔧 Full setup:            python main.py --setup
   ℹ️  Version info:          python main.py --version
   🆘 Help:                  python main.py --help

⚙️ Web App Options:

   python main.py --web --port 8080
   python main.py --web --host 0.0.0.0 --debug
""")
    
    # Interactive mode
    choice = input("👉 Select an option [1/2/3/q]: ").strip()
    
    if choice == '1':
        organize_datasets(project_root)
    elif choice == '2':
        train_model(project_root)
    elif choice == '3':
        args = argparse.Namespace()
        args.web = True
        args.host = '127.0.0.1'
        args.port = 5000
        args.open_browser = True
        args.debug = False
        start_web_app(project_root, args)
    elif choice == 'q':
        print("\n👋 Goodbye!")
    else:
        print("\n❌ Invalid choice. Run 'python main.py --help' for options.")

if __name__ == '__main__':
    main()