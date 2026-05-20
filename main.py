#!/usr/bin/env python3
"""
AURORA Deepfake Detector - Main Entry Point
Professional AI-powered deepfake detection system
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
    parser = argparse.ArgumentParser(
        description='AURORA Deepfake Detector - Professional AI-powered image manipulation detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python main.py --web                # Start web application
  python main.py --web --port 8080    # Start on custom port
  python main.py --web --open-browser # Auto-open browser
  python main.py --setup              # Complete setup workflow
  python main.py --check              # Check system and dependencies
    ''')
    
    parser.add_argument('--setup', action='store_true', help='Complete setup workflow (check + organize)')
    parser.add_argument('--organize', action='store_true', help='Organize datasets into unified structure')
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
    
    # Start web application
    if args.web:
        start_web_app(project_root, args)
        return
    
    # No arguments - show menu
    show_menu(project_root)

def show_version():
    """Show version information"""
    print("""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║   🚀 AURORA DEEPFAKE DETECTOR v3.0                           ║
║                                                              ║
║   Version: 3.0.0                                            ║
║   Release: May 2026                                         ║
║   Framework: PyTorch / Flask                                ║
║                                                              ║
║   Features:                                                 ║
║   ✓ Real vs AI-generated vs Deepfake classification        ║
║   ✓ 10 forensic detection methods                          ║
║   ✓ Modern responsive web interface                         ║
║   ✓ Batch image processing                                  ║
║   ✓ History tracking & analytics                            ║
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
    directories = ['models', 'uploads', 'history', 'web/templates', 'web/static']
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
        print(f"✗ Model not found - using forensic analysis (still works!)")
    
    # Check required packages
    print("\n📦 Checking Python packages...")
    required_packages = {
        'flask': 'Flask',
        'torch': 'PyTorch',
        'cv2': 'OpenCV',
        'PIL': 'Pillow',
        'numpy': 'NumPy'
    }
    
    for module, name in required_packages.items():
        try:
            __import__(module)
            print(f"✓ {name} installed")
        except ImportError:
            print(f"✗ {name} NOT installed")
            print(f"  Run: pip install {name.lower()}")
    
    print("\n" + "="*50)
    print("✅ System check complete!")

def complete_setup(project_root):
    """Complete setup workflow"""
    print("""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║   🚀 AURORA COMPLETE SETUP                                   ║
║                                                              ║
║   This will:                                                ║
║   1. Check your system and dependencies                     ║
║   2. Create necessary folders                               ║
║   3. Verify web interface files                             ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
""")
    
    # Step 1: Check system
    print("\n📋 STEP 1: Checking system...")
    check_system(project_root)
    
    # Step 2: Create model directory if needed
    print("\n📋 STEP 2: Creating directories...")
    os.makedirs(os.path.join(project_root, 'models'), exist_ok=True)
    os.makedirs(os.path.join(project_root, 'uploads'), exist_ok=True)
    os.makedirs(os.path.join(project_root, 'history'), exist_ok=True)
    print("   ✓ All directories created")
    
    # Step 3: Verify web interface
    print("\n📋 STEP 3: Verifying web interface...")
    template_path = os.path.join(project_root, 'web', 'templates', 'upload_image.html')
    if not os.path.exists(template_path):
        print("   ⚠️ Web template not found!")
        print("   Please ensure upload_image.html exists in web/templates/")
    else:
        print("   ✓ Web template found")
    
    print("\n✅ SETUP COMPLETE!")
    print("\n📋 Next steps:")
    print("   1. Start web app:   python main.py --web --open-browser")
    print("   2. Upload images and test the detection")
    print("")

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
    
    if os.path.exists(deepfake_dataset):
        print(f"✓ Found Deepfake dataset: {deepfake_dataset}")
    else:
        print(f"✗ Deepfake dataset not found: {deepfake_dataset}")
    
    # Create organized structure
    target_base = os.path.join(project_root, "datasets", "organized")
    train_real = os.path.join(target_base, "train", "real")
    train_fake = os.path.join(target_base, "train", "fake")
    test_real = os.path.join(target_base, "test", "real")
    test_fake = os.path.join(target_base, "test", "fake")
    
    for dir_path in [train_real, train_fake, test_real, test_fake]:
        os.makedirs(dir_path, exist_ok=True)
    
    print("\n✅ Dataset organization complete!")

def start_web_app(project_root, args):
    """Start the web application"""
    print("\n🌐 STARTING AURORA WEB APPLICATION")
    print("="*50)
    print(f"📍 URL: http://{args.host}:{args.port}")
    print(f"📁 Project: {project_root}")
    print(f"🔧 Debug: {'ON' if args.debug else 'OFF'}")
    print("="*50)
    
    # Check web files
    template_path = os.path.join(project_root, 'web', 'templates', 'upload_image.html')
    if not os.path.exists(template_path):
        print(f"⚠️ Template not found: {template_path}")
        print("   Using fallback interface...")
    
    # Check if model exists
    model_path = os.path.join(project_root, 'models', 'best_model.pth')
    if os.path.exists(model_path):
        size = os.path.getsize(model_path) / (1024 * 1024)
        print(f"✅ Model found ({size:.1f} MB)")
    else:
        print(f"ℹ️ Model not found - using forensic analysis (still works!)")
        print(f"   For better accuracy, train a model or download a pre-trained one")
    
    # Open browser if requested
    if args.open_browser:
        print("\n🌐 Opening browser...")
        def open_browser():
            time.sleep(1.5)
            webbrowser.open(f"http://{args.host}:{args.port}")
        
        import threading
        threading.Thread(target=open_browser, daemon=True).start()
    
    # Start web app
    web_app_path = os.path.join(project_root, 'web', 'app.py')
    
    if not os.path.exists(web_app_path):
        print(f"❌ Web app not found: {web_app_path}")
        print("   Please ensure web/app.py exists")
        return
    
    # Set environment variables
    env = os.environ.copy()
    env['FLASK_APP'] = web_app_path
    env['FLASK_DEBUG'] = '1' if args.debug else '0'
    
    print("\n🚀 Starting server...")
    print("Press CTRL+C to stop\n")
    
    # Run the web app
    try:
        subprocess.run([sys.executable, web_app_path], env=env)
    except KeyboardInterrupt:
        print("\n\n👋 AURORA Deepfake Detector stopped.")
    except Exception as e:
        print(f"❌ Error: {e}")

def show_menu(project_root):
    """Show interactive menu"""
    print("""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║   🛡️  AURORA DEEPFAKE DETECTOR v3.0                          ║
║                                                              ║
║   Professional AI-powered image manipulation detection      ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝

📋 Quick Start:

   1️⃣  Check system:         python main.py --check
   2️⃣  Complete setup:       python main.py --setup
   3️⃣  Launch web app:       python main.py --web --open-browser

🎯 Web App Options:

   python main.py --web --port 8080
   python main.py --web --host 0.0.0.0
   python main.py --web --debug

ℹ️  Other Commands:

   --version                 Show version information
   --help                    Show help message

📊 Features:
   • Real Image Detection
   • AI Generated Detection  
   • Deepfake Detection
   • Batch Processing
   • History & Analytics
   • 10 Forensic Methods

""")
    
    # Interactive mode
    choice = input("👉 Select option [1/2/3/q]: ").strip()
    
    if choice == '1':
        check_system(project_root)
    elif choice == '2':
        complete_setup(project_root)
    elif choice == '3':
        class Args:
            web = True
            host = '127.0.0.1'
            port = 5000
            open_browser = True
            debug = False
        start_web_app(project_root, Args())
    elif choice == 'q':
        print("\n👋 Goodbye!")
    else:
        print("\n❌ Invalid choice. Run 'python main.py --help' for options.")

if __name__ == '__main__':
    main()