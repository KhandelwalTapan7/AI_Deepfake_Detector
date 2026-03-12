#!/usr/bin/env python3
"""
AI Deepfake Detector - Main Entry Point
Enhanced version with more features and better web app integration
"""

import os
import sys
import argparse
import subprocess
import webbrowser
import time

def main():
    parser = argparse.ArgumentParser(description='AI Deepfake Detector')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--organize', action='store_true', help='Organize datasets')
    parser.add_argument('--web', action='store_true', help='Start web application')
    parser.add_argument('--port', type=int, default=5000, help='Web app port')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='Web app host (use 127.0.0.1 for local access)')
    parser.add_argument('--open-browser', action='store_true', help='Automatically open browser when web app starts')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    parser.add_argument('--version', action='store_true', help='Show version information')
    
    args = parser.parse_args()
    
    # Get project root
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    # Add project root to path
    sys.path.insert(0, project_root)
    
    # Show version
    if args.version:
        show_version()
        return
    
    # Organize datasets
    if args.organize:
        print("📁 Organizing datasets...")
        try:
            from src.data_preprocessing.organize_data import organize_datasets
            organize_datasets()
            print("✅ Dataset organization complete!")
        except Exception as e:
            print(f"❌ Error organizing datasets: {e}")
            sys.exit(1)
    
    # Train model
    elif args.train:
        print("🏋️  Training model...")
        print("="*50)
        try:
            from src.model.train_model import main as train_main
            train_main()
            print("\n✅ Training complete!")
        except ImportError as e:
            print(f"❌ Import error: {e}")
            print("\n💡 Make sure you have:")
            print("   - Created the model files")
            print("   - Installed all dependencies")
            print("   - Organized your datasets first (python main.py --organize)")
            sys.exit(1)
        except Exception as e:
            print(f"❌ Error during training: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    # Start web application
    elif args.web:
        print("🌐 Starting web application...")
        print("="*50)
        print(f"📱 URL: http://{args.host}:{args.port}")
        print(f"📁 Project root: {project_root}")
        print(f"🔧 Debug mode: {'ON' if args.debug else 'OFF'}")
        print("="*50)
        
        # Check if model exists
        model_path = os.path.join(project_root, 'models', 'best_model.pth')
        if os.path.exists(model_path):
            model_size = os.path.getsize(model_path) / (1024 * 1024)
            print(f"✅ Model found: {model_path} ({model_size:.2f} MB)")
        else:
            print(f"⚠️ Model not found. The app will run in demo mode.")
            print(f"   Train a model first with: python main.py --train")
        
        # Check if template exists
        template_path = os.path.join(project_root, 'web', 'templates', 'upload_image.html')
        if os.path.exists(template_path):
            print(f"✅ Template found")
        else:
            print(f"❌ Template not found at: {template_path}")
            print("   Please ensure upload_image.html exists in web/templates/")
        
        # Open browser automatically if requested
        if args.open_browser:
            print("\n🌐 Opening browser...")
            time.sleep(2)  # Wait for server to start
            webbrowser.open(f"http://{args.host}:{args.port}")
        
        # Start the web app
        web_app_path = os.path.join(project_root, 'web', 'app.py')
        
        # Build command
        cmd = [sys.executable, web_app_path]
        if args.debug:
            cmd.append('--debug')
        
        try:
            subprocess.run(cmd)
        except KeyboardInterrupt:
            print("\n\n👋 Web application stopped.")
        except Exception as e:
            print(f"❌ Error starting web app: {e}")
            sys.exit(1)
    
    # No arguments - show help
    else:
        show_help()

def show_version():
    """Show version information"""
    print("🚀 AI Deepfake Detector v2.0")
    print("="*40)
    print("Version: 2.0.0")
    print("Release Date: March 2026")
    print("Framework: PyTorch")
    print("Features:")
    print("  - Real vs AI-generated vs Deepfake classification")
    print("  - Modern web interface")
    print("  - Batch processing")
    print("  - History tracking")
    print("  - Statistics dashboard")
    print("  - Export capabilities")
    print("\nFor more information, visit:")
    print("  https://github.com/yourusername/ai-deepfake-detector")

def show_help():
    """Show enhanced help message"""
    print("🚀 AI Deepfake Detector")
    print("="*50)
    print("\n📋 Available commands:")
    print("  --organize        Organize datasets into unified structure")
    print("  --train          Train the model")
    print("  --web            Start web application")
    print("\n⚙️  Web app options:")
    print("  --port PORT      Set port number (default: 5000)")
    print("  --host HOST      Set host address (default: 127.0.0.1)")
    print("  --open-browser   Automatically open browser")
    print("  --debug          Run in debug mode")
    print("\nℹ️  Other options:")
    print("  --version        Show version information")
    print("  --help           Show this help message")
    print("\n📝 Examples:")
    print("  python main.py --organize")
    print("  python main.py --train")
    print("  python main.py --web")
    print("  python main.py --web --port 8080 --open-browser")
    print("  python main.py --web --host 0.0.0.0 --debug")
    print("\n💡 Quick start:")
    print("  1. python main.py --organize    # Prepare datasets")
    print("  2. python main.py --train       # Train model")
    print("  3. python main.py --web         # Launch web app")
    print("\n📚 For more information, visit the documentation:")
    print("   https://github.com/yourusername/ai-deepfake-detector/wiki")

def check_dependencies():
    """Check if required dependencies are installed"""
    required = ['torch', 'flask', 'pillow', 'numpy']
    missing = []
    
    for package in required:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print("⚠️ Missing dependencies:")
        for pkg in missing:
            print(f"   - {pkg}")
        print("\nInstall with: pip install " + " ".join(missing))
        return False
    return True

if __name__ == '__main__':
    main()