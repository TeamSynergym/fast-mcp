#!/usr/bin/env python3
"""
Quick setup and run script for Streamlit Posture Analysis App
"""

import os
import sys
import subprocess
from pathlib import Path


def check_dependencies():
    """Check if required dependencies are installed."""
    required_packages = ['streamlit', 'ultralytics', 'opencv-python', 'pillow']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing dependencies:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n📦 Install them with:")
        print(f"   uv add {' '.join(missing_packages)}")
        return False
    
    print("✅ All dependencies are installed!")
    return True


def check_model():
    """Check if YOLO model exists."""
    model_path = "models/yolopose_v1.pt"
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found at: {model_path}")
        print("\n📥 Please:")
        print("1. Download your yolopose_v1.pt from Colab")
        print("2. Place it in the models/ directory")
        print("3. Run this script again")
        return False
    
    print(f"✅ Model found at: {model_path}")
    return True


def create_directories():
    """Create necessary directories."""
    directories = ["models", "data/test_images"]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"📁 Created directory: {directory}")


def main():
    print("🚀 Streamlit Posture Analysis Setup")
    print("=" * 40)
    
    # Create directories
    create_directories()
    print()
    
    # Check dependencies
    if not check_dependencies():
        return
    
    print()
    
    # Check model
    if not check_model():
        return
    
    print()
    print("🎉 Everything is ready!")
    print("=" * 40)
    print("Starting Streamlit app...")
    print("The app will open in your browser at http://localhost:8501")
    print("Press Ctrl+C to stop the app")
    print()
    
    # Run Streamlit
    try:
        subprocess.run(["streamlit", "run", "streamlit_app.py"], check=True)
    except KeyboardInterrupt:
        print("\n👋 App stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error running Streamlit: {e}")
    except FileNotFoundError:
        print("\n❌ Streamlit not found. Make sure it's installed:")
        print("   uv add streamlit")


if __name__ == "__main__":
    main() 