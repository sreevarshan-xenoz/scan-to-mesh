#!/usr/bin/env python3
"""
Quick launcher script for the intraoral scanner prototype
Handles dependency checking and provides user-friendly startup
"""

import sys
import subprocess
import importlib

def check_dependencies():
    """Check if all required dependencies are installed"""
    required_packages = [
        ('cv2', 'opencv-python'),
        ('open3d', 'open3d'),
        ('numpy', 'numpy'),
        ('scipy', 'scipy'),
        ('skimage', 'scikit-image'),
        ('matplotlib', 'matplotlib'),
        ('sklearn', 'scikit-learn')
    ]
    
    missing_packages = []
    
    for package_name, pip_name in required_packages:
        try:
            importlib.import_module(package_name)
            print(f"✓ {pip_name}")
        except ImportError:
            print(f"✗ {pip_name} - MISSING")
            missing_packages.append(pip_name)
    
    return missing_packages

def install_dependencies(packages):
    """Install missing dependencies"""
    if not packages:
        return True
    
    print(f"\nInstalling missing packages: {', '.join(packages)}")
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install"
        ] + packages)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to install packages: {e}")
        return False

def main():
    print("=== Intraoral Scanner Prototype Launcher ===\n")
    
    print("Checking dependencies...")
    missing = check_dependencies()
    
    if missing:
        print(f"\nMissing {len(missing)} required packages.")
        response = input("Install missing packages? (y/n): ").lower().strip()
        
        if response == 'y':
            if not install_dependencies(missing):
                print("Failed to install dependencies. Please install manually:")
                print(f"pip install {' '.join(missing)}")
                return 1
        else:
            print("Cannot run without required dependencies.")
            return 1
    
    print("\n✓ All dependencies satisfied")
    
    # Import and run the main application
    try:
        from main import main as run_scanner
        print("\nStarting Intraoral Scanner Prototype...")
        print("Make sure your camera is connected and not in use by other applications.\n")
        
        return run_scanner()
        
    except ImportError as e:
        print(f"Error importing main application: {e}")
        return 1
    except Exception as e:
        print(f"Error running application: {e}")
        return 1

if __name__ == "__main__":
    exit(main())