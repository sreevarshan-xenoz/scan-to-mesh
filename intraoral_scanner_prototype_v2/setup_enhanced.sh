#!/bin/bash
# Enhanced IntraoralScan v2 Setup Script
# Sets up PyTorch GPU acceleration and professional dependencies

set -e  # Exit on any error

echo "🦷 Enhanced IntraoralScan v2 Setup"
echo "=================================="
echo

# Check if we're in the right directory
if [ ! -f "main_v2.py" ]; then
    echo "❌ Error: Please run this script from the intraoral_scanner_prototype_v2 directory"
    exit 1
fi

# Check Python version
python_version=$(python3 --version 2>&1 | grep -oP '\d+\.\d+' || echo "0.0")
required_version="3.8"

if ! python3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 8) else 1)" 2>/dev/null; then
    echo "❌ Error: Python 3.8+ required (found Python $python_version)"
    echo "   Please install Python 3.8 or newer"
    exit 1
fi

echo "✅ Python $python_version detected"

# Check for CUDA/GPU support
echo
echo "🔍 Checking GPU support..."

if command -v nvidia-smi >/dev/null 2>&1; then
    gpu_info=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | head -1)
    echo "✅ NVIDIA GPU detected: $gpu_info"
    use_gpu=true
else
    echo "⚠️  No NVIDIA GPU detected - will use CPU fallback"
    use_gpu=false
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv_enhanced" ]; then
    echo
    echo "📦 Creating enhanced virtual environment..."
    python3 -m venv venv_enhanced
    echo "✅ Virtual environment created"
fi

# Activate virtual environment
echo
echo "🔄 Activating virtual environment..."
source venv_enhanced/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install PyTorch with appropriate CUDA support
echo
echo "🔥 Installing PyTorch..."

if [ "$use_gpu" = true ]; then
    # Check CUDA version
    cuda_version=$(nvidia-smi | grep -oP 'CUDA Version: \K\d+\.\d+' || echo "11.8")
    echo "   CUDA version: $cuda_version"
    
    if [[ "$cuda_version" == "12."* ]]; then
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    elif [[ "$cuda_version" == "11."* ]]; then
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
    else
        echo "   Using CPU version due to unsupported CUDA version"
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    fi
else
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
fi

echo "✅ PyTorch installed"

# Install core 3D processing libraries
echo
echo "🌍 Installing 3D processing libraries..."

# Open3D
pip install open3d>=0.17.0

# Computer vision
pip install opencv-python>=4.5.0 scikit-image>=0.19.0

echo "✅ 3D processing libraries installed"

# Install AI/ML libraries
echo
echo "🤖 Installing AI/ML libraries..."

if [ "$use_gpu" = true ]; then
    pip install onnxruntime-gpu>=1.15.0
else
    pip install onnxruntime>=1.15.0
fi

pip install pillow>=9.0.0 numpy>=1.21.0 scipy>=1.7.0

echo "✅ AI/ML libraries installed"

# Install UI framework
echo
echo "🖥️ Installing UI framework..."
pip install PyQt6>=6.5.0

# Try to install PyQt6-3D (may not be available on all systems)
if pip install PyQt6-3D>=6.5.0 2>/dev/null; then
    echo "✅ PyQt6-3D installed"
else
    echo "⚠️  PyQt6-3D not available - basic UI will be used"
fi

# Install service communication
echo
echo "📡 Installing service communication..."
pip install pyzmq>=24.0.0

echo "✅ Service communication installed"

# Install hardware interfaces (optional)
echo
echo "🔌 Installing hardware interfaces..."

# Intel RealSense (may fail if SDK not installed)
if pip install pyrealsense2>=2.50.0 2>/dev/null; then
    echo "✅ Intel RealSense support installed"
else
    echo "⚠️  Intel RealSense SDK not found - webcam fallback will be used"
fi

# USB interfaces
pip install pyusb>=1.2.0 || echo "⚠️  pyusb installation failed - some USB cameras may not work"

# Install remaining dependencies
echo
echo "📚 Installing remaining dependencies..."
pip install psutil>=5.9.0 memory-profiler>=0.60.0
pip install pandas>=1.5.0 matplotlib>=3.5.0
pip install trimesh>=3.15.0
pip install pyyaml>=6.0.0

echo "✅ All dependencies installed"

# Create configuration directory
echo
echo "⚙️ Setting up configuration..."
mkdir -p config data models

# Test the installation
echo
echo "🧪 Testing installation..."

# Test basic imports
python3 -c "
import sys
print('Testing core imports...')

try:
    import numpy as np
    print('✅ NumPy:', np.__version__)
except ImportError as e:
    print('❌ NumPy import failed:', e)
    sys.exit(1)

try:
    import torch
    print('✅ PyTorch:', torch.__version__)
    if torch.cuda.is_available():
        print('✅ CUDA available - GPU acceleration enabled')
        print(f'   GPU: {torch.cuda.get_device_name(0)}')
        print(f'   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
    else:
        print('⚠️  CUDA not available - using CPU')
except ImportError as e:
    print('❌ PyTorch import failed:', e)
    sys.exit(1)

try:
    import open3d as o3d
    print('✅ Open3D:', o3d.__version__)
except ImportError as e:
    print('❌ Open3D import failed:', e)
    sys.exit(1)

try:
    import cv2
    print('✅ OpenCV:', cv2.__version__)
except ImportError as e:
    print('❌ OpenCV import failed:', e)
    sys.exit(1)

try:
    from PyQt6 import QtCore
    print('✅ PyQt6:', QtCore.PYQT_VERSION_STR)
except ImportError as e:
    print('❌ PyQt6 import failed:', e)
    sys.exit(1)

print('✅ All core dependencies working!')
"

if [ $? -eq 0 ]; then
    echo
    echo "🎉 Enhanced IntraoralScan v2 setup completed successfully!"
    echo
    echo "📋 Setup Summary:"
    echo "   ✅ Virtual environment: venv_enhanced"
    echo "   ✅ PyTorch with GPU acceleration: $([ "$use_gpu" = true ] && echo "Enabled" || echo "CPU only")"
    echo "   ✅ Open3D for 3D processing"
    echo "   ✅ PyQt6 for modern UI"
    echo "   ✅ Professional dependencies installed"
    echo
    echo "🚀 Next Steps:"
    echo "1. Activate environment: source venv_enhanced/bin/activate"
    echo "2. Test enhanced TSDF: python test_enhanced_tsdf.py"
    echo "3. Run the application: python main_v2.py"
    echo
    echo "📖 Documentation:"
    echo "   • Enhanced TSDF: processing/gpu_tsdf_enhanced.py"
    echo "   • Configuration: config/system_config.py"
    echo "   • Main application: main_v2.py"
    
    # Show activation command for copy-paste
    echo
    echo "💡 Quick start:"
    echo "   source venv_enhanced/bin/activate && python test_enhanced_tsdf.py"
    
else
    echo
    echo "❌ Setup failed during testing phase"
    echo "   Please check the error messages above and retry"
    exit 1
fi
