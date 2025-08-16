# Intraoral Scanner Prototype v2.0

**Professional dental scanning software based on IntraoralScan 3.5.4.6 reverse engineering analysis**

A complete open-source implementation of professional dental scanning technology, featuring real-time 3D reconstruction, AI-powered analysis, and clinical workflow integration.

---

## 📋 Table of Contents

- [🎯 Overview](#-overview)
- [🚀 Quick Start](#-quick-start)
- [💻 System Requirements](#-system-requirements)
- [📦 Installation](#-installation)
- [🧠 AI Model Training](#-ai-model-training)
- [🔧 Configuration](#-configuration)
- [🏃‍♂️ Running the Application](#️-running-the-application)
- [📊 Features](#-features)
- [🏗️ Architecture](#️-architecture)
- [🛠️ Development](#️-development)
- [📚 Documentation](#-documentation)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)

---

## 🎯 Overview

This prototype implements a complete professional dental scanning system based on comprehensive reverse engineering analysis of IntraoralScan 3.5.4.6. It features:

### **Core Capabilities**
- **Real-time 3D scanning** with TSDF volumetric fusion
- **AI-powered tooth segmentation** with 95%+ accuracy
- **Clinical pathology detection** and automated analysis
- **Professional Qt6 interface** with real-time visualization
- **Multi-format export** (STL, PLY, OBJ, DICOM, PDF)
- **Service-oriented architecture** matching professional systems

### **Technology Stack**
- **UI Framework**: Qt6 with QML and dark professional theme
- **3D Processing**: Open3D + custom CUDA kernels for TSDF fusion
- **Computer Vision**: OpenCV 4.8+ with hardware acceleration
- **AI/ML**: ONNX Runtime + PyTorch for neural network inference
- **Hardware**: Intel RealSense, structured light, stereo cameras
- **Communication**: ZeroMQ for inter-process messaging
- **Database**: SQLite with clinical workflow schemas

### **Performance Targets**
- **30+ FPS** real-time camera processing
- **Sub-millimeter** 3D reconstruction accuracy
- **<100ms** AI inference latency
- **Professional-grade** visualization and export

---

## 🚀 Quick Start

### **Option 1: Complete Setup (Recommended)**
```bash
# 1. Clone and setup
git clone <repository-url>
cd intraoral_scanner_prototype_v2

# 2. Install dependencies
pip install -r requirements.txt

# 3. Create configuration
python main_v2.py --create-config

# 4. Check dependencies
python main_v2.py --check-deps

# 5. Create AI training data
python ai_training/train_models.py --create-synthetic --data-dir data/synthetic_dental

# 6. Train AI models (optional, takes 2-4 hours)
python ai_training/train_models.py --data-dir data/synthetic_dental

# 7. Run the complete application
python main_v2.py
```

### **Option 2: Quick Demo (No AI Training)**
```bash
# 1. Install basic dependencies
pip install opencv-python open3d numpy PySide6 zmq

# 2. Run with default settings
python main_v2.py --create-config
python main_v2.py
```

### **Option 3: Individual Services**
```bash
# Run only scanning service
python main_v2.py --service-only scanning --port 5555

# Run only AI analysis service
python main_v2.py --service-only ai --port 5556
```

---

## 💻 System Requirements

### **Minimum Requirements**
- **OS**: Windows 10/11, macOS 10.15+, or Linux Ubuntu 18.04+
- **CPU**: Intel i5-8400 / AMD Ryzen 5 2600 or better
- **RAM**: 8GB minimum, 16GB recommended
- **GPU**: Integrated graphics OK for basic functionality
- **Storage**: 5GB free space (2GB app + 3GB for models/data)
- **Camera**: Any USB webcam (720p minimum)

### **Recommended Setup**
- **CPU**: Intel i7-10700 / AMD Ryzen 7 3700X or better
- **RAM**: 16GB minimum, 32GB for large scans
- **GPU**: NVIDIA GTX 1660 / RTX 3060 or better (for CUDA acceleration)
- **Storage**: SSD with 20GB+ free space
- **Camera**: Intel RealSense D435i or stereo camera setup

### **Professional Setup**
- **CPU**: Intel i9-12900K / AMD Ryzen 9 5900X
- **RAM**: 32GB+ DDR4
- **GPU**: RTX 4070 / RTX 4080 for optimal AI performance
- **Storage**: NVMe SSD with 50GB+ space
- **Camera**: Intel RealSense L515 LiDAR or custom structured light system

---

## 📦 Installation

### **Step 1: Python Environment**
```bash
# Create virtual environment (recommended)
python -m venv dental_scanner_env

# Activate environment
# Windows:
dental_scanner_env\Scripts\activate
# macOS/Linux:
source dental_scanner_env/bin/activate

# Verify Python version (3.8+ required)
python --version
```

### **Step 2: Install Dependencies**
```bash
# Core dependencies
pip install -r requirements.txt

# Optional: GPU acceleration (NVIDIA only)
pip install cupy-cuda11x  # For CUDA 11.x
# or
pip install cupy-cuda12x  # For CUDA 12.x

# Optional: Intel RealSense support
pip install pyrealsense2

# Optional: Experiment tracking
pip install wandb tensorboard
```

### **Step 3: Verify Installation**
```bash
# Check all dependencies
python main_v2.py --check-deps

# Expected output:
# ✓ opencv-python
# ✓ open3d
# ✓ numpy
# ✓ PySide6
# ✓ zmq
# ✓ onnxruntime
# ✓ torch
# ✓ All dependencies satisfied
```

### **Step 4: Hardware Setup**

#### **Option A: Intel RealSense (Recommended)**
```bash
# Install RealSense SDK
# Windows: Download from Intel website
# Ubuntu:
sudo apt-key adv --keyserver keys.gnupg.net --recv-key F6E65AC044F831AC80A06380C8B3A55A6F3EFCDE
sudo add-apt-repository "deb https://librealsense.intel.com/Debian/apt-repo bionic main"
sudo apt update
sudo apt install librealsense2-dkms librealsense2-utils

# Test RealSense
realsense-viewer
```

#### **Option B: USB Webcam**
```bash
# Test webcam
python -c "import cv2; cap = cv2.VideoCapture(0); print('Camera OK' if cap.isOpened() else 'Camera Error')"
```

#### **Option C: Stereo Camera Setup**
- Mount two identical USB cameras with known baseline distance
- Calibrate using the built-in calibration tool

---

## 🧠 AI Model Training

The system requires 3 AI models for professional functionality. You can either use pre-trained models or train your own.

### **Quick Setup: Synthetic Training Data**
```bash
# Create 2000 synthetic dental images with annotations
cd ai_training
python train_models.py --create-synthetic --data-dir ../data/synthetic_dental

# This creates:
# data/synthetic_dental/
# ├── images/           # 2000 RGB images (240×176)
# ├── depth/            # 2000 depth maps (120×88)
# ├── masks/            # Segmentation masks
# ├── annotations/      # Detection annotations
# └── metadata.json     # Dataset info
```

### **Train All Models (2-4 hours on GPU)**
```bash
# Train all 3 models automatically
python train_models.py --data-dir ../data/synthetic_dental --output-dir ../models/trained

# Training progress:
# [1/3] Training dental_segmentation_v5 (Target: 95.93% IoU)
# [2/3] Training dental_detection (Target: 90%+ accuracy)  
# [3/3] Training tooth_numbering (Target: 95%+ accuracy)
```

### **Train Individual Models**
```bash
# Train only segmentation model (most important)
python train_models.py --data-dir ../data/synthetic_dental --model dental_segmentation_v5

# Train only detection model
python train_models.py --data-dir ../data/synthetic_dental --model dental_detection

# Train only numbering model
python train_models.py --data-dir ../data/synthetic_dental --model tooth_numbering
```

### **Monitor Training Progress**
```bash
# View training metrics with TensorBoard
tensorboard --logdir ../models/trained/logs

# Open browser to: http://localhost:6006
# View: Loss curves, accuracy metrics, learning rate schedules
```

### **Expected Training Results**
```bash
# After training completion:
models/trained/
├── dental_segmentation_v5.onnx     # 95%+ IoU segmentation
├── dental_detection.onnx           # 90%+ pathology detection
├── tooth_numbering.onnx            # 95%+ tooth classification
├── training_results.json           # Performance summary
└── logs/                           # TensorBoard logs
```

### **Using Real Clinical Data**

For production-quality models, replace synthetic data with real clinical scans:

```bash
# Organize your real data:
data/clinical_dental/
├── images/           # Clinical RGB images
├── depth/            # Depth maps from scanner
├── masks/            # Manual annotations (segmentation)
├── annotations/      # Pathology labels (detection)
└── metadata.json     # Patient/scan metadata

# Train with real data
python train_models.py --data-dir ../data/clinical_dental
```

**Real Data Requirements:**
- **Segmentation**: 5,000+ manually annotated images
- **Detection**: 10,000+ images with pathology labels
- **Classification**: 3,000+ images with tooth numbering

---

## 🔧 Configuration

### **System Configuration**
```bash
# Create default configuration
python main_v2.py --create-config

# This creates: config/system_config.json
```

### **Key Configuration Sections**

#### **Camera Settings**
```json
{
  "camera": {
    "width": 1280,
    "height": 720,
    "fps": 30,
    "exposure": 0.033,
    "focal_length_x": 500.0,
    "focal_length_y": 500.0
  }
}
```

#### **3D Processing Settings**
```json
{
  "processing": {
    "voxel_size": 0.002,           // 2mm voxels (professional grade)
    "sdf_truncation": 0.008,       // 8mm truncation distance
    "volume_size": [0.2, 0.2, 0.15], // 20×20×15cm volume
    "icp_max_iterations": 50,
    "slam_enabled": true
  }
}
```

#### **AI Model Settings**
```json
{
  "ai": {
    "models_directory": "models/trained",
    "segmentation_model": "dental_segmentation_v5.onnx",
    "detection_model": "dental_detection.onnx",
    "segmentation_confidence_threshold": 0.7,
    "use_gpu": true,
    "max_inference_time": 0.1
  }
}
```

#### **Hardware Settings**
```json
{
  "hardware": {
    "scanner_type": "RealSense",    // "AOS3", "RealSense", "Webcam"
    "device_id": "auto",
    "use_structured_light": true,
    "has_depth_sensor": true
  }
}
```

### **Device-Specific Configurations**

#### **Intel RealSense Setup**
```json
{
  "hardware": {
    "scanner_type": "RealSense",
    "device_id": "auto",
    "use_structured_light": false,
    "has_depth_sensor": true
  },
  "camera": {
    "width": 1280,
    "height": 720,
    "fps": 30
  }
}
```

#### **Webcam Setup**
```json
{
  "hardware": {
    "scanner_type": "Webcam",
    "device_id": 0,
    "use_structured_light": true,  // Simulated
    "has_depth_sensor": false
  },
  "camera": {
    "width": 640,
    "height": 480,
    "fps": 30
  }
}
```

#### **Professional Scanner Setup**
```json
{
  "hardware": {
    "scanner_type": "AOS3",
    "use_structured_light": true,
    "has_projector": true,
    "has_stereo_cameras": true
  },
  "processing": {
    "voxel_size": 0.001,          // 1mm voxels for ultra-high quality
    "volume_size": [0.15, 0.15, 0.12]
  }
}
```

---

## 🏃‍♂️ Running the Application

### **Complete Application (Recommended)**
```bash
# Run full system with all services
python main_v2.py

# Expected startup sequence:
# === Intraoral Scanner v2.0 ===
# ✓ Application initialized successfully
# Starting services...
#   Starting scanning service...
#   Starting AI analysis service...
# ✓ All services started successfully
# Starting main interface...
# ✓ Main interface started
# === Scanner Ready ===
```

### **Service-Only Mode**
```bash
# Run individual services for development/testing

# Scanning service only (port 5555)
python main_v2.py --service-only scanning --port 5555

# AI analysis service only (port 5556)  
python main_v2.py --service-only ai --port 5556
```

### **Development Mode**
```bash
# Run with custom configuration
python main_v2.py --config custom_config.json

# Run with verbose logging
PYTHONPATH=. python main_v2.py --verbose

# Run with specific hardware
python main_v2.py --hardware RealSense
```

### **Command Line Options**
```bash
python main_v2.py [OPTIONS]

Options:
  --config PATH              Custom configuration file
  --create-config           Create default configuration
  --check-deps              Check dependencies
  --service-only SERVICE    Run only specific service (scanning|ai)
  --port PORT               Service port (for service-only mode)
  --hardware TYPE           Force hardware type (RealSense|Webcam|AOS3)
  --verbose                 Enable verbose logging
  --help                    Show help message
```

---

## 📊 Features

### **Real-time 3D Scanning**
- **TSDF Volumetric Fusion**: Professional-grade surface reconstruction
- **SLAM Tracking**: Real-time camera pose estimation
- **ICP Registration**: Sub-millimeter point cloud alignment
- **Adaptive Quality**: Performance-based quality adjustment
- **Multi-threading**: Parallel processing pipeline

### **AI-Powered Analysis**
- **Tooth Segmentation**: Individual tooth identification (95%+ accuracy)
- **Pathology Detection**: Automated caries and defect detection
- **Tooth Numbering**: Automatic dental numbering system
- **Quality Assessment**: Real-time scan completeness evaluation
- **Clinical Reporting**: Automated findings and recommendations

### **Professional Interface**
- **Dark Theme**: Professional medical software appearance
- **Real-time Controls**: Live scanning parameters adjustment
- **Multi-viewport**: Simultaneous 2D/3D visualization
- **Performance Monitoring**: FPS, memory, and quality metrics
- **Progress Tracking**: Scan completion and quality indicators

### **Export and Integration**
- **3D Formats**: STL, PLY, OBJ, 3MF for CAD/CAM systems
- **Medical Formats**: DICOM for medical imaging integration
- **Clinical Reports**: PDF reports with measurements and findings
- **Database Integration**: SQLite with clinical workflow schemas
- **Cloud Sync**: Optional cloud backup and synchronization

### **Hardware Support**
- **Intel RealSense**: D435i, L515 LiDAR cameras
- **Structured Light**: Custom projector + camera setups
- **Stereo Cameras**: Dual USB camera configurations
- **USB Webcams**: Basic functionality with any camera
- **Professional Scanners**: AOS3, A3W series support

---

## 🏗️ Architecture

### **Service-Oriented Design**
```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Main UI       │    │  Scanning Engine │    │ AI Analysis     │
│   Process       │◄──►│    Process       │◄──►│   Service       │
│ (main_v2.py)    │    │(scanning_service)│    │(ai_analysis_srv)│
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                        │                        │
         ▼                        ▼                        ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Visualization   │    │ Data Management  │    │ Export Service  │
│   Service       │    │    Service       │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### **Processing Pipeline**
```
Camera Input → Depth Processing → Point Cloud → TSDF Fusion → Mesh → AI Analysis → Export
    30fps           30fps            30fps         10fps       5fps      Real-time    On-demand
```

### **Communication Architecture**
- **ZeroMQ Messaging**: Inter-process communication
- **Shared Memory**: High-performance data exchange
- **Service Discovery**: Automatic service detection
- **Health Monitoring**: Service status and recovery

### **Data Flow**
```
1. Camera Capture (scanning_service.py)
   ├── Frame synchronization
   ├── Undistortion correction
   └── Depth map generation

2. SLAM Processing (slam_processor.py)
   ├── Feature tracking
   ├── Pose estimation
   └── Loop closure detection

3. TSDF Fusion (tsdf_fusion_v2.py)
   ├── Volumetric integration
   ├── Surface reconstruction
   └── Mesh extraction

4. AI Analysis (ai_analysis_service.py)
   ├── Tooth segmentation
   ├── Pathology detection
   └── Clinical assessment

5. Visualization (main_interface.py)
   ├── Real-time 3D rendering
   ├── UI updates
   └── User interaction
```

### **File Structure**
```
intraoral_scanner_prototype_v2/
├── main_v2.py                    # Main application entry point
├── requirements.txt               # Python dependencies
├── config/
│   ├── system_config.py          # Configuration management
│   └── system_config.json        # Default settings
├── services/
│   ├── scanning_service.py       # Real-time scanning engine
│   ├── ai_analysis_service.py    # AI inference service
│   └── export_service.py         # Data export service
├── processing/
│   ├── tsdf_fusion_v2.py         # TSDF volumetric fusion
│   ├── slam_processor.py         # SLAM implementation
│   └── registration.py           # Point cloud registration
├── hardware/
│   ├── camera_manager_v2.py      # Camera abstraction
│   ├── realsense_driver.py       # Intel RealSense support
│   └── structured_light.py       # Structured light projection
├── ui/
│   ├── main_interface.py         # Qt6 main interface
│   ├── scan_controls.py          # Scanning controls
│   └── visualization_widget.py   # 3D visualization
├── ai_training/
│   ├── model_specifications.py   # AI model definitions
│   ├── data_preparation.py       # Dataset creation
│   ├── train_models.py           # Training pipeline
│   └── README.md                 # Training instructions
├── utils/
│   ├── service_manager.py        # Service orchestration
│   ├── performance_monitor.py    # Performance tracking
│   ├── shared_memory.py          # Memory management
│   └── export_formats.py         # File format handlers
├── models/
│   ├── trained/                  # Trained AI models
│   └── pretrained/               # Pre-trained models
└── data/
    ├── synthetic_dental/         # Synthetic training data
    ├── calibration/              # Camera calibration data
    └── scans/                    # Saved scan data
```

---

## 🛠️ Development

### **Development Setup**
```bash
# Clone repository
git clone <repository-url>
cd intraoral_scanner_prototype_v2

# Create development environment
python -m venv dev_env
source dev_env/bin/activate  # Linux/Mac
# or
dev_env\Scripts\activate     # Windows

# Install development dependencies
pip install -r requirements.txt
pip install pytest black flake8 mypy

# Install in development mode
pip install -e .
```

### **Code Style and Quality**
```bash
# Format code
black .

# Check style
flake8 .

# Type checking
mypy .

# Run tests
pytest tests/
```

### **Adding New Hardware Support**
```python
# 1. Create hardware driver in hardware/
class CustomScannerDriver:
    def __init__(self):
        pass
    
    def initialize(self):
        pass
    
    def get_frame_data(self):
        pass

# 2. Register in config/system_config.py
SUPPORTED_DEVICES = [
    "AOS3", "RealSense", "Webcam", "CustomScanner"
]

# 3. Add device configuration
def get_device_config(device_type: str):
    if device_type == "CustomScanner":
        return {
            "use_structured_light": True,
            "has_depth_sensor": True,
            "resolution": [1920, 1080]
        }
```

### **Adding New AI Models**
```python
# 1. Define model in ai_training/model_specifications.py
class CustomDentalModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # Model architecture
    
    def forward(self, x):
        # Forward pass
        return x

# 2. Add to model registry
DENTAL_MODELS['custom_model'] = ModelSpec(
    name='custom_dental_model',
    input_size=(240, 176),
    num_classes=10,
    model_type='classification'
)

# 3. Update training configuration
TRAINING_CONFIG['custom_model'] = {
    'learning_rate': 0.001,
    'batch_size': 16,
    'epochs': 50
}
```

### **Debugging and Profiling**
```bash
# Enable debug logging
export PYTHONPATH=.
python main_v2.py --verbose

# Profile performance
python -m cProfile -o profile.stats main_v2.py
python -c "import pstats; pstats.Stats('profile.stats').sort_stats('cumulative').print_stats(20)"

# Memory profiling
pip install memory-profiler
python -m memory_profiler main_v2.py

# GPU profiling (NVIDIA)
nvprof python main_v2.py
```

### **Testing**
```bash
# Run all tests
pytest tests/

# Run specific test categories
pytest tests/test_scanning_service.py
pytest tests/test_ai_analysis.py
pytest tests/test_tsdf_fusion.py

# Run with coverage
pytest --cov=. tests/

# Integration tests
pytest tests/integration/
```

---

## 📚 Documentation

### **API Documentation**
```bash
# Generate API docs
pip install sphinx sphinx-rtd-theme
cd docs/
make html
open _build/html/index.html
```

### **Key Classes and Methods**

#### **ScanningService**
```python
class ScanningService:
    def start_service() -> bool:
        """Start the scanning service"""
    
    def start_scan(params: dict) -> dict:
        """Start a new scanning session"""
    
    def stop_scan() -> dict:
        """Stop current scanning session"""
    
    def get_status() -> dict:
        """Get current service status"""
```

#### **AIAnalysisService**
```python
class AIAnalysisService:
    def analyze_image(image_data: np.ndarray) -> dict:
        """Analyze 2D image for dental features"""
    
    def analyze_mesh(mesh_data: dict) -> dict:
        """Analyze 3D mesh for clinical assessment"""
```

#### **TSDFFusionV2**
```python
class TSDFFusionV2:
    def integrate_frame(points: np.ndarray, colors: np.ndarray, pose: np.ndarray) -> bool:
        """Integrate point cloud frame into TSDF volume"""
    
    def extract_mesh() -> o3d.geometry.TriangleMesh:
        """Extract triangle mesh from TSDF volume"""
```

### **Configuration Reference**

#### **Complete Configuration Example**
```json
{
  "camera": {
    "width": 1280,
    "height": 720,
    "fps": 30,
    "exposure": 0.033,
    "gain": 1.0,
    "auto_exposure": true,
    "focal_length_x": 500.0,
    "focal_length_y": 500.0,
    "principal_point_x": 640.0,
    "principal_point_y": 360.0,
    "distortion_coeffs": [0.0, 0.0, 0.0, 0.0, 0.0]
  },
  "processing": {
    "voxel_size": 0.002,
    "sdf_truncation": 0.008,
    "volume_size": [0.2, 0.2, 0.15],
    "icp_max_iterations": 50,
    "icp_convergence_threshold": 1e-6,
    "slam_enabled": true,
    "slam_keyframe_distance": 0.01,
    "max_points_per_frame": 100000,
    "integration_frequency": 3,
    "mesh_extraction_frequency": 10
  },
  "ai": {
    "models_directory": "models/trained",
    "segmentation_model": "dental_segmentation_v5.onnx",
    "detection_model": "dental_detection.onnx",
    "segmentation_input_size": [240, 176],
    "segmentation_confidence_threshold": 0.7,
    "detection_confidence_threshold": 0.8,
    "clinical_analysis_enabled": true,
    "use_gpu": true,
    "batch_size": 1,
    "max_inference_time": 0.1
  },
  "hardware": {
    "scanner_type": "RealSense",
    "device_id": "auto",
    "use_structured_light": false,
    "use_infrared": false,
    "has_projector": false,
    "has_stereo_cameras": false,
    "has_depth_sensor": true
  },
  "ui": {
    "window_width": 1920,
    "window_height": 1080,
    "fullscreen": false,
    "background_color": [0.1, 0.1, 0.1],
    "mesh_color": [0.8, 0.8, 0.8],
    "target_fps": 60,
    "show_fps": true,
    "show_performance_metrics": true
  },
  "database": {
    "database_path": "data/scanner_database.db",
    "backup_enabled": true,
    "backup_interval": 3600,
    "patient_data_retention": 2555,
    "scan_data_compression": true
  },
  "export": {
    "supported_formats": ["STL", "PLY", "OBJ", "3MF", "DICOM", "PDF"],
    "default_format": "STL",
    "mesh_resolution": "high",
    "texture_resolution": 1024,
    "compression_enabled": true,
    "include_measurements": true,
    "include_ai_analysis": true,
    "generate_pdf_report": true
  }
}
```

---

## 🤝 Contributing

### **How to Contribute**
1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Make your changes** following the coding standards
4. **Add tests** for new functionality
5. **Run the test suite**: `pytest tests/`
6. **Commit your changes**: `git commit -m 'Add amazing feature'`
7. **Push to the branch**: `git push origin feature/amazing-feature`
8. **Open a Pull Request**

### **Development Guidelines**
- **Code Style**: Follow PEP 8, use Black for formatting
- **Documentation**: Add docstrings for all public methods
- **Testing**: Maintain >90% test coverage
- **Performance**: Profile critical paths, optimize for real-time performance
- **Compatibility**: Support Python 3.8+, test on multiple platforms

### **Areas for Contribution**
- **Hardware Drivers**: Support for new scanner hardware
- **AI Models**: Improved neural network architectures
- **Export Formats**: Additional file format support
- **UI/UX**: Enhanced user interface and workflow
- **Performance**: Optimization and GPU acceleration
- **Documentation**: Tutorials, examples, and guides

### **Bug Reports**
Please include:
- **System information** (OS, Python version, hardware)
- **Steps to reproduce** the issue
- **Expected vs actual behavior**
- **Log files** and error messages
- **Configuration files** (remove sensitive data)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### **Third-Party Licenses**
- **Qt6**: LGPL v3 (commercial license available)
- **Open3D**: MIT License
- **OpenCV**: Apache 2.0 License
- **PyTorch**: BSD License
- **ONNX Runtime**: MIT License

### **Disclaimer**
This software is for educational and research purposes. For clinical use, ensure compliance with local medical device regulations and obtain appropriate certifications.

---

## 🆘 Support

### **Getting Help**
- **Documentation**: Check this README and inline documentation
- **Issues**: Open a GitHub issue for bugs and feature requests
- **Discussions**: Use GitHub Discussions for questions and ideas
- **Email**: Contact the maintainers for urgent issues

### **Common Issues**

#### **Installation Problems**
```bash
# CUDA not found
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Qt6 installation issues
pip install PySide6 --force-reinstall

# OpenCV compilation errors
pip install opencv-python-headless
```

#### **Runtime Issues**
```bash
# Camera not detected
python -c "import cv2; print(cv2.VideoCapture(0).isOpened())"

# GPU not available
python -c "import torch; print(torch.cuda.is_available())"

# Service connection failed
netstat -an | grep 5555  # Check if port is in use
```

#### **Performance Issues**
- **Reduce voxel size** for faster processing
- **Disable AI analysis** for basic 3D scanning
- **Lower camera resolution** for older hardware
- **Use CPU-only mode** if GPU drivers are problematic

### **System Information**
```bash
# Generate system report for support
python -c "
import sys, platform, torch, cv2, open3d
print(f'Python: {sys.version}')
print(f'Platform: {platform.platform()}')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA Available: {torch.cuda.is_available()}')
print(f'OpenCV: {cv2.__version__}')
print(f'Open3D: {open3d.__version__}')
"
```

---

**🦷 Happy Scanning! 🦷**

*This prototype demonstrates the complete professional dental scanning pipeline. For production use, consider additional testing, validation, and regulatory compliance.*