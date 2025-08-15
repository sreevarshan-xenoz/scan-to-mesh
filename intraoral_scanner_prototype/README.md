# Intraoral Scanner Prototype

A minimal open-source prototype of real-time 3D dental scanning software, inspired by our reverse engineering analysis of IntraoralScan 3.5.4.6.

## Features

- Real-time camera capture and processing
- Structured light pattern projection simulation
- Point cloud generation from stereo vision
- Basic mesh fusion using TSDF
- 3D visualization with Open3D
- Simple dental segmentation using computer vision

## Architecture

```
Camera Input → Stereo Processing → Point Cloud → TSDF Fusion → Mesh → Visualization
     ↓              ↓                 ↓            ↓          ↓         ↓
  OpenCV        Depth Estimation   Open3D      Volumetric   Marching   Real-time
  Capture       + Calibration      Points      Integration   Cubes     Display
```

## Requirements

- Python 3.8+
- OpenCV 4.5+
- Open3D 0.17+
- NumPy
- Camera (webcam or stereo setup)

## Quick Start

```bash
pip install -r requirements.txt
python main.py
```

## Components

- `camera_manager.py` - Camera capture and calibration
- `stereo_processor.py` - Depth estimation from stereo pairs
- `mesh_fusion.py` - TSDF-based mesh reconstruction
- `dental_segmentation.py` - Basic tooth detection
- `visualizer.py` - 3D mesh visualization
- `main.py` - Application orchestrator