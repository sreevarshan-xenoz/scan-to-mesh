"""
System Configuration - Professional scanner configuration based on analysis
Implements the configuration patterns discovered in IntraoralScan analysis
Enhanced with Meshroom integration for professional 3D reconstruction
"""

import json
import os
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Any
from pathlib import Path

@dataclass
class CameraConfig:
    """Camera configuration matching professional scanner specs"""
    width: int = 1280
    height: int = 720
    fps: int = 30
    exposure: float = 0.033  # 33ms exposure
    gain: float = 1.0
    auto_exposure: bool = True
    
    # Structured light parameters
    pattern_frequency: int = 16  # Pattern stripe frequency
    pattern_phases: int = 8      # Number of phase shifts
    projection_power: float = 0.8 # LED/projector power
    
    # Calibration parameters (will be loaded from calibration)
    focal_length_x: float = 500.0
    focal_length_y: float = 500.0
    principal_point_x: float = 640.0
    principal_point_y: float = 360.0
    distortion_coeffs: List[float] = None
    
    def __post_init__(self):
        if self.distortion_coeffs is None:
            self.distortion_coeffs = [0.0, 0.0, 0.0, 0.0, 0.0]

@dataclass
class ProcessingConfig:
    """3D processing configuration based on professional system analysis"""
    # TSDF Volume parameters (matching Sn3DSpeckleFusion.dll behavior)
    voxel_size: float = 0.002  # 2mm voxels (professional grade)
    sdf_truncation: float = 0.008  # 8mm truncation distance
    volume_size: List[float] = None  # [0.2, 0.2, 0.15] meters (20x20x15cm)
    
    # Enhanced GPU TSDF configuration
    use_enhanced_gpu_tsdf: bool = True  # Use PyTorch GPU acceleration
    gpu_memory_limit_mb: float = 1024.0  # 1GB GPU memory limit
    
    # Registration parameters (matching Sn3DRegistration.dll)
    icp_max_iterations: int = 50
    icp_convergence_threshold: float = 1e-6
    icp_max_correspondence_distance: float = 0.005  # 5mm
    
    # SLAM parameters (matching Sn3DScanSlam.dll)
    slam_enabled: bool = True
    slam_keyframe_distance: float = 0.01  # 1cm keyframe spacing
    slam_loop_closure_threshold: float = 0.02  # 2cm loop closure
    
    # Performance parameters
    max_points_per_frame: int = 100000
    integration_frequency: int = 3  # Integrate every 3rd frame
    mesh_extraction_frequency: int = 10  # Extract mesh every 10th integration
    
    def __post_init__(self):
        if self.volume_size is None:
            self.volume_size = [0.2, 0.2, 0.15]  # Default dental arch size

@dataclass
class AIConfig:
    """AI configuration matching professional model specifications"""
    # Model paths and configurations
    models_directory: str = "models"
    
    # Segmentation model config (matching analysis findings)
    segmentation_model: str = "dental_segmentation_v5.onnx"
    segmentation_input_size: List[int] = None  # [240, 176] from analysis
    segmentation_confidence_threshold: float = 0.7
    
    # Detection model config
    detection_model: str = "dental_detection_v3.onnx"
    detection_confidence_threshold: float = 0.8
    
    # Clinical analysis config
    clinical_analysis_enabled: bool = True
    pathology_detection_enabled: bool = True
    quality_assessment_enabled: bool = True
    
    # Performance settings
    use_gpu: bool = True
    batch_size: int = 1  # Real-time processing
    max_inference_time: float = 0.1  # 100ms max inference time
    
    def __post_init__(self):
        if self.segmentation_input_size is None:
            self.segmentation_input_size = [240, 176]  # From analysis

@dataclass
class HardwareConfig:
    """Hardware configuration for different scanner types"""
    # Scanner type (matching analysis device types)
    scanner_type: str = "AOS3"  # AOS3, AOS3-LAB, RealSense, Webcam
    
    # Device-specific settings
    device_id: str = "auto"  # Auto-detect or specific device ID
    use_structured_light: bool = True
    use_infrared: bool = False  # IR scanning capability
    
    # Supported scanner configurations from analysis
    supported_devices: List[str] = None
    
    # Hardware capabilities
    has_projector: bool = False
    has_stereo_cameras: bool = False
    has_depth_sensor: bool = True
    
    def __post_init__(self):
        if self.supported_devices is None:
            self.supported_devices = [
                "AOS3", "AOS3-LAB", "AOS", "A3S", "A3I", "A3W", 
                "RealSense", "Webcam", "StereoCamera"
            ]

@dataclass
class UIConfig:
    """UI configuration matching professional interface"""
    # Window settings
    window_width: int = 1920
    window_height: int = 1080
    fullscreen: bool = False
    
    # Visualization settings
    background_color: List[float] = None  # [0.1, 0.1, 0.1] dark gray
    mesh_color: List[float] = None       # [0.8, 0.8, 0.8] light gray
    point_size: float = 2.0
    
    # Real-time display settings
    target_fps: int = 60
    show_fps: bool = True
    show_performance_metrics: bool = True
    
    # Clinical display settings
    show_tooth_numbers: bool = True
    show_quality_indicators: bool = True
    show_measurements: bool = True
    
    def __post_init__(self):
        if self.background_color is None:
            self.background_color = [0.1, 0.1, 0.1]
        if self.mesh_color is None:
            self.mesh_color = [0.8, 0.8, 0.8]

@dataclass
class MeshroomConfig:
    """Meshroom integration configuration for professional 3D reconstruction"""
    # Meshroom installation settings
    enabled: bool = True
    meshroom_path: str = "/opt/Meshroom-2023.3.0"
    meshroom_batch_executable: str = "meshroom_batch"
    
    # Project settings
    temp_directory: str = "/tmp/meshroom_dental"
    project_base_name: str = "dental_scan"
    auto_cleanup: bool = True
    max_concurrent_sessions: int = 2
    
    # Quality presets
    quality_presets: Dict[str, Dict[str, Any]] = None
    default_preset: str = "dental_scan"
    
    # Keyframe detection settings
    keyframe_threshold: float = 0.15  # Similarity threshold for keyframe detection
    min_keyframe_distance: int = 10   # Minimum frames between keyframes
    max_keyframes_per_session: int = 200
    
    # Processing settings
    mesh_resolution: str = "high"  # low, medium, high, ultra
    texture_resolution: int = 1024
    enable_texturing: bool = True
    enable_mesh_simplification: bool = False
    
    # Performance settings
    cpu_cores: int = -1  # -1 for auto-detect
    memory_limit_gb: int = 8
    processing_timeout: int = 3600  # 1 hour timeout
    
    def __post_init__(self):
        if self.quality_presets is None:
            self.quality_presets = {
                "dental_scan": {
                    "description": "Optimized for intraoral dental scanning",
                    "keyframe_threshold": 0.15,
                    "min_keyframe_distance": 10,
                    "max_images": 200,
                    "mesh_resolution": "high",
                    "enable_texturing": True,
                    "processing_nodes": [
                        "CameraInit", "FeatureExtraction", "ImageMatching",
                        "FeatureMatching", "StructureFromMotion", "PrepareDenseScene",
                        "DepthMap", "DepthMapFilter", "Meshing", "MeshFiltering",
                        "Texturing"
                    ]
                },
                "real_time": {
                    "description": "Fast processing for real-time preview",
                    "keyframe_threshold": 0.25,
                    "min_keyframe_distance": 5,
                    "max_images": 50,
                    "mesh_resolution": "medium",
                    "enable_texturing": False,
                    "processing_nodes": [
                        "CameraInit", "FeatureExtraction", "ImageMatching",
                        "FeatureMatching", "StructureFromMotion", "PrepareDenseScene",
                        "DepthMap", "DepthMapFilter", "Meshing"
                    ]
                },
                "high_quality": {
                    "description": "Maximum quality for final reconstruction",
                    "keyframe_threshold": 0.10,
                    "min_keyframe_distance": 15,
                    "max_images": 500,
                    "mesh_resolution": "ultra",
                    "enable_texturing": True,
                    "processing_nodes": [
                        "CameraInit", "FeatureExtraction", "ImageMatching",
                        "FeatureMatching", "StructureFromMotion", "PrepareDenseScene",
                        "DepthMap", "DepthMapFilter", "Meshing", "MeshFiltering",
                        "MeshDecimate", "Texturing"
                    ]
                }
            }

@dataclass
class DatabaseConfig:
    """Database configuration for clinical workflow"""
    # Database settings
    database_path: str = "data/scanner_database.db"
    backup_enabled: bool = True
    backup_interval: int = 3600  # 1 hour
    
    # Clinical data settings
    patient_data_retention: int = 2555  # 7 years in days
    scan_data_compression: bool = True
    auto_export_enabled: bool = False
    
    # Sync settings (matching DentalNetwork analysis)
    cloud_sync_enabled: bool = False
    sync_interval: int = 300  # 5 minutes
    offline_mode: bool = True

@dataclass
class ExportConfig:
    """Export configuration for various formats"""
    # Export formats (matching analysis findings)
    supported_formats: List[str] = None
    default_format: str = "STL"
    
    # Export quality settings
    mesh_resolution: str = "high"  # low, medium, high, ultra
    texture_resolution: int = 1024
    compression_enabled: bool = True
    
    # Clinical export settings
    include_measurements: bool = True
    include_ai_analysis: bool = True
    generate_pdf_report: bool = True
    
    # DICOM settings
    dicom_enabled: bool = True
    dicom_institution: str = "Dental Clinic"
    dicom_manufacturer: str = "OpenSource Scanner"
    
    def __post_init__(self):
        if self.supported_formats is None:
            self.supported_formats = ["STL", "PLY", "OBJ", "3MF", "DICOM", "PDF"]

@dataclass
class SystemFlagsConfig:
    simulation_mode: bool = True  # run without real camera
    enable_meshroom: bool = False  # disabled for first tests
    enable_ai: bool = False        # AI disabled initially
    save_intermediate: bool = False
    verbose_logging: bool = True

@dataclass
class SystemConfig:
    """Main system configuration manager"""
    # Use default_factory to avoid mutable default issues
    camera: CameraConfig = field(default_factory=CameraConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    ai: AIConfig = field(default_factory=AIConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    meshroom: MeshroomConfig = field(default_factory=MeshroomConfig)
    flags: SystemFlagsConfig = field(default_factory=SystemFlagsConfig)

    def validate_config(self):
        issues = []
        
        # Validate camera configuration
        if self.camera.width <= 0 or self.camera.height <= 0:
            issues.append("Invalid camera resolution")
        
        if self.camera.fps <= 0 or self.camera.fps > 120:
            issues.append("Invalid camera FPS")
        
        # Validate processing configuration
        if self.processing.voxel_size <= 0:
            issues.append("Invalid voxel size")
        
        if self.processing.sdf_truncation <= 0:
            issues.append("Invalid SDF truncation distance")
        
        # Validate AI configuration
        if self.ai.models_directory and not os.path.exists(self.ai.models_directory):
            issues.append(f"AI models directory not found: {self.ai.models_directory}")
        
        # Validate hardware configuration
        if self.hardware.scanner_type not in self.hardware.supported_devices:
            issues.append(f"Unsupported scanner type: {self.hardware.scanner_type}")
        
        return issues

    def save_config(self, path: str = "config/system_config.json"):
        data = {
            'camera': asdict(self.camera),
            'processing': asdict(self.processing),
            'ai': asdict(self.ai),
            'hardware': asdict(self.hardware),
            'ui': asdict(self.ui),
            'database': asdict(self.database),
            'export': asdict(self.export),
            'meshroom': asdict(self.meshroom),
            'flags': asdict(self.flags)
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

_config_instance: SystemConfig = None

def get_config() -> SystemConfig:
    global _config_instance
    if _config_instance is None:
        _config_instance = SystemConfig()
    return _config_instance

def reload_config():
    """Reload configuration from file"""
    global config
    config.load_config()

def save_config():
    """Save current configuration to file"""
    global config
    config.save_config()