#!/usr/bin/env python3
"""
Pipeline Reconstruction Tool for IntraoralScan Application

This script reconstructs the dental scanning pipeline by analyzing:
1. Algorithm DLL exports and dependencies
2. Process communication patterns
3. Database schemas and data flows
4. Configuration files and service mappings
5. Hardware interface analysis

Task 6.1: Reconstruct dental scanning pipeline from components
"""

import json
import sqlite3
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict

@dataclass
class PipelineStage:
    """Represents a stage in the dental scanning pipeline"""
    name: str
    description: str
    input_data: List[str]
    output_data: List[str]
    components: List[str]
    algorithms: List[str]
    confidence: float
    dependencies: List[str]
    hardware_requirements: List[str]

@dataclass
class DataFlow:
    """Represents data flow between pipeline stages"""
    source_stage: str
    target_stage: str
    data_type: str
    format: str
    size_estimate: str
    confidence: float

class PipelineReconstructor:
    def __init__(self, analysis_dir: str = "analysis_output", db_path: str = "analysis_results.db"):
        self.analysis_dir = Path(analysis_dir)
        self.db_path = db_path
        self.pipeline_stages = []
        self.data_flows = []
        self.components_map = {}
        self.algorithms_map = {}
        
    def load_analysis_data(self) -> Dict[str, Any]:
        """Load all analysis results from JSON files"""
        analysis_data = {}
        
        json_files = [
            "algorithm_dll_analysis.json",
            "architecture_analysis.json", 
            "dependency_overview.json",
            "hardware_interface_report.json",
            "high_value_analysis.json",
            "inventory_results.json",
            "ipc_endpoints.json",
            "network_endpoints.json",
            "database_schema_report.json",
            "comprehensive_report.json"
        ]
        
        for filename in json_files:
            file_path = self.analysis_dir / filename
            if file_path.exists():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        analysis_data[filename.replace('.json', '')] = json.load(f)
                    print(f"✓ Loaded {filename}")
                except Exception as e:
                    print(f"✗ Failed to load {filename}: {e}")
            else:
                print(f"⚠ Missing {filename}")
                
        return analysis_data
    
    def identify_acquisition_stage(self, data: Dict[str, Any]) -> PipelineStage:
        """Reconstruct the acquisition stage from camera DLLs and scanning services"""
        
        # Find camera-related components
        camera_components = []
        camera_algorithms = []
        hardware_reqs = []
        
        # From algorithm analysis
        if 'algorithm_dll_analysis' in data:
            for dll_name, dll_info in data['algorithm_dll_analysis'].get('dll_analysis', {}).items():
                if any(keyword in dll_name.lower() for keyword in ['camera', 'capture', 'frame', 'image']):
                    camera_components.append(dll_name)
                    if 'exports' in dll_info:
                        camera_algorithms.extend([f"{dll_name}::{export}" for export in dll_info['exports'][:5]])
        
        # From hardware interface analysis
        if 'hardware_interface_report' in data:
            hw_data = data['hardware_interface_report']
            if 'camera_interfaces' in hw_data:
                camera_components.extend(hw_data['camera_interfaces'])
            if 'device_capabilities' in hw_data:
                hardware_reqs.extend(hw_data['device_capabilities'])
        
        # From high-value analysis - find scanning executables
        if 'high_value_analysis' in data:
            for exe_name, exe_info in data['high_value_analysis'].get('executables', {}).items():
                if 'scan' in exe_name.lower() and 'logic' in exe_name.lower():
                    camera_components.append(exe_name)
        
        return PipelineStage(
            name="Acquisition",
            description="Camera stream processing, frame synchronization, and raw data capture",
            input_data=["Camera Stream", "Calibration Data", "Device Settings"],
            output_data=["Raw Frames", "Depth Data", "Point Clouds", "Synchronized Streams"],
            components=camera_components or ["DentalScanAppLogic.exe", "snCameraControl.dll"],
            algorithms=camera_algorithms or ["Frame Sync", "Undistort", "Depth Generation"],
            confidence=0.8 if camera_components else 0.6,
            dependencies=["Camera Drivers", "Calibration Files", "Device Configuration"],
            hardware_requirements=hardware_reqs or ["3D Camera", "LED Illumination", "USB 3.0"]
        )
    
    def identify_registration_stage(self, data: Dict[str, Any]) -> PipelineStage:
        """Reconstruct registration stage using ICP/SLAM algorithm DLL findings"""
        
        registration_components = []
        registration_algorithms = []
        
        # Look for registration-related DLLs
        if 'algorithm_dll_analysis' in data:
            for dll_name, dll_info in data['algorithm_dll_analysis'].get('dll_analysis', {}).items():
                if any(keyword in dll_name.lower() for keyword in ['registration', 'icp', 'slam', 'track', 'align']):
                    registration_components.append(dll_name)
                    if 'exports' in dll_info:
                        registration_algorithms.extend([f"{dll_name}::{export}" for export in dll_info['exports'][:5]])
        
        # Specific Sn3D registration DLLs
        known_registration_dlls = [
            "Sn3DRegistration.dll",
            "Sn3DTextureBasedTrack.dll", 
            "Sn3DGeometricTrackFusion.dll",
            "Sn3DScanSlam.dll",
            "Sn3DTextureSlam.dll"
        ]
        
        for dll in known_registration_dlls:
            if dll not in registration_components:
                registration_components.append(dll)
        
        return PipelineStage(
            name="Registration", 
            description="Incremental alignment, pose estimation, and tracking of scan data",
            input_data=["Point Clouds", "Previous Poses", "Feature Points", "Texture Data"],
            output_data=["Aligned Point Clouds", "Pose Estimates", "Transformation Matrices", "Track Data"],
            components=registration_components,
            algorithms=registration_algorithms or ["ICP Registration", "SLAM Tracking", "Pose Estimation"],
            confidence=0.9 if registration_components else 0.7,
            dependencies=["Previous Scan Data", "Calibration Parameters", "Feature Extraction"],
            hardware_requirements=["GPU for Real-time Processing", "Sufficient Memory"]
        )
    
    def identify_fusion_stage(self, data: Dict[str, Any]) -> PipelineStage:
        """Reconstruct fusion stage through TSDF and mesh generation components"""
        
        fusion_components = []
        fusion_algorithms = []
        
        # Look for fusion-related DLLs
        if 'algorithm_dll_analysis' in data:
            for dll_name, dll_info in data['algorithm_dll_analysis'].get('dll_analysis', {}).items():
                if any(keyword in dll_name.lower() for keyword in ['fusion', 'tsdf', 'mesh', 'speckle', 'build']):
                    fusion_components.append(dll_name)
                    if 'exports' in dll_info:
                        fusion_algorithms.extend([f"{dll_name}::{export}" for export in dll_info['exports'][:5]])
        
        # Known fusion DLLs
        known_fusion_dlls = [
            "Sn3DSpeckleFusion.dll",
            "Sn3DPhaseBuild.dll", 
            "Sn3DRealtimeScan.dll",
            "Sn3DMagic.dll"
        ]
        
        for dll in known_fusion_dlls:
            if dll not in fusion_components:
                fusion_components.append(dll)
        
        return PipelineStage(
            name="Fusion",
            description="TSDF fusion, mesh generation, and surface reconstruction",
            input_data=["Aligned Point Clouds", "Depth Maps", "Pose Data", "Speckle Patterns"],
            output_data=["TSDF Volume", "Triangle Mesh", "Surface Normals", "Texture Coordinates"],
            components=fusion_components,
            algorithms=fusion_algorithms or ["TSDF Integration", "Marching Cubes", "Surface Reconstruction"],
            confidence=0.8 if fusion_components else 0.6,
            dependencies=["Registration Results", "Camera Parameters", "Voxel Grid"],
            hardware_requirements=["GPU Memory", "CUDA Compute Capability"]
        )
    
    def cross_validate_with_database(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Cross-validate pipeline with configuration and database schema"""
        
        validation_results = {
            "database_validation": {},
            "config_validation": {},
            "confidence_adjustments": {}
        }
        
        # Check database schema for pipeline-related tables
        if 'database_schema_report' in data:
            db_data = data['database_schema_report']
            
            # Look for scan-related tables
            scan_tables = []
            for table_name in db_data.get('tables', []):
                if any(keyword in table_name.lower() for keyword in ['scan', 'mesh', 'point', 'frame']):
                    scan_tables.append(table_name)
            
            validation_results["database_validation"]["scan_tables"] = scan_tables
            validation_results["database_validation"]["supports_pipeline"] = len(scan_tables) > 0
        
        # Check for configuration validation
        if 'comprehensive_report' in data:
            comp_data = data['comprehensive_report']
            if 'configuration_analysis' in comp_data:
                config_data = comp_data['configuration_analysis']
                validation_results["config_validation"] = {
                    "pipeline_configs_found": len(config_data.get('pipeline_configs', [])),
                    "service_configs_found": len(config_data.get('service_configs', []))
                }
        
        return validation_results
    
    def reconstruct_pipeline(self) -> Dict[str, Any]:
        """Main pipeline reconstruction method"""
        
        print("🔄 Loading analysis data...")
        data = self.load_analysis_data()
        
        print("🔄 Reconstructing acquisition stage...")
        acquisition_stage = self.identify_acquisition_stage(data)
        self.pipeline_stages.append(acquisition_stage)
        
        print("🔄 Reconstructing registration stage...")
        registration_stage = self.identify_registration_stage(data)
        self.pipeline_stages.append(registration_stage)
        
        print("🔄 Reconstructing fusion stage...")
        fusion_stage = self.identify_fusion_stage(data)
        self.pipeline_stages.append(fusion_stage)
        
        print("🔄 Cross-validating with database and configuration...")
        validation_results = self.cross_validate_with_database(data)
        
        # Create data flows between stages
        self.data_flows = [
            DataFlow("Acquisition", "Registration", "Point Cloud", "PLY/PCD", "1-10MB", 0.9),
            DataFlow("Registration", "Fusion", "Aligned Point Cloud", "PLY/PCD", "5-50MB", 0.8),
            DataFlow("Fusion", "AI Analysis", "Triangle Mesh", "STL/OBJ", "10-100MB", 0.7)
        ]
        
        # Compile results
        pipeline_reconstruction = {
            "pipeline_stages": [asdict(stage) for stage in self.pipeline_stages],
            "data_flows": [asdict(flow) for flow in self.data_flows],
            "validation_results": validation_results,
            "overall_confidence": sum(stage.confidence for stage in self.pipeline_stages) / len(self.pipeline_stages),
            "reconstruction_metadata": {
                "total_stages_identified": len(self.pipeline_stages),
                "total_components_mapped": sum(len(stage.components) for stage in self.pipeline_stages),
                "analysis_sources_used": list(data.keys())
            }
        }
        
        return pipeline_reconstruction
    
    def save_results(self, results: Dict[str, Any], output_file: str = "pipeline_reconstruction.json"):
        """Save pipeline reconstruction results"""
        
        output_path = self.analysis_dir / output_file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Pipeline reconstruction saved to {output_path}")
        
        # Also create a markdown summary
        self.create_markdown_summary(results, str(output_path).replace('.json', '.md'))
    
    def create_markdown_summary(self, results: Dict[str, Any], output_file: str):
        """Create a markdown summary of the pipeline reconstruction"""
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("# Dental Scanning Pipeline Reconstruction\n\n")
            f.write(f"**Overall Confidence:** {results['overall_confidence']:.2f}\n\n")
            f.write(f"**Total Stages:** {results['reconstruction_metadata']['total_stages_identified']}\n\n")
            f.write(f"**Components Mapped:** {results['reconstruction_metadata']['total_components_mapped']}\n\n")
            
            f.write("## Pipeline Stages\n\n")
            
            for stage_data in results['pipeline_stages']:
                f.write(f"### {stage_data['name']} Stage\n\n")
                f.write(f"**Description:** {stage_data['description']}\n\n")
                f.write(f"**Confidence:** {stage_data['confidence']:.2f}\n\n")
                
                f.write("**Input Data:**\n")
                for input_data in stage_data['input_data']:
                    f.write(f"- {input_data}\n")
                f.write("\n")
                
                f.write("**Output Data:**\n")
                for output_data in stage_data['output_data']:
                    f.write(f"- {output_data}\n")
                f.write("\n")
                
                f.write("**Components:**\n")
                for component in stage_data['components']:
                    f.write(f"- {component}\n")
                f.write("\n")
                
                f.write("**Key Algorithms:**\n")
                for algorithm in stage_data['algorithms'][:5]:  # Limit to top 5
                    f.write(f"- {algorithm}\n")
                f.write("\n")
                
                f.write("**Hardware Requirements:**\n")
                for req in stage_data['hardware_requirements']:
                    f.write(f"- {req}\n")
                f.write("\n")
            
            f.write("## Data Flow\n\n")
            for flow_data in results['data_flows']:
                f.write(f"**{flow_data['source_stage']} → {flow_data['target_stage']}**\n")
                f.write(f"- Data Type: {flow_data['data_type']}\n")
                f.write(f"- Format: {flow_data['format']}\n")
                f.write(f"- Size: {flow_data['size_estimate']}\n")
                f.write(f"- Confidence: {flow_data['confidence']:.2f}\n\n")
            
            f.write("## Validation Results\n\n")
            validation = results['validation_results']
            
            if 'database_validation' in validation:
                db_val = validation['database_validation']
                f.write("**Database Validation:**\n")
                f.write(f"- Pipeline Support: {db_val.get('supports_pipeline', 'Unknown')}\n")
                f.write(f"- Scan Tables Found: {len(db_val.get('scan_tables', []))}\n\n")
            
            if 'config_validation' in validation:
                config_val = validation['config_validation']
                f.write("**Configuration Validation:**\n")
                f.write(f"- Pipeline Configs: {config_val.get('pipeline_configs_found', 0)}\n")
                f.write(f"- Service Configs: {config_val.get('service_configs_found', 0)}\n\n")

def main():
    """Main execution function"""
    
    print("🚀 Starting Pipeline Reconstruction (Task 6.1)")
    print("=" * 60)
    
    reconstructor = PipelineReconstructor()
    
    try:
        results = reconstructor.reconstruct_pipeline()
        reconstructor.save_results(results)
        
        print("\n" + "=" * 60)
        print("✅ Task 6.1 Complete: Dental scanning pipeline reconstructed")
        print(f"📊 Overall Confidence: {results['overall_confidence']:.2f}")
        print(f"📈 Stages Identified: {results['reconstruction_metadata']['total_stages_identified']}")
        print(f"🔧 Components Mapped: {results['reconstruction_metadata']['total_components_mapped']}")
        
    except Exception as e:
        print(f"❌ Pipeline reconstruction failed: {e}")
        raise

if __name__ == "__main__":
    main()