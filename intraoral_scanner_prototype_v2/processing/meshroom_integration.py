"""
Meshroom Integration - Professional 3D Reconstruction Pipeline
Integrates AliceVision Meshroom for enhanced mesh reconstruction
"""

import os
import subprocess
import json
import tempfile
import shutil
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import numpy as np
import time

class MeshroomIntegration:
    """
    Professional Meshroom integration for enhanced 3D reconstruction
    
    Combines our real-time SLAM with Meshroom's professional photogrammetry pipeline
    for high-quality mesh reconstruction and texture mapping.
    """
    
    def __init__(self, meshroom_path: Optional[str] = None):
        """
        Initialize Meshroom integration
        
        Args:
            meshroom_path: Path to Meshroom installation (auto-detect if None)
        """
        self.meshroom_path = self._find_meshroom_installation(meshroom_path)
        self.meshroom_cli = None
        self.temp_dir = None
        self.current_project = None
        
        # Meshroom pipeline configurations
        self.pipeline_configs = {
            'dental_scan': {
                'description': 'Optimized for intraoral dental scanning',
                'nodes': [
                    'CameraInit', 'FeatureExtraction', 'ImageMatching',
                    'FeatureMatching', 'StructureFromMotion', 'PrepareDenseScene',
                    'DepthMap', 'DepthMapFilter', 'Meshing', 'MeshFiltering',
                    'Texturing'
                ],
                'parameters': {
                    'FeatureExtraction': {
                        'describerTypes': ['sift'],
                        'maxNbFeatures': 10000,
                        'contrastFiltering': 'GridSort'
                    },
                    'DepthMap': {
                        'downscale': 2,
                        'minViewAngle': 2.0,
                        'maxViewAngle': 70.0
                    },
                    'Meshing': {
                        'estimateSpaceFromSfM': True,
                        'maxInputPoints': 50000000,
                        'maxPoints': 5000000
                    },
                    'Texturing': {
                        'textureSide': 8192,
                        'downscale': 2,
                        'unwrapMethod': 'Basic'
                    }
                }
            },
            'real_time': {
                'description': 'Fast reconstruction for real-time feedback',
                'nodes': [
                    'CameraInit', 'FeatureExtraction', 'ImageMatching',
                    'FeatureMatching', 'StructureFromMotion', 'Meshing'
                ],
                'parameters': {
                    'FeatureExtraction': {
                        'maxNbFeatures': 5000,
                        'contrastFiltering': 'NoFiltering'
                    },
                    'Meshing': {
                        'estimateSpaceFromSfM': True,
                        'maxPoints': 1000000
                    }
                }
            },
            'high_quality': {
                'description': 'Maximum quality for final deliverables',
                'nodes': [
                    'CameraInit', 'FeatureExtraction', 'ImageMatching',
                    'FeatureMatching', 'StructureFromMotion', 'PrepareDenseScene',
                    'DepthMap', 'DepthMapFilter', 'Meshing', 'MeshFiltering',
                    'MeshResampling', 'Texturing', 'MeshDenoising'
                ],
                'parameters': {
                    'FeatureExtraction': {
                        'maxNbFeatures': 50000,
                        'contrastFiltering': 'GridSort'
                    },
                    'DepthMap': {
                        'downscale': 1,
                        'minViewAngle': 1.0,
                        'maxViewAngle': 85.0
                    },
                    'Meshing': {
                        'estimateSpaceFromSfM': True,
                        'maxInputPoints': 100000000,
                        'maxPoints': 10000000
                    },
                    'Texturing': {
                        'textureSide': 16384,
                        'downscale': 1,
                        'unwrapMethod': 'LSCM'
                    }
                }
            }
        }
        
        # Initialize if Meshroom found
        if self.meshroom_path:
            self._initialize_meshroom()
        
    def _find_meshroom_installation(self, custom_path: Optional[str] = None) -> Optional[str]:
        """Find Meshroom installation path"""
        if custom_path and os.path.exists(custom_path):
            return custom_path
        
        # Common installation paths
        search_paths = [
            '/opt/Meshroom',
            '/usr/local/bin/Meshroom',
            '~/Applications/Meshroom',
            'C:/Program Files/Meshroom',
            '~/meshroom',
            './meshroom'
        ]
        
        for path in search_paths:
            expanded_path = os.path.expanduser(path)
            if os.path.exists(expanded_path):
                meshroom_cli = os.path.join(expanded_path, 'meshroom_batch')
                if os.path.exists(meshroom_cli):
                    return expanded_path
        
        # Try to find in PATH
        try:
            result = subprocess.run(['which', 'meshroom_batch'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                return os.path.dirname(result.stdout.strip())
        except:
            pass
        
        print("⚠️  Meshroom not found. Please install from: https://github.com/alicevision/Meshroom")
        return None
    
    def _initialize_meshroom(self) -> bool:
        """Initialize Meshroom CLI interface"""
        try:
            self.meshroom_cli = os.path.join(self.meshroom_path, 'meshroom_batch')
            
            # Test Meshroom installation
            result = subprocess.run([self.meshroom_cli, '--version'], 
                                  capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                print(f"✅ Meshroom initialized successfully")
                print(f"   Version: {result.stdout.strip()}")
                print(f"   Path: {self.meshroom_path}")
                return True
            else:
                print(f"❌ Meshroom CLI test failed: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"❌ Failed to initialize Meshroom: {e}")
            return False
    
    def is_available(self) -> bool:
        """Check if Meshroom is available for use"""
        return self.meshroom_cli is not None and os.path.exists(self.meshroom_cli)
    
    def create_project(self, name: str, pipeline_type: str = 'dental_scan') -> bool:
        """
        Create new Meshroom project
        
        Args:
            name: Project name
            pipeline_type: Pipeline configuration ('dental_scan', 'real_time', 'high_quality')
            
        Returns:
            bool: True if project created successfully
        """
        if not self.is_available():
            print("❌ Meshroom not available")
            return False
        
        try:
            # Create temporary project directory
            self.temp_dir = tempfile.mkdtemp(prefix=f'meshroom_{name}_')
            
            # Create project structure
            project_structure = {
                'images': os.path.join(self.temp_dir, 'images'),
                'output': os.path.join(self.temp_dir, 'output'),
                'cache': os.path.join(self.temp_dir, 'cache'),
                'logs': os.path.join(self.temp_dir, 'logs')
            }
            
            for dir_path in project_structure.values():
                os.makedirs(dir_path, exist_ok=True)
            
            # Store project configuration
            self.current_project = {
                'name': name,
                'pipeline_type': pipeline_type,
                'structure': project_structure,
                'config': self.pipeline_configs.get(pipeline_type, self.pipeline_configs['dental_scan'])
            }
            
            print(f"✅ Meshroom project '{name}' created")
            print(f"   Pipeline: {pipeline_type}")
            print(f"   Directory: {self.temp_dir}")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to create Meshroom project: {e}")
            return False
    
    def add_images_from_slam(self, slam_frames: List[Dict[str, Any]]) -> bool:
        """
        Add images from SLAM tracking to Meshroom project
        
        Args:
            slam_frames: List of SLAM frames with images and poses
            
        Returns:
            bool: True if images added successfully
        """
        if not self.current_project:
            print("❌ No active Meshroom project")
            return False
        
        try:
            images_dir = self.current_project['structure']['images']
            added_count = 0
            
            for i, frame in enumerate(slam_frames):
                if 'color_image' in frame:
                    # Save color image
                    image_path = os.path.join(images_dir, f'frame_{i:06d}.jpg')
                    
                    # Save image (assuming OpenCV format)
                    import cv2
                    cv2.imwrite(image_path, frame['color_image'])
                    
                    # Save camera pose if available
                    if 'pose' in frame:
                        pose_path = os.path.join(images_dir, f'frame_{i:06d}_pose.json')
                        pose_data = {
                            'pose_matrix': frame['pose'].tolist(),
                            'timestamp': frame.get('timestamp', i),
                            'frame_id': i
                        }
                        with open(pose_path, 'w') as f:
                            json.dump(pose_data, f, indent=2)
                    
                    added_count += 1
            
            print(f"✅ Added {added_count} images from SLAM to Meshroom project")
            return added_count > 0
            
        except Exception as e:
            print(f"❌ Failed to add SLAM images: {e}")
            return False
    
    def add_images_from_directory(self, image_directory: str) -> bool:
        """
        Add images from directory to Meshroom project
        
        Args:
            image_directory: Path to directory containing images
            
        Returns:
            bool: True if images added successfully
        """
        if not self.current_project:
            print("❌ No active Meshroom project")
            return False
        
        try:
            images_dir = self.current_project['structure']['images']
            
            # Copy images to project
            image_extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp'}
            copied_count = 0
            
            for file_name in os.listdir(image_directory):
                file_ext = os.path.splitext(file_name)[1].lower()
                if file_ext in image_extensions:
                    src_path = os.path.join(image_directory, file_name)
                    dst_path = os.path.join(images_dir, file_name)
                    shutil.copy2(src_path, dst_path)
                    copied_count += 1
            
            print(f"✅ Added {copied_count} images from directory to Meshroom project")
            return copied_count > 0
            
        except Exception as e:
            print(f"❌ Failed to add images from directory: {e}")
            return False
    
    def run_reconstruction(self, quality_preset: str = None) -> Optional[Dict[str, str]]:
        """
        Run Meshroom 3D reconstruction
        
        Args:
            quality_preset: Override pipeline quality ('real_time', 'dental_scan', 'high_quality')
            
        Returns:
            Dict with output file paths or None if failed
        """
        if not self.current_project:
            print("❌ No active Meshroom project")
            return None
        
        try:
            # Use specified preset or project default
            pipeline_type = quality_preset or self.current_project['pipeline_type']
            config = self.pipeline_configs.get(pipeline_type, self.pipeline_configs['dental_scan'])
            
            # Prepare Meshroom command
            images_dir = self.current_project['structure']['images']
            output_dir = self.current_project['structure']['output']
            cache_dir = self.current_project['structure']['cache']
            
            cmd = [
                self.meshroom_cli,
                '--input', images_dir,
                '--output', output_dir,
                '--cache', cache_dir
            ]
            
            # Add pipeline-specific parameters
            if pipeline_type == 'real_time':
                cmd.extend(['--pipeline', 'photogrammetry_fast'])
            elif pipeline_type == 'high_quality':
                cmd.extend(['--pipeline', 'photogrammetry_hq'])
            else:
                cmd.extend(['--pipeline', 'photogrammetry'])
            
            print(f"🚀 Starting Meshroom reconstruction ({pipeline_type})...")
            print(f"   Command: {' '.join(cmd)}")
            
            # Run Meshroom reconstruction
            start_time = time.time()
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=self.temp_dir
            )
            
            duration = time.time() - start_time
            
            if result.returncode == 0:
                # Find output files
                output_files = self._find_output_files(output_dir)
                
                print(f"✅ Meshroom reconstruction completed in {duration:.1f}s")
                print(f"   Output files: {len(output_files)}")
                
                return output_files
            else:
                print(f"❌ Meshroom reconstruction failed")
                print(f"   Error: {result.stderr}")
                return None
                
        except Exception as e:
            print(f"❌ Meshroom reconstruction error: {e}")
            return None
    
    def _find_output_files(self, output_dir: str) -> Dict[str, str]:
        """Find and categorize output files from Meshroom"""
        output_files = {}
        
        # Common Meshroom output patterns
        file_patterns = {
            'mesh': ['*.obj', '*.ply', '*.abc'],
            'texture': ['*.jpg', '*.png', '*.exr'],
            'cameras': ['*.sfm', '*.json'],
            'point_cloud': ['*.ply', '*.abc'],
            'dense_point_cloud': ['*_dense.ply']
        }
        
        try:
            for root, dirs, files in os.walk(output_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    file_ext = os.path.splitext(file)[1].lower()
                    
                    # Categorize files
                    if file_ext in ['.obj', '.ply'] and 'mesh' in file.lower():
                        output_files['mesh'] = file_path
                    elif file_ext in ['.jpg', '.png'] and 'texture' in file.lower():
                        output_files['texture'] = file_path
                    elif file_ext == '.sfm':
                        output_files['cameras'] = file_path
                    elif 'dense' in file.lower() and file_ext == '.ply':
                        output_files['dense_point_cloud'] = file_path
                    elif file_ext == '.ply' and 'point_cloud' not in output_files:
                        output_files['point_cloud'] = file_path
                        
        except Exception as e:
            print(f"Warning: Error scanning output files: {e}")
        
        return output_files
    
    def get_reconstruction_quality_metrics(self, output_files: Dict[str, str]) -> Dict[str, Any]:
        """
        Analyze reconstruction quality metrics
        
        Args:
            output_files: Dictionary of output file paths
            
        Returns:
            Dictionary of quality metrics
        """
        metrics = {
            'files_generated': len(output_files),
            'mesh_available': 'mesh' in output_files,
            'texture_available': 'texture' in output_files,
            'cameras_available': 'cameras' in output_files,
            'file_sizes': {}
        }
        
        try:
            # Analyze file sizes
            for file_type, file_path in output_files.items():
                if os.path.exists(file_path):
                    size_mb = os.path.getsize(file_path) / (1024 * 1024)
                    metrics['file_sizes'][file_type] = f"{size_mb:.1f} MB"
            
            # Basic mesh analysis if available
            if 'mesh' in output_files and output_files['mesh'].endswith('.ply'):
                try:
                    mesh_stats = self._analyze_mesh_ply(output_files['mesh'])
                    metrics.update(mesh_stats)
                except Exception as e:
                    print(f"Warning: Could not analyze mesh: {e}")
            
        except Exception as e:
            print(f"Warning: Error computing quality metrics: {e}")
        
        return metrics
    
    def _analyze_mesh_ply(self, mesh_path: str) -> Dict[str, Any]:
        """Basic PLY mesh analysis"""
        stats = {}
        
        try:
            with open(mesh_path, 'r') as f:
                header_lines = []
                for line in f:
                    header_lines.append(line.strip())
                    if line.strip() == 'end_header':
                        break
                
                # Parse header for vertex/face counts
                for line in header_lines:
                    if line.startswith('element vertex'):
                        stats['vertex_count'] = int(line.split()[-1])
                    elif line.startswith('element face'):
                        stats['face_count'] = int(line.split()[-1])
                        
        except Exception as e:
            print(f"Warning: PLY analysis failed: {e}")
        
        return stats
    
    def export_for_dental_workflow(self, output_files: Dict[str, str], 
                                 export_dir: str) -> Dict[str, str]:
        """
        Export Meshroom results in dental workflow formats
        
        Args:
            output_files: Meshroom output files
            export_dir: Directory for exported files
            
        Returns:
            Dictionary of exported file paths
        """
        os.makedirs(export_dir, exist_ok=True)
        exported_files = {}
        
        try:
            # Copy mesh files with standard dental names
            if 'mesh' in output_files:
                mesh_ext = os.path.splitext(output_files['mesh'])[1]
                dental_mesh_path = os.path.join(export_dir, f'dental_scan{mesh_ext}')
                shutil.copy2(output_files['mesh'], dental_mesh_path)
                exported_files['dental_mesh'] = dental_mesh_path
            
            # Copy texture with standard name
            if 'texture' in output_files:
                texture_ext = os.path.splitext(output_files['texture'])[1]
                dental_texture_path = os.path.join(export_dir, f'dental_texture{texture_ext}')
                shutil.copy2(output_files['texture'], dental_texture_path)
                exported_files['dental_texture'] = dental_texture_path
            
            # Export additional formats if needed
            if 'mesh' in output_files and output_files['mesh'].endswith('.obj'):
                # Convert to STL for CAD compatibility
                stl_path = os.path.join(export_dir, 'dental_scan.stl')
                if self._convert_obj_to_stl(output_files['mesh'], stl_path):
                    exported_files['dental_stl'] = stl_path
            
            print(f"✅ Exported {len(exported_files)} files for dental workflow")
            
        except Exception as e:
            print(f"❌ Export failed: {e}")
        
        return exported_files
    
    def _convert_obj_to_stl(self, obj_path: str, stl_path: str) -> bool:
        """Convert OBJ to STL format (basic implementation)"""
        try:
            # This is a simplified conversion - in practice you might want to use
            # a proper library like Open3D or trimesh
            import subprocess
            
            # Try using meshlab if available
            result = subprocess.run([
                'meshlabserver', '-i', obj_path, '-o', stl_path
            ], capture_output=True)
            
            return result.returncode == 0 and os.path.exists(stl_path)
            
        except:
            print("Warning: OBJ to STL conversion failed (meshlab not available)")
            return False
    
    def cleanup_project(self):
        """Clean up temporary project files"""
        if self.temp_dir and os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir)
                print(f"✅ Cleaned up project directory: {self.temp_dir}")
            except Exception as e:
                print(f"Warning: Cleanup failed: {e}")
        
        self.current_project = None
        self.temp_dir = None
    
    def get_status(self) -> Dict[str, Any]:
        """Get current Meshroom integration status"""
        return {
            'meshroom_available': self.is_available(),
            'meshroom_path': self.meshroom_path,
            'active_project': self.current_project['name'] if self.current_project else None,
            'pipeline_configs': list(self.pipeline_configs.keys()),
            'temp_directory': self.temp_dir
        }

# Integration with existing SLAM system
class SLAMMeshroomBridge:
    """
    Bridge between our real-time SLAM and Meshroom reconstruction
    """
    
    def __init__(self, slam_processor, meshroom_integration):
        self.slam = slam_processor
        self.meshroom = meshroom_integration
        self.frame_buffer = []
        self.max_frames = 100  # Maximum frames to buffer
        
    def start_slam_capture(self, project_name: str):
        """Start capturing SLAM frames for Meshroom reconstruction"""
        if not self.meshroom.create_project(project_name, 'dental_scan'):
            return False
        
        self.frame_buffer = []
        print(f"✅ Started SLAM capture for Meshroom project: {project_name}")
        return True
    
    def capture_slam_frame(self, color_image, depth_image, pose):
        """Capture a SLAM frame for later Meshroom processing"""
        if len(self.frame_buffer) < self.max_frames:
            frame_data = {
                'color_image': color_image,
                'depth_image': depth_image,
                'pose': pose,
                'timestamp': time.time()
            }
            self.frame_buffer.append(frame_data)
            return True
        return False
    
    def process_captured_frames(self, quality: str = 'dental_scan'):
        """Process captured SLAM frames through Meshroom"""
        if not self.frame_buffer:
            print("❌ No frames captured")
            return None
        
        # Add frames to Meshroom
        if not self.meshroom.add_images_from_slam(self.frame_buffer):
            return None
        
        # Run reconstruction
        return self.meshroom.run_reconstruction(quality)

# Test function
def test_meshroom_integration():
    """Test Meshroom integration functionality"""
    print("🧪 Testing Meshroom Integration...")
    
    # Initialize Meshroom
    meshroom = MeshroomIntegration()
    
    if not meshroom.is_available():
        print("❌ Meshroom not available for testing")
        return False
    
    # Test project creation
    if not meshroom.create_project("test_dental_scan", "dental_scan"):
        return False
    
    # Display status
    status = meshroom.get_status()
    print("📊 Meshroom Status:")
    for key, value in status.items():
        print(f"   {key}: {value}")
    
    # Cleanup
    meshroom.cleanup_project()
    
    print("✅ Meshroom integration test completed")
    return True

if __name__ == "__main__":
    test_meshroom_integration()
