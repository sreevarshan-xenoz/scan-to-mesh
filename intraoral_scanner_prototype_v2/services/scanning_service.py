"""
Scanning Service - Professional scanning engine based on DentalScanAppLogic.exe analysis
Implements the complete acquisition → registration → fusion pipeline
"""

import time
import threading
import queue
import numpy as np
from typing import Optional, Tuple, Dict, Any
import zmq
import json

from config.system_config import get_config
from hardware.camera_manager_v2 import CameraManagerV2
from processing.slam_processor import SLAMProcessor
from processing.tsdf_fusion_v2 import TSDFFusionV2
from utils.performance_monitor import PerformanceMonitor
from utils.shared_memory import SharedMemoryManager

class ScanningService:
    """
    Professional scanning service implementing the complete pipeline:
    Camera → SLAM → Registration → TSDF Fusion → Mesh
    """
    
    def __init__(self, service_port: int = 5555):
        self.config = get_config()
        self.service_port = service_port
        
        # Initialize components
        self.camera_manager = CameraManagerV2()
        self.slam_processor = SLAMProcessor()
        self.tsdf_fusion = TSDFFusionV2()
        self.performance_monitor = PerformanceMonitor()
        self.shared_memory = SharedMemoryManager()
        
        # Service state
        self.is_running = False
        self.is_scanning = False
        self.scan_session_id = None
        
        # Threading
        self.processing_thread = None
        self.communication_thread = None
        
        # Data queues
        self.frame_queue = queue.Queue(maxsize=10)
        self.result_queue = queue.Queue(maxsize=5)
        
        # ZeroMQ communication
        self.context = zmq.Context()
        self.socket = None
        
        # Performance metrics
        self.frame_count = 0
        self.processing_times = []
        
    def start_service(self) -> bool:
        """Start the scanning service"""
        try:
            # Initialize hardware
            if not self.camera_manager.initialize():
                print("ERROR: Failed to initialize camera")
                return False
            
            # Initialize SLAM processor
            if not self.slam_processor.initialize():
                print("ERROR: Failed to initialize SLAM processor")
                return False
            
            # Initialize TSDF fusion
            if not self.tsdf_fusion.initialize():
                print("ERROR: Failed to initialize TSDF fusion")
                return False
            
            # Setup communication
            self.socket = self.context.socket(zmq.REP)
            self.socket.bind(f"tcp://*:{self.service_port}")
            
            # Start service threads
            self.is_running = True
            self.processing_thread = threading.Thread(target=self._processing_loop)
            self.communication_thread = threading.Thread(target=self._communication_loop)
            
            self.processing_thread.start()
            self.communication_thread.start()
            
            print(f"Scanning service started on port {self.service_port}")
            return True
            
        except Exception as e:
            print(f"Error starting scanning service: {e}")
            return False
    
    def stop_service(self):
        """Stop the scanning service"""
        self.is_running = False
        self.is_scanning = False
        
        # Stop threads
        if self.processing_thread:
            self.processing_thread.join()
        if self.communication_thread:
            self.communication_thread.join()
        
        # Cleanup components
        self.camera_manager.cleanup()
        self.slam_processor.cleanup()
        self.tsdf_fusion.cleanup()
        
        # Cleanup communication
        if self.socket:
            self.socket.close()
        self.context.term()
        
        print("Scanning service stopped")
    
    def _processing_loop(self):
        """Main processing loop - implements the professional pipeline"""
        while self.is_running:
            try:
                if not self.is_scanning:
                    time.sleep(0.01)
                    continue
                
                start_time = time.time()
                
                # Stage 1: Acquisition (matching DentalScanAppLogic.exe)
                frame_data = self.camera_manager.get_frame_data()
                if frame_data is None:
                    continue
                
                # Stage 2: SLAM Processing (matching Sn3DScanSlam.dll)
                slam_result = self.slam_processor.process_frame(
                    frame_data['color_image'],
                    frame_data['depth_image'],
                    frame_data['timestamp']
                )
                
                if slam_result is None:
                    continue
                
                # Stage 3: Point Cloud Generation
                points, colors = self._generate_point_cloud(
                    frame_data['color_image'],
                    frame_data['depth_image'],
                    slam_result['camera_pose']
                )
                
                if len(points) == 0:
                    continue
                
                # Stage 4: Registration and Fusion (matching Sn3DSpeckleFusion.dll)
                integration_result = self.tsdf_fusion.integrate_frame(
                    points, colors, slam_result['camera_pose']
                )
                
                # Stage 5: Mesh Extraction (periodic)
                mesh = None
                if self.frame_count % self.config.processing.mesh_extraction_frequency == 0:
                    mesh = self.tsdf_fusion.extract_mesh()
                
                # Update performance metrics
                processing_time = time.time() - start_time
                self.processing_times.append(processing_time)
                self.frame_count += 1
                
                # Store results in shared memory
                result_data = {
                    'frame_id': self.frame_count,
                    'timestamp': frame_data['timestamp'],
                    'camera_pose': slam_result['camera_pose'].tolist(),
                    'points_count': len(points),
                    'integration_success': integration_result,
                    'mesh_vertices': len(mesh.vertices) if mesh else 0,
                    'processing_time': processing_time,
                    'slam_confidence': slam_result.get('confidence', 0.0)
                }
                
                # Update shared memory
                self.shared_memory.update_scan_data(result_data)
                
                # Send to result queue for communication
                if not self.result_queue.full():
                    self.result_queue.put(result_data)
                
                # Performance monitoring
                self.performance_monitor.update_metrics({
                    'fps': 1.0 / processing_time if processing_time > 0 else 0,
                    'processing_time': processing_time,
                    'points_per_frame': len(points),
                    'memory_usage': self.shared_memory.get_memory_usage()
                })
                
            except Exception as e:
                print(f"Error in processing loop: {e}")
                time.sleep(0.01)
    
    def _communication_loop(self):
        """Handle communication with other services"""
        while self.is_running:
            try:
                # Check for incoming messages (non-blocking)
                if self.socket.poll(timeout=10):  # 10ms timeout
                    message = self.socket.recv_json()
                    response = self._handle_message(message)
                    self.socket.send_json(response)
                
            except Exception as e:
                print(f"Error in communication loop: {e}")
                time.sleep(0.01)
    
    def _handle_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Handle incoming service messages"""
        command = message.get('command', '')
        
        if command == 'start_scan':
            return self._start_scan(message.get('params', {}))
        elif command == 'stop_scan':
            return self._stop_scan()
        elif command == 'get_status':
            return self._get_status()
        elif command == 'get_performance':
            return self._get_performance_metrics()
        elif command == 'configure':
            return self._configure_service(message.get('params', {}))
        else:
            return {'status': 'error', 'message': f'Unknown command: {command}'}
    
    def _start_scan(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Start a new scanning session"""
        try:
            if self.is_scanning:
                return {'status': 'error', 'message': 'Scan already in progress'}
            
            # Generate new scan session ID
            self.scan_session_id = f"scan_{int(time.time())}"
            
            # Reset components
            self.slam_processor.reset()
            self.tsdf_fusion.reset()
            self.performance_monitor.reset()
            
            # Configure scan parameters
            scan_type = params.get('scan_type', 'full_arch')
            quality = params.get('quality', 'high')
            
            # Update processing parameters based on scan type
            if scan_type == 'quadrant':
                self.config.processing.volume_size = [0.1, 0.1, 0.08]  # Smaller volume
            elif scan_type == 'full_arch':
                self.config.processing.volume_size = [0.2, 0.2, 0.15]  # Full arch
            
            # Start camera capture
            if not self.camera_manager.start_capture():
                return {'status': 'error', 'message': 'Failed to start camera'}
            
            # Start scanning
            self.is_scanning = True
            self.frame_count = 0
            
            return {
                'status': 'success',
                'scan_session_id': self.scan_session_id,
                'message': 'Scan started successfully'
            }
            
        except Exception as e:
            return {'status': 'error', 'message': f'Failed to start scan: {e}'}
    
    def _stop_scan(self) -> Dict[str, Any]:
        """Stop current scanning session"""
        try:
            if not self.is_scanning:
                return {'status': 'error', 'message': 'No scan in progress'}
            
            # Stop scanning
            self.is_scanning = False
            
            # Stop camera capture
            self.camera_manager.stop_capture()
            
            # Get final mesh
            final_mesh = self.tsdf_fusion.extract_mesh()
            
            # Get scan statistics
            stats = {
                'total_frames': self.frame_count,
                'average_processing_time': np.mean(self.processing_times) if self.processing_times else 0,
                'mesh_vertices': len(final_mesh.vertices) if final_mesh else 0,
                'mesh_triangles': len(final_mesh.triangles) if final_mesh else 0,
                'scan_duration': time.time() - (self.processing_times[0] if self.processing_times else time.time())
            }
            
            return {
                'status': 'success',
                'scan_session_id': self.scan_session_id,
                'statistics': stats,
                'message': 'Scan completed successfully'
            }
            
        except Exception as e:
            return {'status': 'error', 'message': f'Failed to stop scan: {e}'}
    
    def _get_status(self) -> Dict[str, Any]:
        """Get current service status"""
        return {
            'status': 'success',
            'service_running': self.is_running,
            'scanning': self.is_scanning,
            'scan_session_id': self.scan_session_id,
            'frame_count': self.frame_count,
            'camera_connected': self.camera_manager.is_connected(),
            'slam_initialized': self.slam_processor.is_initialized(),
            'tsdf_initialized': self.tsdf_fusion.is_initialized()
        }
    
    def _get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        metrics = self.performance_monitor.get_metrics()
        
        return {
            'status': 'success',
            'metrics': metrics,
            'processing_times': self.processing_times[-100:],  # Last 100 frames
            'memory_usage': self.shared_memory.get_memory_usage()
        }
    
    def _configure_service(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Configure service parameters"""
        try:
            # Update camera configuration
            if 'camera' in params:
                camera_params = params['camera']
                if 'resolution' in camera_params:
                    self.config.camera.width = camera_params['resolution'][0]
                    self.config.camera.height = camera_params['resolution'][1]
                if 'fps' in camera_params:
                    self.config.camera.fps = camera_params['fps']
            
            # Update processing configuration
            if 'processing' in params:
                processing_params = params['processing']
                if 'voxel_size' in processing_params:
                    self.config.processing.voxel_size = processing_params['voxel_size']
                if 'integration_frequency' in processing_params:
                    self.config.processing.integration_frequency = processing_params['integration_frequency']
            
            # Save configuration
            self.config.save_config()
            
            return {'status': 'success', 'message': 'Configuration updated'}
            
        except Exception as e:
            return {'status': 'error', 'message': f'Failed to update configuration: {e}'}
    
    def _generate_point_cloud(self, color_image: np.ndarray, depth_image: np.ndarray, 
                            camera_pose: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Generate point cloud from color and depth images"""
        height, width = depth_image.shape
        
        # Camera intrinsics
        fx = self.config.camera.focal_length_x
        fy = self.config.camera.focal_length_y
        cx = self.config.camera.principal_point_x
        cy = self.config.camera.principal_point_y
        
        # Create coordinate grids
        u, v = np.meshgrid(np.arange(width), np.arange(height))
        
        # Valid depth mask
        valid_depth = (depth_image > 0) & (depth_image < 2.0)  # 0-2m range
        
        if not np.any(valid_depth):
            return np.array([]), np.array([])
        
        # Convert to 3D coordinates
        z = depth_image[valid_depth]
        x = (u[valid_depth] - cx) * z / fx
        y = (v[valid_depth] - cy) * z / fy
        
        # Stack coordinates
        points_camera = np.column_stack((x, y, z))
        
        # Transform to world coordinates using camera pose
        points_homogeneous = np.column_stack([points_camera, np.ones(len(points_camera))])
        points_world = (camera_pose @ points_homogeneous.T).T[:, :3]
        
        # Get corresponding colors
        if len(color_image.shape) == 3:
            colors = color_image[valid_depth] / 255.0
        else:
            gray_colors = color_image[valid_depth] / 255.0
            colors = np.column_stack((gray_colors, gray_colors, gray_colors))
        
        return points_world, colors

def main():
    """Run scanning service as standalone process"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Dental Scanning Service')
    parser.add_argument('--port', type=int, default=5555, help='Service port')
    parser.add_argument('--config', type=str, help='Configuration file path')
    
    args = parser.parse_args()
    
    # Initialize service
    service = ScanningService(service_port=args.port)
    
    try:
        if service.start_service():
            print("Scanning service running. Press Ctrl+C to stop.")
            while True:
                time.sleep(1)
        else:
            print("Failed to start scanning service")
    except KeyboardInterrupt:
        print("\nShutting down scanning service...")
    finally:
        service.stop_service()

if __name__ == "__main__":
    main()