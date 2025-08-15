"""
Main Application - Intraoral Scanner Prototype
Orchestrates all components for real-time 3D dental scanning
"""

import cv2
import numpy as np
import time
import threading
from typing import Optional

from camera_manager import CameraManager
from stereo_processor import StereoProcessor
from mesh_fusion import MeshFusion
from dental_segmentation import DentalSegmentation
from visualizer import ScanVisualizer

class IntraoralScannerApp:
    def __init__(self):
        # Initialize components
        self.camera_manager = CameraManager()
        self.stereo_processor = StereoProcessor()
        self.mesh_fusion = MeshFusion()
        self.dental_segmentation = DentalSegmentation()
        self.visualizer = ScanVisualizer("Intraoral Scanner Prototype")
        
        # Application state
        self.is_scanning = False
        self.is_running = False
        self.scan_thread = None
        
        # Performance metrics
        self.frame_count = 0
        self.start_time = None
        self.last_fps_update = 0
        self.current_fps = 0
        
        # Scanning parameters
        self.frames_per_integration = 5  # Integrate every N frames
        self.frame_counter = 0
        
    def initialize(self) -> bool:
        """Initialize all components"""
        print("Initializing Intraoral Scanner Prototype...")
        
        # Initialize camera
        if not self.camera_manager.start_capture():
            print("ERROR: Failed to initialize camera")
            return False
        print("✓ Camera initialized")
        
        # Initialize visualizer
        if not self.visualizer.initialize():
            print("ERROR: Failed to initialize visualizer")
            return False
        print("✓ Visualizer initialized")
        
        # Print controls
        self.visualizer.print_controls()
        
        print("✓ All components initialized successfully")
        return True
    
    def start_scanning(self):
        """Start the scanning process"""
        if self.is_scanning:
            return
            
        print("Starting scan...")
        self.is_scanning = True
        self.is_running = True
        self.start_time = time.time()
        self.frame_count = 0
        
        # Reset mesh fusion
        self.mesh_fusion.reset_volume()
        
        # Start visualization
        self.visualizer.start_visualization()
        
        # Start scanning thread
        self.scan_thread = threading.Thread(target=self._scanning_loop)
        self.scan_thread.start()
        
        print("Scan started. Press 'q' in camera window to stop.")
    
    def _scanning_loop(self):
        """Main scanning loop"""
        while self.is_running:
            try:
                # Get structured light pair from camera
                frame_with_pattern, frame_without_pattern = self.camera_manager.get_structured_light_pair()
                
                if frame_with_pattern is None or frame_without_pattern is None:
                    time.sleep(0.01)
                    continue
                
                # Undistort frames
                frame_with_pattern = self.camera_manager.undistort_frame(frame_with_pattern)
                frame_without_pattern = self.camera_manager.undistort_frame(frame_without_pattern)
                
                # Compute depth using structured light
                depth_map = self.stereo_processor.compute_depth_from_structured_light(
                    frame_with_pattern, frame_without_pattern)
                
                # Generate point cloud
                points, colors = self.stereo_processor.generate_point_cloud(
                    frame_without_pattern, depth_map, self.camera_manager.camera_matrix)
                
                if len(points) > 0:
                    # Filter point cloud
                    points, colors = self.stereo_processor.filter_point_cloud(points, colors)
                    
                    # Update point cloud visualization
                    self.visualizer.update_point_cloud(points, colors)
                    
                    # Integrate into mesh every N frames
                    if self.frame_counter % self.frames_per_integration == 0:
                        # Simple camera pose (identity for now - in real system this would come from SLAM)
                        camera_pose = np.eye(4)
                        
                        # Integrate point cloud into TSDF volume
                        if self.mesh_fusion.integrate_point_cloud(points, colors, camera_pose):
                            # Extract and update mesh
                            mesh = self.mesh_fusion.extract_mesh()
                            if mesh is not None:
                                self.visualizer.update_mesh(mesh)
                
                # Perform dental segmentation on 2D image
                segments = self.dental_segmentation.segment_oral_cavity(frame_without_pattern)
                teeth = self.dental_segmentation.detect_individual_teeth(
                    segments['teeth'], frame_without_pattern)
                
                # Create segmentation overlay
                overlay = self.dental_segmentation.create_segmentation_overlay(
                    frame_without_pattern, segments)
                
                # Draw detected teeth
                for tooth in teeth:
                    cv2.drawContours(overlay, [tooth['contour']], -1, (0, 255, 0), 2)
                    cv2.circle(overlay, tuple(map(int, tooth['centroid'])), 5, (0, 255, 0), -1)
                
                # Display camera feed with overlay
                self._display_camera_feed(overlay, depth_map)
                
                # Update performance metrics
                self._update_performance_metrics()
                
                self.frame_counter += 1
                
                # Check for exit condition
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    self._save_current_scan()
                elif key == ord('r'):
                    self._reset_scan()
                
            except Exception as e:
                print(f"Error in scanning loop: {e}")
                time.sleep(0.1)
        
        self.stop_scanning()
    
    def _display_camera_feed(self, color_image: np.ndarray, depth_image: np.ndarray):
        """Display camera feed with overlays"""
        # Resize for display
        display_height = 480
        aspect_ratio = color_image.shape[1] / color_image.shape[0]
        display_width = int(display_height * aspect_ratio)
        
        color_display = cv2.resize(color_image, (display_width, display_height))
        
        # Create depth visualization
        depth_normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)
        depth_colored = cv2.applyColorMap(depth_normalized.astype(np.uint8), cv2.COLORMAP_JET)
        depth_display = cv2.resize(depth_colored, (display_width, display_height))
        
        # Combine side by side
        combined = np.hstack([color_display, depth_display])
        
        # Add text overlay
        fps_text = f"FPS: {self.current_fps:.1f}"
        frame_text = f"Frames: {self.frame_count}"
        mesh_info = self.mesh_fusion.get_volume_info()
        mesh_text = f"Vertices: {mesh_info['mesh_vertices']}"
        
        cv2.putText(combined, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(combined, frame_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(combined, mesh_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Add instructions
        instructions = [
            "Controls: 'q' - quit, 's' - save, 'r' - reset",
            "Left: Color + Segmentation, Right: Depth"
        ]
        
        for i, instruction in enumerate(instructions):
            cv2.putText(combined, instruction, (10, combined.shape[0] - 40 + i * 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow("Intraoral Scanner - Camera Feed", combined)
    
    def _update_performance_metrics(self):
        """Update FPS and other performance metrics"""
        self.frame_count += 1
        current_time = time.time()
        
        if current_time - self.last_fps_update >= 1.0:  # Update every second
            if self.start_time:
                elapsed_time = current_time - self.start_time
                self.current_fps = self.frame_count / elapsed_time
            
            self.last_fps_update = current_time
    
    def _save_current_scan(self):
        """Save current scan to file"""
        timestamp = int(time.time())
        filename = f"scan_{timestamp}.ply"
        
        if self.mesh_fusion.save_mesh(filename):
            print(f"Scan saved: {filename}")
        else:
            print("No mesh to save")
    
    def _reset_scan(self):
        """Reset current scan"""
        print("Resetting scan...")
        self.mesh_fusion.reset_volume()
        self.frame_count = 0
        self.start_time = time.time()
        print("Scan reset")
    
    def stop_scanning(self):
        """Stop the scanning process"""
        if not self.is_scanning:
            return
            
        print("Stopping scan...")
        self.is_running = False
        self.is_scanning = False
        
        if self.scan_thread:
            self.scan_thread.join()
        
        # Stop components
        self.camera_manager.stop_capture()
        self.visualizer.stop_visualization()
        
        # Close OpenCV windows
        cv2.destroyAllWindows()
        
        print("Scan stopped")
    
    def run(self):
        """Main application entry point"""
        try:
            if not self.initialize():
                return False
            
            self.start_scanning()
            
            # Keep main thread alive
            while self.is_running:
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        except Exception as e:
            print(f"Application error: {e}")
        finally:
            self.stop_scanning()
        
        return True

def main():
    """Application entry point"""
    print("=== Intraoral Scanner Prototype ===")
    print("Based on reverse engineering analysis of IntraoralScan 3.5.4.6")
    print("Open source implementation using OpenCV, Open3D, and Python")
    print("=====================================\n")
    
    app = IntraoralScannerApp()
    success = app.run()
    
    if success:
        print("Application completed successfully")
    else:
        print("Application failed to start")
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())