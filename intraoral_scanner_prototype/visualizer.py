"""
Visualizer - Real-time 3D visualization using Open3D
Provides interactive 3D mesh display similar to professional scanning software
"""

import open3d as o3d
import numpy as np
import threading
import time
from typing import Optional, List, Dict, Callable

class ScanVisualizer:
    def __init__(self, window_name: str = "Intraoral Scanner Prototype"):
        self.window_name = window_name
        self.vis = None
        self.mesh = None
        self.point_cloud = None
        self.is_running = False
        self.update_thread = None
        
        # Visualization settings
        self.show_mesh = True
        self.show_point_cloud = False
        self.show_wireframe = False
        self.background_color = [0.1, 0.1, 0.1]  # Dark gray
        
        # Callback functions
        self.on_key_callback = None
        
    def initialize(self) -> bool:
        """Initialize the 3D visualizer"""
        try:
            self.vis = o3d.visualization.Visualizer()
            self.vis.create_window(self.window_name, width=1024, height=768)
            
            # Set up rendering options
            render_option = self.vis.get_render_option()
            render_option.background_color = np.array(self.background_color)
            render_option.mesh_show_back_face = True
            render_option.mesh_show_wireframe = self.show_wireframe
            render_option.point_size = 2.0
            
            # Set up camera
            view_control = self.vis.get_view_control()
            view_control.set_front([0, 0, -1])
            view_control.set_up([0, -1, 0])
            view_control.set_zoom(0.8)
            
            # Register key callbacks
            self.vis.register_key_callback(ord('M'), self._toggle_mesh)
            self.vis.register_key_callback(ord('P'), self._toggle_point_cloud)
            self.vis.register_key_callback(ord('W'), self._toggle_wireframe)
            self.vis.register_key_callback(ord('R'), self._reset_view)
            self.vis.register_key_callback(ord('S'), self._save_screenshot)
            
            return True
            
        except Exception as e:
            print(f"Error initializing visualizer: {e}")
            return False
    
    def start_visualization(self):
        """Start the visualization loop in a separate thread"""
        if not self.vis:
            if not self.initialize():
                return False
        
        self.is_running = True
        self.update_thread = threading.Thread(target=self._visualization_loop)
        self.update_thread.start()
        return True
    
    def _visualization_loop(self):
        """Main visualization loop"""
        while self.is_running:
            try:
                # Update visualization
                self.vis.poll_events()
                self.vis.update_renderer()
                
                # Small delay to prevent excessive CPU usage
                time.sleep(0.016)  # ~60 FPS
                
            except Exception as e:
                print(f"Visualization error: {e}")
                break
    
    def update_mesh(self, mesh: o3d.geometry.TriangleMesh):
        """Update the displayed mesh"""
        if not self.vis or not self.is_running:
            return
            
        try:
            # Remove old mesh if exists
            if self.mesh is not None:
                self.vis.remove_geometry(self.mesh, reset_bounding_box=False)
            
            # Add new mesh
            if mesh is not None and len(mesh.vertices) > 0:
                self.mesh = mesh
                
                # Ensure mesh has colors
                if not mesh.has_vertex_colors():
                    # Default to light gray
                    mesh.paint_uniform_color([0.8, 0.8, 0.8])
                
                if self.show_mesh:
                    self.vis.add_geometry(self.mesh, reset_bounding_box=False)
                    
        except Exception as e:
            print(f"Error updating mesh: {e}")
    
    def update_point_cloud(self, points: np.ndarray, colors: np.ndarray = None):
        """Update the displayed point cloud"""
        if not self.vis or not self.is_running:
            return
            
        try:
            # Remove old point cloud if exists
            if self.point_cloud is not None:
                self.vis.remove_geometry(self.point_cloud, reset_bounding_box=False)
            
            # Create new point cloud
            if len(points) > 0:
                self.point_cloud = o3d.geometry.PointCloud()
                self.point_cloud.points = o3d.utility.Vector3dVector(points)
                
                if colors is not None:
                    self.point_cloud.colors = o3d.utility.Vector3dVector(colors)
                else:
                    # Default to white
                    self.point_cloud.paint_uniform_color([1.0, 1.0, 1.0])
                
                if self.show_point_cloud:
                    self.vis.add_geometry(self.point_cloud, reset_bounding_box=False)
                    
        except Exception as e:
            print(f"Error updating point cloud: {e}")
    
    def add_coordinate_frame(self, size: float = 0.05, origin: np.ndarray = None):
        """Add coordinate frame for reference"""
        if not self.vis:
            return
            
        try:
            if origin is None:
                origin = [0, 0, 0]
                
            coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=size, origin=origin)
            self.vis.add_geometry(coord_frame)
            
        except Exception as e:
            print(f"Error adding coordinate frame: {e}")
    
    def _toggle_mesh(self, vis):
        """Toggle mesh visibility"""
        self.show_mesh = not self.show_mesh
        
        if self.mesh is not None:
            if self.show_mesh:
                vis.add_geometry(self.mesh, reset_bounding_box=False)
            else:
                vis.remove_geometry(self.mesh, reset_bounding_box=False)
        
        print(f"Mesh visibility: {'ON' if self.show_mesh else 'OFF'}")
        return False
    
    def _toggle_point_cloud(self, vis):
        """Toggle point cloud visibility"""
        self.show_point_cloud = not self.show_point_cloud
        
        if self.point_cloud is not None:
            if self.show_point_cloud:
                vis.add_geometry(self.point_cloud, reset_bounding_box=False)
            else:
                vis.remove_geometry(self.point_cloud, reset_bounding_box=False)
        
        print(f"Point cloud visibility: {'ON' if self.show_point_cloud else 'OFF'}")
        return False
    
    def _toggle_wireframe(self, vis):
        """Toggle wireframe mode"""
        self.show_wireframe = not self.show_wireframe
        render_option = vis.get_render_option()
        render_option.mesh_show_wireframe = self.show_wireframe
        
        print(f"Wireframe mode: {'ON' if self.show_wireframe else 'OFF'}")
        return False
    
    def _reset_view(self, vis):
        """Reset camera view"""
        view_control = vis.get_view_control()
        view_control.set_front([0, 0, -1])
        view_control.set_up([0, -1, 0])
        view_control.set_zoom(0.8)
        
        print("View reset")
        return False
    
    def _save_screenshot(self, vis):
        """Save screenshot"""
        timestamp = int(time.time())
        filename = f"scan_screenshot_{timestamp}.png"
        vis.capture_screen_image(filename)
        print(f"Screenshot saved: {filename}")
        return False
    
    def set_camera_pose(self, pose: np.ndarray):
        """Set camera pose for visualization"""
        if not self.vis:
            return
            
        try:
            view_control = self.vis.get_view_control()
            camera_params = view_control.convert_to_pinhole_camera_parameters()
            
            # Update extrinsic matrix
            camera_params.extrinsic = pose
            view_control.convert_from_pinhole_camera_parameters(camera_params)
            
        except Exception as e:
            print(f"Error setting camera pose: {e}")
    
    def get_screenshot(self) -> Optional[np.ndarray]:
        """Get current screenshot as numpy array"""
        if not self.vis:
            return None
            
        try:
            # Capture screen to temporary file
            temp_filename = "temp_screenshot.png"
            self.vis.capture_screen_image(temp_filename)
            
            # Read back as numpy array
            import cv2
            image = cv2.imread(temp_filename)
            
            # Clean up temp file
            import os
            if os.path.exists(temp_filename):
                os.remove(temp_filename)
            
            return image
            
        except Exception as e:
            print(f"Error capturing screenshot: {e}")
            return None
    
    def stop_visualization(self):
        """Stop the visualization"""
        self.is_running = False
        
        if self.update_thread:
            self.update_thread.join()
        
        if self.vis:
            self.vis.destroy_window()
            self.vis = None
    
    def print_controls(self):
        """Print keyboard controls"""
        print("\n=== Visualization Controls ===")
        print("M - Toggle mesh visibility")
        print("P - Toggle point cloud visibility") 
        print("W - Toggle wireframe mode")
        print("R - Reset camera view")
        print("S - Save screenshot")
        print("Mouse - Rotate/zoom/pan view")
        print("ESC - Exit")
        print("===============================\n")

class MultiViewVisualizer:
    """Multi-viewport visualizer for different views"""
    
    def __init__(self):
        self.viewers = {}
        self.active_viewer = None
    
    def add_viewer(self, name: str, window_name: str = None) -> ScanVisualizer:
        """Add a new viewer"""
        if window_name is None:
            window_name = f"Scanner - {name}"
            
        viewer = ScanVisualizer(window_name)
        self.viewers[name] = viewer
        
        if self.active_viewer is None:
            self.active_viewer = name
            
        return viewer
    
    def get_viewer(self, name: str) -> Optional[ScanVisualizer]:
        """Get viewer by name"""
        return self.viewers.get(name)
    
    def start_all(self):
        """Start all viewers"""
        for viewer in self.viewers.values():
            viewer.start_visualization()
    
    def stop_all(self):
        """Stop all viewers"""
        for viewer in self.viewers.values():
            viewer.stop_visualization()