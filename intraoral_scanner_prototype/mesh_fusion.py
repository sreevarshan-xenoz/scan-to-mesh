"""
Mesh Fusion - TSDF-based volumetric fusion for real-time mesh reconstruction
Implements the core mesh fusion approach used in professional scanning systems
"""

import numpy as np
import open3d as o3d
from typing import List, Tuple, Optional

class MeshFusion:
    def __init__(self, voxel_size: float = 0.005, sdf_trunc: float = 0.02):
        """
        Initialize TSDF volume for mesh fusion
        
        Args:
            voxel_size: Size of each voxel in meters (5mm default)
            sdf_trunc: Truncation distance for SDF in meters (2cm default)
        """
        self.voxel_size = voxel_size
        self.sdf_trunc = sdf_trunc
        
        # Initialize TSDF volume
        self.volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=voxel_size,
            sdf_trunc=sdf_trunc,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
        )
        
        # Track camera poses for registration
        self.camera_poses = []
        self.frame_count = 0
        
        # Current mesh
        self.current_mesh = None
        
    def integrate_point_cloud(self, 
                            points: np.ndarray, 
                            colors: np.ndarray, 
                            camera_pose: np.ndarray = None) -> bool:
        """
        Integrate new point cloud into TSDF volume
        
        Args:
            points: Nx3 array of 3D points
            colors: Nx3 array of RGB colors [0,1]
            camera_pose: 4x4 transformation matrix (identity if None)
        """
        if len(points) == 0:
            return False
            
        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        # Set default camera pose if not provided
        if camera_pose is None:
            camera_pose = np.eye(4)
        
        # Create RGBD image from point cloud (simplified approach)
        # In real system, this would use actual RGBD data
        rgbd = self._point_cloud_to_rgbd(pcd, camera_pose)
        
        if rgbd is not None:
            # Integrate into TSDF volume
            intrinsic = o3d.camera.PinholeCameraIntrinsic(
                width=640, height=480, fx=500, fy=500, cx=320, cy=240)
            
            extrinsic = np.linalg.inv(camera_pose)  # Open3D uses inverse convention
            
            self.volume.integrate(rgbd, intrinsic, extrinsic)
            self.camera_poses.append(camera_pose)
            self.frame_count += 1
            
            return True
        
        return False
    
    def _point_cloud_to_rgbd(self, pcd: o3d.geometry.PointCloud, camera_pose: np.ndarray) -> Optional[o3d.geometry.RGBDImage]:
        """Convert point cloud to RGBD image for TSDF integration"""
        try:
            # Project point cloud to image plane
            width, height = 640, 480
            intrinsic = o3d.camera.PinholeCameraIntrinsic(
                width=width, height=height, fx=500, fy=500, cx=320, cy=240)
            
            # Transform points to camera coordinate system
            points = np.asarray(pcd.points)
            colors = np.asarray(pcd.colors)
            
            # Apply camera pose transformation
            points_homogeneous = np.column_stack([points, np.ones(len(points))])
            camera_points = (np.linalg.inv(camera_pose) @ points_homogeneous.T).T[:, :3]
            
            # Project to image plane
            fx, fy = intrinsic.get_focal_length()
            cx, cy = intrinsic.get_principal_point()
            
            # Only keep points in front of camera
            valid_depth = camera_points[:, 2] > 0
            if not np.any(valid_depth):
                return None
                
            camera_points = camera_points[valid_depth]
            colors = colors[valid_depth]
            
            # Project to pixel coordinates
            u = (camera_points[:, 0] * fx / camera_points[:, 2] + cx).astype(int)
            v = (camera_points[:, 1] * fy / camera_points[:, 2] + cy).astype(int)
            
            # Keep points within image bounds
            valid_pixels = (u >= 0) & (u < width) & (v >= 0) & (v < height)
            if not np.any(valid_pixels):
                return None
                
            u = u[valid_pixels]
            v = v[valid_pixels]
            depths = camera_points[valid_pixels, 2]
            pixel_colors = colors[valid_pixels]
            
            # Create depth and color images
            depth_image = np.zeros((height, width), dtype=np.float32)
            color_image = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Fill images (handle multiple points per pixel by taking closest)
            for i in range(len(u)):
                if depth_image[v[i], u[i]] == 0 or depths[i] < depth_image[v[i], u[i]]:
                    depth_image[v[i], u[i]] = depths[i]
                    color_image[v[i], u[i]] = (pixel_colors[i] * 255).astype(np.uint8)
            
            # Convert to Open3D format
            o3d_color = o3d.geometry.Image(color_image)
            o3d_depth = o3d.geometry.Image(depth_image)
            
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d_color, o3d_depth, depth_scale=1.0, depth_trunc=1.0)
            
            return rgbd
            
        except Exception as e:
            print(f"Error converting point cloud to RGBD: {e}")
            return None
    
    def extract_mesh(self) -> Optional[o3d.geometry.TriangleMesh]:
        """Extract triangle mesh from TSDF volume"""
        try:
            mesh = self.volume.extract_triangle_mesh()
            
            if len(mesh.vertices) == 0:
                return None
                
            # Clean up mesh
            mesh.remove_degenerate_triangles()
            mesh.remove_duplicated_triangles()
            mesh.remove_duplicated_vertices()
            mesh.remove_non_manifold_edges()
            
            # Smooth mesh
            mesh = mesh.filter_smooth_simple(number_of_iterations=1)
            
            # Compute normals
            mesh.compute_vertex_normals()
            mesh.compute_triangle_normals()
            
            self.current_mesh = mesh
            return mesh
            
        except Exception as e:
            print(f"Error extracting mesh: {e}")
            return None
    
    def get_current_mesh(self) -> Optional[o3d.geometry.TriangleMesh]:
        """Get current reconstructed mesh"""
        return self.current_mesh
    
    def reset_volume(self):
        """Reset TSDF volume for new scan"""
        self.volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=self.voxel_size,
            sdf_trunc=self.sdf_trunc,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
        )
        self.camera_poses = []
        self.frame_count = 0
        self.current_mesh = None
    
    def save_mesh(self, filename: str) -> bool:
        """Save current mesh to file"""
        if self.current_mesh is None:
            return False
            
        try:
            o3d.io.write_triangle_mesh(filename, self.current_mesh)
            return True
        except Exception as e:
            print(f"Error saving mesh: {e}")
            return False
    
    def get_volume_info(self) -> dict:
        """Get information about current TSDF volume"""
        return {
            'frame_count': self.frame_count,
            'voxel_size': self.voxel_size,
            'sdf_trunc': self.sdf_trunc,
            'has_mesh': self.current_mesh is not None,
            'mesh_vertices': len(self.current_mesh.vertices) if self.current_mesh else 0,
            'mesh_triangles': len(self.current_mesh.triangles) if self.current_mesh else 0
        }