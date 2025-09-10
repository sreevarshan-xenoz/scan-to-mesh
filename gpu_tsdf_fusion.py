"""
GPU-Accelerated TSDF Fusion for Real-Time Dental Scanning

Based on IntraoralScan's Sn3DSpeckleFusion.dll analysis
Implements professional-grade TSDF with CUDA acceleration
"""

import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, List
import time
import threading
from dataclasses import dataclass

@dataclass
class TSDFConfig:
    # Volume parameters (optimized for dental scanning)
    volume_size: Tuple[float, float, float] = (0.12, 0.12, 0.08)  # 12cm x 12cm x 8cm
    voxel_size: float = 0.002  # 2mm voxels for high quality
    
    # TSDF parameters
    truncation_distance: float = 0.01  # 1cm truncation
    max_weight: float = 100.0
    
    # Performance parameters
    use_gpu: bool = True
    device: str = "cuda:0"
    max_integration_distance: float = 0.05  # 5cm max integration range
    
    # Memory management
    max_volume_memory_mb: float = 1024.0  # 1GB max GPU memory for volume

class GPUTSDFFusion:
    """
    GPU-accelerated TSDF fusion for real-time dental scanning
    
    Based on analysis of IntraoralScan's volumetric fusion approach,
    implementing professional-grade TSDF with CUDA optimization.
    """
    
    def __init__(self, config: TSDFConfig):
        self.config = config
        self.device = torch.device(config.device if config.use_gpu and torch.cuda.is_available() else "cpu")
        
        # Calculate volume dimensions
        self.volume_dims = tuple(int(size / config.voxel_size) for size in config.volume_size)
        print(f"TSDF Volume dimensions: {self.volume_dims} ({np.prod(self.volume_dims):,} voxels)")
        
        # Initialize TSDF volume on GPU
        self.tsdf_volume = torch.zeros(self.volume_dims, dtype=torch.float32, device=self.device)
        self.weight_volume = torch.zeros(self.volume_dims, dtype=torch.float32, device=self.device)
        self.color_volume = torch.zeros((*self.volume_dims, 3), dtype=torch.float32, device=self.device)
        
        # Volume origin in world coordinates
        self.volume_origin = torch.tensor([
            -config.volume_size[0] / 2,
            -config.volume_size[1] / 2,
            -config.volume_size[2] / 2
        ], dtype=torch.float32, device=self.device)
        
        # Performance monitoring
        self.integration_count = 0
        self.total_integration_time = 0.0
        self.last_mesh_extraction_time = 0.0
        
        # Thread safety
        self.lock = threading.Lock()
        
        print(f"TSDF Fusion initialized on {self.device}")
        print(f"Volume size: {config.volume_size} meters")
        print(f"Voxel size: {config.voxel_size} meters")
        print(f"Memory usage: {self.estimate_memory_usage():.1f} MB")
    
    def estimate_memory_usage(self) -> float:
        """Estimate GPU memory usage in MB"""
        voxel_count = np.prod(self.volume_dims)
        
        # TSDF volume: float32
        tsdf_memory = voxel_count * 4
        # Weight volume: float32
        weight_memory = voxel_count * 4
        # Color volume: float32 x 3
        color_memory = voxel_count * 4 * 3
        
        total_bytes = tsdf_memory + weight_memory + color_memory
        return total_bytes / (1024 * 1024)  # Convert to MB
    
    def integrate_frame(self, 
                       depth_image: np.ndarray, 
                       color_image: np.ndarray,
                       camera_intrinsics: Dict[str, float],
                       camera_pose: np.ndarray,
                       depth_scale: float = 1000.0) -> bool:
        """
        Integrate a new depth/color frame into the TSDF volume
        
        Args:
            depth_image: Depth image (H, W) in millimeters
            color_image: Color image (H, W, 3) in [0, 255]
            camera_intrinsics: Camera intrinsic parameters
            camera_pose: 4x4 camera pose matrix (world to camera)
            depth_scale: Scale factor for depth values (default: mm to meters)
            
        Returns:
            bool: True if integration successful
        """
        start_time = time.time()
        
        try:
            with self.lock:
                # Convert inputs to tensors
                depth_tensor = torch.from_numpy(depth_image.astype(np.float32)).to(self.device) / depth_scale
                color_tensor = torch.from_numpy(color_image.astype(np.float32)).to(self.device) / 255.0
                
                # Camera parameters
                fx, fy = camera_intrinsics['fx'], camera_intrinsics['fy']
                cx, cy = camera_intrinsics['cx'], camera_intrinsics['cy']
                
                # Camera pose
                pose_tensor = torch.from_numpy(camera_pose.astype(np.float32)).to(self.device)
                
                # Generate voxel coordinates
                voxel_coords = self._generate_voxel_coordinates()
                
                # Transform voxel coordinates to camera space
                camera_coords = self._transform_to_camera_space(voxel_coords, pose_tensor)
                
                # Project to image coordinates
                image_coords, valid_mask = self._project_to_image(
                    camera_coords, fx, fy, cx, cy, depth_tensor.shape
                )
                
                # Sample depth and color values
                sampled_depth, sampled_color, depth_valid = self._sample_image_values(
                    depth_tensor, color_tensor, image_coords, valid_mask
                )
                
                # Compute TSDF values
                tsdf_values, weights = self._compute_tsdf_values(
                    camera_coords, sampled_depth, depth_valid
                )
                
                # Update TSDF volume
                self._update_volume(tsdf_values, weights, sampled_color, valid_mask & depth_valid)
                
                # Update statistics
                self.integration_count += 1
                integration_time = time.time() - start_time
                self.total_integration_time += integration_time
                
                if self.integration_count % 10 == 0:
                    avg_time = self.total_integration_time / self.integration_count
                    print(f"Integration {self.integration_count}: {integration_time:.3f}s "
                          f"(avg: {avg_time:.3f}s)")
                
                return True
                
        except Exception as e:
            print(f"TSDF integration failed: {e}")
            return False
    
    def _generate_voxel_coordinates(self) -> torch.Tensor:
        """Generate 3D coordinates for all voxels in world space"""
        # Create coordinate grids
        x = torch.arange(self.volume_dims[0], dtype=torch.float32, device=self.device)
        y = torch.arange(self.volume_dims[1], dtype=torch.float32, device=self.device)
        z = torch.arange(self.volume_dims[2], dtype=torch.float32, device=self.device)
        
        # Scale to world coordinates
        x = x * self.config.voxel_size + self.volume_origin[0]
        y = y * self.config.voxel_size + self.volume_origin[1]
        z = z * self.config.voxel_size + self.volume_origin[2]
        
        # Create meshgrid
        X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')
        
        # Stack to get (N, 3) coordinates
        coords = torch.stack([X.flatten(), Y.flatten(), Z.flatten()], dim=1)
        
        return coords  # Shape: (N_voxels, 3)
    
    def _transform_to_camera_space(self, 
                                  world_coords: torch.Tensor, 
                                  camera_pose: torch.Tensor) -> torch.Tensor:
        """Transform world coordinates to camera space"""
        # Add homogeneous coordinate
        ones = torch.ones(world_coords.shape[0], 1, dtype=torch.float32, device=self.device)
        world_coords_hom = torch.cat([world_coords, ones], dim=1)  # (N, 4)
        
        # Transform to camera space (pose is world-to-camera)
        camera_coords_hom = torch.matmul(world_coords_hom, camera_pose.T)  # (N, 4)
        
        return camera_coords_hom[:, :3]  # Return (N, 3)
    
    def _project_to_image(self, 
                         camera_coords: torch.Tensor,
                         fx: float, fy: float, cx: float, cy: float,
                         image_shape: Tuple[int, int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Project camera coordinates to image coordinates"""
        # Perspective projection
        x = camera_coords[:, 0] / (camera_coords[:, 2] + 1e-8)  # Avoid division by zero
        y = camera_coords[:, 1] / (camera_coords[:, 2] + 1e-8)
        
        # Apply intrinsics
        u = fx * x + cx
        v = fy * y + cy
        
        # Check bounds
        height, width = image_shape
        valid_mask = (
            (u >= 0) & (u < width) & 
            (v >= 0) & (v < height) & 
            (camera_coords[:, 2] > 0)  # In front of camera
        )
        
        image_coords = torch.stack([u, v], dim=1)  # (N, 2)
        
        return image_coords, valid_mask
    
    def _sample_image_values(self, 
                           depth_image: torch.Tensor,
                           color_image: torch.Tensor,
                           image_coords: torch.Tensor,
                           valid_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample depth and color values from images using bilinear interpolation"""
        height, width = depth_image.shape
        
        # Normalize coordinates to [-1, 1] for grid_sample
        u_norm = 2.0 * image_coords[:, 0] / (width - 1) - 1.0
        v_norm = 2.0 * image_coords[:, 1] / (height - 1) - 1.0
        
        # Create grid for sampling
        grid = torch.stack([u_norm, v_norm], dim=1).unsqueeze(0).unsqueeze(0)  # (1, 1, N, 2)
        
        # Sample depth
        depth_expanded = depth_image.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        sampled_depth = F.grid_sample(
            depth_expanded, grid, mode='bilinear', padding_mode='zeros', align_corners=True
        ).squeeze()  # (N,)
        
        # Sample color
        color_expanded = color_image.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
        sampled_color = F.grid_sample(
            color_expanded, grid, mode='bilinear', padding_mode='zeros', align_corners=True
        ).squeeze(0).permute(1, 0)  # (N, 3)
        
        # Check if depth is valid (non-zero)
        depth_valid = sampled_depth > 0
        
        return sampled_depth, sampled_color, depth_valid
    
    def _compute_tsdf_values(self, 
                           camera_coords: torch.Tensor,
                           sampled_depth: torch.Tensor,
                           depth_valid: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute TSDF values and weights"""
        # Distance from camera
        voxel_depth = camera_coords[:, 2]
        
        # Signed distance function
        sdf = sampled_depth - voxel_depth
        
        # Truncate SDF
        tsdf = torch.clamp(sdf / self.config.truncation_distance, -1.0, 1.0)
        
        # Compute weights (higher weight for closer voxels)
        weights = torch.ones_like(tsdf)
        
        # Zero weight for invalid depth or voxels outside truncation distance
        weights[~depth_valid] = 0
        weights[torch.abs(sdf) > self.config.truncation_distance] = 0
        
        # Distance-based weighting
        weights = weights / (voxel_depth + 1e-8)
        
        return tsdf, weights
    
    def _update_volume(self, 
                      tsdf_values: torch.Tensor,
                      weights: torch.Tensor,
                      colors: torch.Tensor,
                      valid_mask: torch.Tensor):
        """Update TSDF volume with new values"""
        # Reshape to volume dimensions
        tsdf_vol = tsdf_values.view(self.volume_dims)
        weight_vol = weights.view(self.volume_dims)
        color_vol = colors.view(*self.volume_dims, 3)
        valid_vol = valid_mask.view(self.volume_dims)
        
        # Only update valid voxels
        update_mask = valid_vol & (weight_vol > 0)
        
        # Weighted average update
        old_tsdf = self.tsdf_volume[update_mask]
        old_weight = self.weight_volume[update_mask]
        new_tsdf = tsdf_vol[update_mask]
        new_weight = weight_vol[update_mask]
        
        # Update weights
        total_weight = old_weight + new_weight
        total_weight = torch.clamp(total_weight, 0, self.config.max_weight)
        
        # Update TSDF (weighted average)
        updated_tsdf = (old_tsdf * old_weight + new_tsdf * new_weight) / (total_weight + 1e-8)
        
        # Update color (weighted average)
        old_color = self.color_volume[update_mask]
        new_color = color_vol[update_mask]
        updated_color = (old_color * old_weight.unsqueeze(-1) + 
                        new_color * new_weight.unsqueeze(-1)) / (total_weight.unsqueeze(-1) + 1e-8)
        
        # Write back to volume
        self.tsdf_volume[update_mask] = updated_tsdf
        self.weight_volume[update_mask] = total_weight
        self.color_volume[update_mask] = updated_color
    
    def extract_mesh(self, 
                    min_weight_threshold: float = 1.0,
                    level: float = 0.0) -> Optional[o3d.geometry.TriangleMesh]:
        """
        Extract triangle mesh from TSDF volume using marching cubes
        
        Args:
            min_weight_threshold: Minimum weight for valid voxels
            level: Isosurface level for marching cubes
            
        Returns:
            Open3D triangle mesh or None if extraction fails
        """
        start_time = time.time()
        
        try:
            with self.lock:
                # Move volumes to CPU for marching cubes
                tsdf_cpu = self.tsdf_volume.cpu().numpy()
                weight_cpu = self.weight_volume.cpu().numpy()
                color_cpu = self.color_volume.cpu().numpy()
                
                # Mask out low-weight voxels
                mask = weight_cpu < min_weight_threshold
                tsdf_cpu[mask] = 1.0  # Set to positive (outside surface)
                
                # Create Open3D TSDF volume
                volume_o3d = o3d.geometry.TSDFVolume(
                    voxel_length=self.config.voxel_size,
                    sdf_trunc=self.config.truncation_distance,
                    color_type=o3d.geometry.TSDFVolumeColorType.RGB8
                )
                
                # Extract mesh using Open3D's marching cubes
                mesh = volume_o3d.extract_triangle_mesh()
                
                if len(mesh.vertices) == 0:
                    print("No vertices extracted from TSDF volume")
                    return None
                
                # Post-processing
                mesh.remove_duplicated_vertices()
                mesh.remove_degenerate_triangles()
                mesh.remove_unreferenced_vertices()
                
                # Compute normals
                mesh.compute_vertex_normals()
                mesh.compute_triangle_normals()
                
                extraction_time = time.time() - start_time
                self.last_mesh_extraction_time = extraction_time
                
                print(f"Mesh extracted: {len(mesh.vertices)} vertices, "
                      f"{len(mesh.triangles)} triangles ({extraction_time:.3f}s)")
                
                return mesh
                
        except Exception as e:
            print(f"Mesh extraction failed: {e}")
            return None
    
    def reset_volume(self):
        """Reset the TSDF volume to empty state"""
        with self.lock:
            self.tsdf_volume.zero_()
            self.weight_volume.zero_()
            self.color_volume.zero_()
            
            self.integration_count = 0
            self.total_integration_time = 0.0
            
            print("TSDF volume reset")
    
    def get_statistics(self) -> Dict[str, float]:
        """Get performance and quality statistics"""
        with self.lock:
            non_zero_voxels = torch.sum(self.weight_volume > 0).item()
            total_voxels = np.prod(self.volume_dims)
            
            avg_integration_time = (self.total_integration_time / max(1, self.integration_count))
            
            return {
                'integration_count': self.integration_count,
                'avg_integration_time': avg_integration_time,
                'last_mesh_extraction_time': self.last_mesh_extraction_time,
                'occupied_voxels': non_zero_voxels,
                'total_voxels': total_voxels,
                'volume_occupancy': non_zero_voxels / total_voxels,
                'memory_usage_mb': self.estimate_memory_usage()
            }

# Example usage and testing
def test_tsdf_fusion():
    """Test the GPU TSDF fusion with synthetic data"""
    print("Testing GPU TSDF Fusion...")
    
    # Configuration for dental scanning
    config = TSDFConfig(
        volume_size=(0.08, 0.08, 0.06),  # 8cm x 8cm x 6cm for single tooth
        voxel_size=0.001,  # 1mm voxels for high detail
        use_gpu=True
    )
    
    # Initialize TSDF fusion
    tsdf = GPUTSDFFusion(config)
    
    # Synthetic camera parameters (typical for dental cameras)
    intrinsics = {
        'fx': 800.0,
        'fy': 800.0,
        'cx': 320.0,
        'cy': 240.0
    }
    
    # Generate synthetic frames
    for i in range(10):
        # Synthetic depth image (640x480)
        depth_image = np.random.rand(480, 640) * 50 + 20  # 20-70mm depth
        
        # Synthetic color image
        color_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Synthetic camera pose (slight rotation around tooth)
        angle = i * 0.1
        pose = np.array([
            [np.cos(angle), 0, np.sin(angle), 0],
            [0, 1, 0, 0],
            [-np.sin(angle), 0, np.cos(angle), 0.05],  # 5cm from origin
            [0, 0, 0, 1]
        ])
        
        # Integrate frame
        success = tsdf.integrate_frame(depth_image, color_image, intrinsics, pose)
        print(f"Frame {i+1}: {'Success' if success else 'Failed'}")
    
    # Extract mesh
    mesh = tsdf.extract_mesh()
    if mesh:
        print(f"Final mesh: {len(mesh.vertices)} vertices")
        
        # Visualize (optional)
        # o3d.visualization.draw_geometries([mesh])
    
    # Print statistics
    stats = tsdf.get_statistics()
    print("\nTSDF Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

if __name__ == "__main__":
    test_tsdf_fusion()
