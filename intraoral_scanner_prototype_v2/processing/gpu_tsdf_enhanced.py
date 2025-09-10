"""
Enhanced GPU TSDF Fusion for Professional Dental Scanning

Based on reverse engineering analysis of Sn3DSpeckleFusion.dll (38.9MB)
Implements professional-grade TSDF with PyTorch GPU acceleration
"""

import numpy as np
import time
import threading
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass

# GPU acceleration imports with fallbacks
try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False

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

class EnhancedTSDFFusion:
    """
    Enhanced GPU-accelerated TSDF fusion for real-time dental scanning
    
    Based on analysis of IntraoralScan's volumetric fusion approach,
    implementing professional-grade TSDF with PyTorch optimization.
    """
    
    def __init__(self, config: TSDFConfig = None):
        self.config = config or TSDFConfig()
        
        # Device selection
        if TORCH_AVAILABLE and self.config.use_gpu and torch.cuda.is_available():
            self.device = torch.device(self.config.device)
            self.use_gpu = True
        else:
            self.device = torch.device("cpu")
            self.use_gpu = False
        
        # Calculate volume dimensions
        self.volume_dims = tuple(int(size / self.config.voxel_size) for size in self.config.volume_size)
        print(f"TSDF Volume dimensions: {self.volume_dims} ({np.prod(self.volume_dims):,} voxels)")
        
        # Initialize TSDF volume
        self.tsdf_volume = None
        self.weight_volume = None
        self.color_volume = None
        
        # Volume origin in world coordinates
        self.volume_origin = None
        
        # Performance monitoring
        self.integration_count = 0
        self.total_integration_time = 0.0
        self.last_mesh_extraction_time = 0.0
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Current mesh cache
        self.current_mesh = None
        self.mesh_dirty = False
        
    def initialize(self) -> bool:
        """Initialize TSDF volume and GPU resources"""
        try:
            if TORCH_AVAILABLE:
                # Initialize TSDF volume on GPU/CPU
                self.tsdf_volume = torch.zeros(self.volume_dims, dtype=torch.float32, device=self.device)
                self.weight_volume = torch.zeros(self.volume_dims, dtype=torch.float32, device=self.device)
                self.color_volume = torch.zeros((*self.volume_dims, 3), dtype=torch.float32, device=self.device)
                
                # Volume origin in world coordinates
                self.volume_origin = torch.tensor([
                    -self.config.volume_size[0] / 2,
                    -self.config.volume_size[1] / 2,
                    -self.config.volume_size[2] / 2
                ], dtype=torch.float32, device=self.device)
            else:
                # Fallback to numpy arrays
                self.tsdf_volume = np.zeros(self.volume_dims, dtype=np.float32)
                self.weight_volume = np.zeros(self.volume_dims, dtype=np.float32)
                self.color_volume = np.zeros((*self.volume_dims, 3), dtype=np.float32)
                
                self.volume_origin = np.array([
                    -self.config.volume_size[0] / 2,
                    -self.config.volume_size[1] / 2,
                    -self.config.volume_size[2] / 2
                ], dtype=np.float32)
            
            print(f"Enhanced TSDF Fusion initialized on {self.device}")
            print(f"Volume size: {self.config.volume_size} meters")
            print(f"Voxel size: {self.config.voxel_size} meters")
            print(f"Memory usage: {self.estimate_memory_usage():.1f} MB")
            
            return True
            
        except Exception as e:
            print(f"Error initializing Enhanced TSDF fusion: {e}")
            return False
    
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
                if TORCH_AVAILABLE and self.use_gpu:
                    success = self._integrate_frame_torch(
                        depth_image, color_image, camera_intrinsics, camera_pose, depth_scale)
                else:
                    success = self._integrate_frame_numpy(
                        depth_image, color_image, camera_intrinsics, camera_pose, depth_scale)
                
                if success:
                    self.integration_count += 1
                    self.mesh_dirty = True
                    
                    # Update performance metrics
                    integration_time = time.time() - start_time
                    self.total_integration_time += integration_time
                    
                    if self.integration_count % 10 == 0:
                        avg_time = self.total_integration_time / self.integration_count
                        print(f"Integration {self.integration_count}: {integration_time:.3f}s "
                              f"(avg: {avg_time:.3f}s)")
                
                return success
                
        except Exception as e:
            print(f"TSDF integration failed: {e}")
            return False
    
    def _integrate_frame_torch(self, depth_image: np.ndarray, color_image: np.ndarray,
                              camera_intrinsics: Dict[str, float], camera_pose: np.ndarray,
                              depth_scale: float) -> bool:
        """PyTorch GPU-accelerated frame integration"""
        try:
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
            self._update_volume_torch(tsdf_values, weights, sampled_color, valid_mask & depth_valid)
            
            return True
            
        except Exception as e:
            print(f"PyTorch integration error: {e}")
            return False
    
    def _integrate_frame_numpy(self, depth_image: np.ndarray, color_image: np.ndarray,
                              camera_intrinsics: Dict[str, float], camera_pose: np.ndarray,
                              depth_scale: float) -> bool:
        """NumPy CPU fallback integration"""
        try:
            # Convert inputs
            depth = depth_image.astype(np.float32) / depth_scale
            color = color_image.astype(np.float32) / 255.0
            
            # Camera parameters
            fx, fy = camera_intrinsics['fx'], camera_intrinsics['fy']
            cx, cy = camera_intrinsics['cx'], camera_intrinsics['cy']
            
            # Generate voxel coordinates (simplified for CPU)
            x_coords = np.arange(self.volume_dims[0]) * self.config.voxel_size + self.volume_origin[0]
            y_coords = np.arange(self.volume_dims[1]) * self.config.voxel_size + self.volume_origin[1]
            z_coords = np.arange(self.volume_dims[2]) * self.config.voxel_size + self.volume_origin[2]
            
            # Simple integration (basic implementation)
            for i in range(0, self.volume_dims[0], 4):  # Subsample for performance
                for j in range(0, self.volume_dims[1], 4):
                    for k in range(0, self.volume_dims[2], 4):
                        world_point = np.array([x_coords[i], y_coords[j], z_coords[k], 1.0])
                        camera_point = camera_pose @ world_point
                        
                        if camera_point[2] > 0:  # In front of camera
                            u = int(fx * camera_point[0] / camera_point[2] + cx)
                            v = int(fy * camera_point[1] / camera_point[2] + cy)
                            
                            if 0 <= u < depth.shape[1] and 0 <= v < depth.shape[0]:
                                depth_val = depth[v, u]
                                if depth_val > 0:
                                    sdf = depth_val - camera_point[2]
                                    
                                    if abs(sdf) < self.config.truncation_distance:
                                        tsdf_val = sdf / self.config.truncation_distance
                                        weight = 1.0
                                        
                                        # Update TSDF
                                        old_weight = self.weight_volume[i, j, k]
                                        new_weight = old_weight + weight
                                        
                                        if new_weight > 0:
                                            self.tsdf_volume[i, j, k] = (
                                                self.tsdf_volume[i, j, k] * old_weight + 
                                                tsdf_val * weight
                                            ) / new_weight
                                            self.weight_volume[i, j, k] = min(new_weight, self.config.max_weight)
                                            
                                            # Update color
                                            if len(color.shape) == 3:
                                                pixel_color = color[v, u]
                                                self.color_volume[i, j, k] = (
                                                    self.color_volume[i, j, k] * old_weight + 
                                                    pixel_color * weight
                                                ) / new_weight
            
            return True
            
        except Exception as e:
            print(f"NumPy integration error: {e}")
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
    
    def _transform_to_camera_space(self, world_coords: torch.Tensor, camera_pose: torch.Tensor) -> torch.Tensor:
        """Transform world coordinates to camera space"""
        # Add homogeneous coordinate
        ones = torch.ones(world_coords.shape[0], 1, dtype=torch.float32, device=self.device)
        world_coords_hom = torch.cat([world_coords, ones], dim=1)  # (N, 4)
        
        # Transform to camera space (pose is world-to-camera)
        camera_coords_hom = torch.matmul(world_coords_hom, camera_pose.T)  # (N, 4)
        
        return camera_coords_hom[:, :3]  # Return (N, 3)
    
    def _project_to_image(self, camera_coords: torch.Tensor, fx: float, fy: float, 
                         cx: float, cy: float, image_shape: Tuple[int, int]) -> Tuple[torch.Tensor, torch.Tensor]:
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
    
    def _sample_image_values(self, depth_image: torch.Tensor, color_image: torch.Tensor,
                           image_coords: torch.Tensor, valid_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
    
    def _compute_tsdf_values(self, camera_coords: torch.Tensor, sampled_depth: torch.Tensor,
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
    
    def _update_volume_torch(self, tsdf_values: torch.Tensor, weights: torch.Tensor,
                           colors: torch.Tensor, valid_mask: torch.Tensor):
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
    
    def extract_mesh(self, min_weight_threshold: float = 1.0) -> Optional[Dict[str, np.ndarray]]:
        """
        Extract triangle mesh from TSDF volume
        
        Args:
            min_weight_threshold: Minimum weight for valid voxels
            
        Returns:
            Dictionary with 'vertices', 'triangles', 'colors' or None if extraction fails
        """
        start_time = time.time()
        
        try:
            with self.lock:
                if TORCH_AVAILABLE and isinstance(self.tsdf_volume, torch.Tensor):
                    # Move volumes to CPU for marching cubes
                    tsdf_cpu = self.tsdf_volume.cpu().numpy()
                    weight_cpu = self.weight_volume.cpu().numpy()
                    color_cpu = self.color_volume.cpu().numpy()
                else:
                    tsdf_cpu = self.tsdf_volume
                    weight_cpu = self.weight_volume
                    color_cpu = self.color_volume
                
                # Mask out low-weight voxels
                mask = weight_cpu < min_weight_threshold
                tsdf_cpu[mask] = 1.0  # Set to positive (outside surface)
                
                # Simple marching cubes implementation
                vertices, triangles = self._marching_cubes_simple(tsdf_cpu)
                
                if len(vertices) == 0:
                    print("No vertices extracted from TSDF volume")
                    return None
                
                # Convert to world coordinates
                vertices = vertices * self.config.voxel_size + self.volume_origin
                
                # Generate vertex colors (simplified)
                colors = np.ones((len(vertices), 3)) * 0.8  # Default gray
                
                extraction_time = time.time() - start_time
                self.last_mesh_extraction_time = extraction_time
                
                print(f"Mesh extracted: {len(vertices)} vertices, "
                      f"{len(triangles)} triangles ({extraction_time:.3f}s)")
                
                mesh_data = {
                    'vertices': vertices,
                    'triangles': triangles,
                    'colors': colors
                }
                
                self.current_mesh = mesh_data
                self.mesh_dirty = False
                
                return mesh_data
                
        except Exception as e:
            print(f"Mesh extraction failed: {e}")
            return None
    
    def _marching_cubes_simple(self, volume: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Simple marching cubes implementation"""
        try:
            # Try to use skimage if available
            from skimage import measure
            vertices, faces, normals, values = measure.marching_cubes(volume, level=0.0)
            return vertices, faces
        except ImportError:
            # Fallback to very simple surface extraction
            print("Warning: skimage not available, using simple surface extraction")
            
            # Find surface voxels (where TSDF changes sign)
            vertices = []
            triangles = []
            
            for i in range(1, volume.shape[0]-1):
                for j in range(1, volume.shape[1]-1):
                    for k in range(1, volume.shape[2]-1):
                        if volume[i,j,k] < 0:  # Inside surface
                            # Check if any neighbor is outside
                            neighbors = [
                                volume[i+1,j,k], volume[i-1,j,k],
                                volume[i,j+1,k], volume[i,j-1,k],
                                volume[i,j,k+1], volume[i,j,k-1]
                            ]
                            if any(n > 0 for n in neighbors):
                                vertices.append([i, j, k])
            
            # Convert to numpy array
            vertices = np.array(vertices, dtype=np.float32)
            triangles = np.array([], dtype=np.int32).reshape(0, 3)  # No triangles in simple version
            
            return vertices, triangles
    
    def reset_volume(self):
        """Reset the TSDF volume to empty state"""
        with self.lock:
            if TORCH_AVAILABLE and isinstance(self.tsdf_volume, torch.Tensor):
                self.tsdf_volume.zero_()
                self.weight_volume.zero_()
                self.color_volume.zero_()
            else:
                self.tsdf_volume.fill(0)
                self.weight_volume.fill(0)
                self.color_volume.fill(0)
            
            self.integration_count = 0
            self.total_integration_time = 0.0
            self.current_mesh = None
            self.mesh_dirty = False
            
            print("Enhanced TSDF volume reset")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get performance and quality statistics"""
        with self.lock:
            if TORCH_AVAILABLE and isinstance(self.weight_volume, torch.Tensor):
                non_zero_voxels = torch.sum(self.weight_volume > 0).item()
            else:
                non_zero_voxels = np.sum(self.weight_volume > 0)
            
            total_voxels = np.prod(self.volume_dims)
            avg_integration_time = (self.total_integration_time / max(1, self.integration_count))
            
            return {
                'integration_count': self.integration_count,
                'avg_integration_time': avg_integration_time,
                'last_mesh_extraction_time': self.last_mesh_extraction_time,
                'occupied_voxels': int(non_zero_voxels),
                'total_voxels': int(total_voxels),
                'volume_occupancy': float(non_zero_voxels) / total_voxels,
                'memory_usage_mb': self.estimate_memory_usage(),
                'use_gpu': self.use_gpu,
                'device': str(self.device)
            }
    
    def is_initialized(self) -> bool:
        """Check if TSDF is initialized"""
        return self.tsdf_volume is not None
    
    def cleanup(self):
        """Cleanup resources"""
        with self.lock:
            if TORCH_AVAILABLE and isinstance(self.tsdf_volume, torch.Tensor):
                del self.tsdf_volume
                del self.weight_volume
                del self.color_volume
                torch.cuda.empty_cache()
            
            self.tsdf_volume = None
            self.weight_volume = None
            self.color_volume = None
            self.current_mesh = None

# Test function
def test_enhanced_tsdf():
    """Test the Enhanced TSDF fusion with synthetic data"""
    print("Testing Enhanced TSDF Fusion...")
    
    # Configuration for dental scanning
    config = TSDFConfig(
        volume_size=(0.08, 0.08, 0.06),  # 8cm x 8cm x 6cm
        voxel_size=0.001,  # 1mm voxels
        use_gpu=True
    )
    
    # Initialize TSDF fusion
    tsdf = EnhancedTSDFFusion(config)
    
    if not tsdf.initialize():
        print("Failed to initialize TSDF")
        return
    
    # Synthetic camera parameters
    intrinsics = {
        'fx': 800.0,
        'fy': 800.0,
        'cx': 320.0,
        'cy': 240.0
    }
    
    # Generate synthetic frames
    for i in range(5):
        # Synthetic depth image (640x480)
        depth_image = np.random.rand(480, 640) * 50 + 20  # 20-70mm depth
        
        # Synthetic color image
        color_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Synthetic camera pose
        angle = i * 0.2
        pose = np.array([
            [np.cos(angle), 0, np.sin(angle), 0],
            [0, 1, 0, 0],
            [-np.sin(angle), 0, np.cos(angle), 0.05],
            [0, 0, 0, 1]
        ])
        
        # Integrate frame
        success = tsdf.integrate_frame(depth_image, color_image, intrinsics, pose)
        print(f"Frame {i+1}: {'Success' if success else 'Failed'}")
    
    # Extract mesh
    mesh_data = tsdf.extract_mesh()
    if mesh_data:
        print(f"Final mesh: {len(mesh_data['vertices'])} vertices")
    
    # Print statistics
    stats = tsdf.get_statistics()
    print("\nEnhanced TSDF Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Cleanup
    tsdf.cleanup()

if __name__ == "__main__":
    test_enhanced_tsdf()
