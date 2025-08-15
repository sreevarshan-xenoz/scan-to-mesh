"""
Stereo Processor - Generates depth maps from stereo camera pairs
Implements structured light depth estimation similar to professional scanners
"""

import cv2
import numpy as np
from typing import Tuple, Optional

class StereoProcessor:
    def __init__(self, baseline: float = 0.1, focal_length: float = 500.0):
        self.baseline = baseline  # Distance between cameras in meters
        self.focal_length = focal_length  # Focal length in pixels
        
        # Stereo matcher for depth estimation
        self.stereo = cv2.StereoBM_create(numDisparities=64, blockSize=15)
        
        # Alternative: StereoSGBM for better quality
        self.stereo_sgbm = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=64,
            blockSize=5,
            P1=8 * 3 * 5**2,
            P2=32 * 3 * 5**2,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=32
        )
        
    def compute_depth_from_stereo(self, left_img: np.ndarray, right_img: np.ndarray) -> np.ndarray:
        """Compute depth map from stereo pair"""
        # Convert to grayscale
        if len(left_img.shape) == 3:
            left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
        else:
            left_gray = left_img
            
        if len(right_img.shape) == 3:
            right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
        else:
            right_gray = right_img
        
        # Compute disparity
        disparity = self.stereo_sgbm.compute(left_gray, right_gray).astype(np.float32) / 16.0
        
        # Convert disparity to depth
        # Depth = (baseline * focal_length) / disparity
        depth = np.zeros_like(disparity)
        valid_pixels = disparity > 0
        depth[valid_pixels] = (self.baseline * self.focal_length) / disparity[valid_pixels]
        
        return depth
    
    def compute_depth_from_structured_light(self, 
                                          frame_with_pattern: np.ndarray, 
                                          frame_without_pattern: np.ndarray) -> np.ndarray:
        """Compute depth using structured light approach"""
        # Convert to grayscale
        if len(frame_with_pattern.shape) == 3:
            with_pattern = cv2.cvtColor(frame_with_pattern, cv2.COLOR_BGR2GRAY)
        else:
            with_pattern = frame_with_pattern
            
        if len(frame_without_pattern.shape) == 3:
            without_pattern = cv2.cvtColor(frame_without_pattern, cv2.COLOR_BGR2GRAY)
        else:
            without_pattern = frame_without_pattern
        
        # Compute pattern difference
        pattern_diff = cv2.absdiff(with_pattern, without_pattern)
        
        # Apply threshold to get pattern regions
        _, pattern_mask = cv2.threshold(pattern_diff, 30, 255, cv2.THRESH_BINARY)
        
        # Simplified depth estimation based on pattern deformation
        # In real system, this would analyze pattern phase shifts
        depth = np.zeros_like(pattern_diff, dtype=np.float32)
        
        # Use pattern intensity as rough depth estimate (simplified)
        depth = pattern_diff.astype(np.float32) / 255.0 * 0.1  # Scale to ~10cm range
        
        # Apply median filter to reduce noise
        depth = cv2.medianBlur(depth, 5)
        
        return depth
    
    def generate_point_cloud(self, 
                           color_img: np.ndarray, 
                           depth_img: np.ndarray, 
                           camera_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Generate 3D point cloud from depth image"""
        height, width = depth_img.shape
        
        # Create coordinate grids
        u, v = np.meshgrid(np.arange(width), np.arange(height))
        
        # Get camera intrinsics
        fx = camera_matrix[0, 0]
        fy = camera_matrix[1, 1]
        cx = camera_matrix[0, 2]
        cy = camera_matrix[1, 2]
        
        # Convert to 3D coordinates
        valid_depth = depth_img > 0
        z = depth_img[valid_depth]
        x = (u[valid_depth] - cx) * z / fx
        y = (v[valid_depth] - cy) * z / fy
        
        # Stack coordinates
        points = np.column_stack((x, y, z))
        
        # Get corresponding colors
        if len(color_img.shape) == 3:
            colors = color_img[valid_depth] / 255.0  # Normalize to [0,1]
        else:
            # Grayscale to RGB
            gray_colors = color_img[valid_depth] / 255.0
            colors = np.column_stack((gray_colors, gray_colors, gray_colors))
        
        return points, colors
    
    def filter_point_cloud(self, points: np.ndarray, colors: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply basic filtering to point cloud"""
        # Remove points too close or too far
        valid_z = (points[:, 2] > 0.01) & (points[:, 2] < 0.5)  # 1cm to 50cm
        
        # Remove statistical outliers (simplified)
        if len(points) > 100:
            # Compute distances to neighbors (simplified)
            from scipy.spatial.distance import cdist
            if len(points) > 1000:
                # Sample for performance
                sample_idx = np.random.choice(len(points), 1000, replace=False)
                sample_points = points[sample_idx]
            else:
                sample_points = points
                sample_idx = np.arange(len(points))
            
            # Find outliers based on distance to nearest neighbors
            distances = cdist(sample_points, sample_points)
            np.fill_diagonal(distances, np.inf)
            min_distances = np.min(distances, axis=1)
            
            # Remove points with unusually large distances to neighbors
            distance_threshold = np.percentile(min_distances, 95)
            valid_distances = min_distances < distance_threshold
            
            if len(points) > 1000:
                # Map back to full point cloud
                valid_sample = valid_z[sample_idx] & valid_distances
                valid_full = np.zeros(len(points), dtype=bool)
                valid_full[sample_idx] = valid_sample
                valid_z = valid_z & valid_full
            else:
                valid_z = valid_z & valid_distances
        
        return points[valid_z], colors[valid_z]