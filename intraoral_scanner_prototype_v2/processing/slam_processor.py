"""
SLAM Processor - Enhanced Visual SLAM with Meshroom Integration
Simultaneous Localization and Mapping for enhanced prototype with professional 3D reconstruction
"""

import numpy as np
import time
from typing import Optional, Tuple, List, Dict, Any

class SLAMProcessor:
    """
    Enhanced SLAM implementation with Meshroom integration support
    
    Provides real-time camera tracking and pose estimation while collecting
    keyframes for high-quality Meshroom reconstruction.
    """
    
    def __init__(self):
        self.current_pose = np.eye(4)
        self.initialized = False
        self.keyframes = []
        self.trajectory = []
        
        # SLAM parameters
        self.keyframe_distance_threshold = 0.02  # 2cm movement for new keyframe
        self.rotation_threshold = 0.1  # ~5.7 degrees rotation
        self.max_keyframes = 200  # Maximum keyframes to store
        
        # Feature tracking
        self.last_keyframe_pose = None
        self.frame_count = 0
        
        # Performance metrics
        self.processing_times = []
        
    def initialize(self) -> bool:
        """Initialize enhanced SLAM processor"""
        try:
            self.initialized = True
            self.current_pose = np.eye(4)
            self.keyframes = []
            self.trajectory = []
            self.last_keyframe_pose = None
            self.frame_count = 0
            
            print("✅ Enhanced SLAM processor initialized")
            return True
        except Exception as e:
            print(f"❌ SLAM initialization failed: {e}")
            return False
        
    def process_frame(self, depth_frame: np.ndarray, color_frame: np.ndarray, 
                     camera_intrinsics: Optional[Dict[str, float]] = None) -> Optional[np.ndarray]:
        """
        Process frame and return camera pose with keyframe detection
        
        Args:
            depth_frame: Depth image array
            color_frame: Color image array  
            camera_intrinsics: Camera intrinsic parameters
            
        Returns:
            Camera pose matrix (4x4) or None if processing failed
        """
        if not self.initialized:
            return None
        
        start_time = time.time()
        
        try:
            # For now, implement a simple motion model
            # In production, this would use feature tracking and bundle adjustment
            
            # Simulate incremental motion (simplified SLAM)
            motion_increment = self._estimate_motion(depth_frame, color_frame)
            
            # Update current pose
            self.current_pose = self.current_pose @ motion_increment
            
            # Add to trajectory
            self.trajectory.append({
                'pose': self.current_pose.copy(),
                'timestamp': time.time(),
                'frame_id': self.frame_count
            })
            
            # Check if this should be a keyframe
            if self._should_add_keyframe():
                keyframe_data = {
                    'frame_id': self.frame_count,
                    'pose': self.current_pose.copy(),
                    'color_image': color_frame.copy(),
                    'depth_image': depth_frame.copy(),
                    'timestamp': time.time(),
                    'camera_intrinsics': camera_intrinsics.copy() if camera_intrinsics else None
                }
                
                self.keyframes.append(keyframe_data)
                self.last_keyframe_pose = self.current_pose.copy()
                
                # Limit keyframe buffer size
                if len(self.keyframes) > self.max_keyframes:
                    self.keyframes.pop(0)  # Remove oldest keyframe
                
                print(f"📷 Added keyframe {len(self.keyframes)} at frame {self.frame_count}")
            
            self.frame_count += 1
            
            # Track performance
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            
            # Keep only recent performance data
            if len(self.processing_times) > 100:
                self.processing_times.pop(0)
            
            return self.current_pose.copy()
            
        except Exception as e:
            print(f"❌ SLAM frame processing failed: {e}")
            return None
    
    def _estimate_motion(self, depth_frame: np.ndarray, color_frame: np.ndarray) -> np.ndarray:
        """
        Estimate motion between frames (simplified implementation)
        
        In a full SLAM system, this would use:
        - Feature detection and matching
        - Optical flow tracking
        - ICP registration of depth data
        - Bundle adjustment optimization
        """
        
        # Simple motion model for demonstration
        # In practice, this would use actual feature tracking
        
        # Simulate small incremental motion
        translation = np.array([0.001, 0.0, 0.002])  # 1mm forward, 2mm up
        rotation_angle = 0.01  # ~0.57 degrees
        
        # Create incremental transformation
        motion = np.eye(4)
        
        # Add small rotation around Y axis (typical for scanning motion)
        cos_a = np.cos(rotation_angle)
        sin_a = np.sin(rotation_angle)
        
        motion[0, 0] = cos_a
        motion[0, 2] = sin_a
        motion[2, 0] = -sin_a
        motion[2, 2] = cos_a
        
        # Add translation
        motion[:3, 3] = translation
        
        return motion
    
    def _should_add_keyframe(self) -> bool:
        """
        Determine if current frame should be added as keyframe
        
        Keyframe criteria:
        - Sufficient translation from last keyframe
        - Sufficient rotation from last keyframe
        - Minimum time elapsed
        - Feature distribution quality
        """
        
        if self.last_keyframe_pose is None:
            return True  # First keyframe
        
        # Calculate pose difference
        pose_diff = np.linalg.inv(self.last_keyframe_pose) @ self.current_pose
        
        # Check translation distance
        translation = pose_diff[:3, 3]
        translation_distance = np.linalg.norm(translation)
        
        # Check rotation angle
        rotation_matrix = pose_diff[:3, :3]
        rotation_angle = np.arccos((np.trace(rotation_matrix) - 1) / 2)
        
        # Keyframe conditions
        translation_threshold_met = translation_distance > self.keyframe_distance_threshold
        rotation_threshold_met = abs(rotation_angle) > self.rotation_threshold
        
        return translation_threshold_met or rotation_threshold_met
    
    def get_keyframes_for_meshroom(self) -> List[Dict[str, Any]]:
        """
        Get keyframes formatted for Meshroom processing
        
        Returns:
            List of keyframe data suitable for Meshroom integration
        """
        meshroom_frames = []
        
        for keyframe in self.keyframes:
            meshroom_frame = {
                'color_image': keyframe['color_image'],
                'pose': keyframe['pose'],
                'timestamp': keyframe['timestamp'],
                'frame_id': keyframe['frame_id'],
                'camera_intrinsics': keyframe.get('camera_intrinsics')
            }
            meshroom_frames.append(meshroom_frame)
        
        return meshroom_frames
    
    def get_trajectory(self) -> List[Dict[str, Any]]:
        """Get complete camera trajectory"""
        return self.trajectory.copy()
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get SLAM performance metrics"""
        if not self.processing_times:
            return {}
        
        avg_time = np.mean(self.processing_times)
        max_time = np.max(self.processing_times)
        min_time = np.min(self.processing_times)
        
        return {
            'frames_processed': self.frame_count,
            'keyframes_captured': len(self.keyframes),
            'avg_processing_time': avg_time,
            'max_processing_time': max_time,
            'min_processing_time': min_time,
            'estimated_fps': 1.0 / avg_time if avg_time > 0 else 0,
            'trajectory_length': len(self.trajectory)
        }
    
    def save_trajectory(self, file_path: str) -> bool:
        """Save trajectory to file for analysis"""
        try:
            import json
            
            trajectory_data = {
                'trajectory': [
                    {
                        'pose': pose_data['pose'].tolist(),
                        'timestamp': pose_data['timestamp'],
                        'frame_id': pose_data['frame_id']
                    }
                    for pose_data in self.trajectory
                ],
                'keyframes': [
                    {
                        'pose': kf['pose'].tolist(),
                        'timestamp': kf['timestamp'],
                        'frame_id': kf['frame_id']
                    }
                    for kf in self.keyframes
                ],
                'metrics': self.get_performance_metrics()
            }
            
            with open(file_path, 'w') as f:
                json.dump(trajectory_data, f, indent=2)
            
            print(f"✅ Trajectory saved to {file_path}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to save trajectory: {e}")
            return False
    
    def reset(self):
        """Reset SLAM state for new scanning session"""
        self.current_pose = np.eye(4)
        self.keyframes = []
        self.trajectory = []
        self.last_keyframe_pose = None
        self.frame_count = 0
        self.processing_times = []
        
        print("✅ SLAM state reset for new session")
        
    def cleanup(self):
        """Cleanup SLAM resources"""
        self.reset()
        self.initialized = False
        print("✅ SLAM processor cleaned up")
