"""
Camera Manager - Handles camera capture and calibration
Simulates the structured light scanning approach used in professional systems
"""

import cv2
import numpy as np
import threading
import time
from typing import Tuple, Optional, List

class CameraManager:
    def __init__(self, camera_id: int = 0, width: int = 640, height: int = 480):
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.cap = None
        self.is_running = False
        self.current_frame = None
        self.frame_lock = threading.Lock()
        
        # Structured light patterns (simulated)
        self.patterns = self._generate_patterns()
        self.pattern_index = 0
        
        # Camera calibration parameters (would be loaded from calibration)
        self.camera_matrix = np.array([[500, 0, width/2],
                                     [0, 500, height/2],
                                     [0, 0, 1]], dtype=np.float32)
        self.dist_coeffs = np.zeros((4, 1))
        
    def _generate_patterns(self) -> List[np.ndarray]:
        """Generate structured light patterns for depth estimation"""
        patterns = []
        
        # Vertical stripes
        for shift in range(0, 8, 2):
            pattern = np.zeros((self.height, self.width), dtype=np.uint8)
            for x in range(self.width):
                if (x + shift) % 16 < 8:
                    pattern[:, x] = 255
            patterns.append(pattern)
            
        # Horizontal stripes  
        for shift in range(0, 8, 2):
            pattern = np.zeros((self.height, self.width), dtype=np.uint8)
            for y in range(self.height):
                if (y + shift) % 16 < 8:
                    pattern[y, :] = 255
            patterns.append(pattern)
            
        return patterns
    
    def start_capture(self) -> bool:
        """Start camera capture"""
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            return False
            
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        self.is_running = True
        self.capture_thread = threading.Thread(target=self._capture_loop)
        self.capture_thread.start()
        
        return True
    
    def _capture_loop(self):
        """Main capture loop running in separate thread"""
        while self.is_running:
            ret, frame = self.cap.read()
            if ret:
                with self.frame_lock:
                    self.current_frame = frame.copy()
            time.sleep(1/30)  # 30 FPS
    
    def get_frame(self) -> Optional[np.ndarray]:
        """Get current frame"""
        with self.frame_lock:
            return self.current_frame.copy() if self.current_frame is not None else None
    
    def get_structured_light_pair(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Simulate structured light capture by getting frame with and without pattern"""
        # In real system, this would project pattern and capture
        # For simulation, we'll use current frame as "with pattern"
        frame_with_pattern = self.get_frame()
        
        if frame_with_pattern is None:
            return None, None
            
        # Simulate pattern projection by overlaying current pattern
        current_pattern = self.patterns[self.pattern_index % len(self.patterns)]
        pattern_3ch = cv2.cvtColor(current_pattern, cv2.COLOR_GRAY2BGR)
        
        # Blend pattern with captured frame
        frame_with_pattern = cv2.addWeighted(frame_with_pattern, 0.7, pattern_3ch, 0.3, 0)
        
        # Simulate frame without pattern (just current frame)
        frame_without_pattern = self.get_frame()
        
        self.pattern_index += 1
        
        return frame_with_pattern, frame_without_pattern
    
    def undistort_frame(self, frame: np.ndarray) -> np.ndarray:
        """Apply camera calibration to undistort frame"""
        return cv2.undistort(frame, self.camera_matrix, self.dist_coeffs)
    
    def stop_capture(self):
        """Stop camera capture"""
        self.is_running = False
        if hasattr(self, 'capture_thread'):
            self.capture_thread.join()
        if self.cap:
            self.cap.release()
    
    def calibrate_camera(self, calibration_images: List[np.ndarray]) -> bool:
        """Perform camera calibration using checkerboard pattern"""
        # Simplified calibration - in real system this would be more sophisticated
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        
        # Prepare object points (checkerboard corners in 3D)
        pattern_size = (9, 6)  # 9x6 checkerboard
        objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
        
        objpoints = []  # 3D points
        imgpoints = []  # 2D points
        
        for img in calibration_images:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
            
            if ret:
                objpoints.append(objp)
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners2)
        
        if len(objpoints) > 0:
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
                objpoints, imgpoints, (self.width, self.height), None, None)
            
            if ret:
                self.camera_matrix = mtx
                self.dist_coeffs = dist
                return True
        
        return False