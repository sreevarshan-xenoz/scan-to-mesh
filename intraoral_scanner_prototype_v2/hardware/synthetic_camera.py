"""Synthetic camera for simulation mode.
Generates deterministic color + depth frames simulating a dental arch sweep.
"""
import numpy as np
import time

class SyntheticCamera:
    def __init__(self, width=640, height=480, total_frames=500):
        self.width = width
        self.height = height
        self.frame_index = 0
        self.total_frames = total_frames
        self.is_initialized = False
        # Precompute a base depth shape (curved arch)
        u = np.linspace(-1, 1, self.width)
        v = np.linspace(-1, 1, self.height)
        uu, vv = np.meshgrid(u, v)
        arch = 0.45 + 0.05 * np.exp(-((uu*1.8)**2 + (vv*3.2)**2))  # central bump
        self.base_depth = arch.astype(np.float32)

    def initialize(self):
        self.start_time = time.time()
        self.is_initialized = True
        return True

    def is_connected(self):
        return self.is_initialized

    def get_frame_data(self):
        if not self.is_initialized or self.frame_index >= self.total_frames:
            return None
        t = self.frame_index / max(1, (self.total_frames - 1))
        # Simulate slight motion: yaw + small translation
        yaw = (t - 0.5) * 0.6  # radians
        # Add procedural noise to depth
        noise = (np.random.randn(self.height, self.width) * 0.002).astype(np.float32)
        depth = self.base_depth + noise
        # Create simple color gradient with moving highlight (avoid scalar astype)
        color = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        grad = np.linspace(0, 2 * np.pi, self.width, dtype=np.float32)
        sinus = np.sin(grad + yaw)
        channel0 = (80.0 + 60.0 * sinus).clip(0, 255)
        color[..., 0] = channel0.astype(np.uint8)
        color[..., 1] = int(t * 255)
        color[..., 2] = 200
        timestamp = time.time()
        frame = {
            'color_image': color,
            'depth_image': depth,
            'timestamp': timestamp,
            'frame_index': self.frame_index
        }
        self.frame_index += 1
        return frame

    def shutdown(self):
        self.is_initialized = False

    # Compatibility with expected camera manager interface
    def start_capture(self):
        return True

    def stop_capture(self):
        return True
