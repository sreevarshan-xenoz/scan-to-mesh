"""
Performance Monitor - Track system performance for enhanced prototype
"""

import time
import psutil
from typing import Dict, Any

class PerformanceMonitor:
    """Monitor system performance and resource usage"""
    
    def __init__(self):
        self.start_time = time.time()
        self.metrics = {}
        
    def update_metrics(self):
        """Update performance metrics"""
        self.metrics = {
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'timestamp': time.time()
        }
        
    def get_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        self.update_metrics()
        return self.metrics.copy()
