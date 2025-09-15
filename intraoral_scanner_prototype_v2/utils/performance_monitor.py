"""
Performance Monitor - Track system performance for enhanced prototype
"""

import time
try:  # Optional dependency
    import psutil  # type: ignore
    _HAS_PSUTIL = True
except Exception:  # pragma: no cover
    psutil = None  # type: ignore
    _HAS_PSUTIL = False
from typing import Dict, Any

class PerformanceMonitor:
    """Monitor system performance and resource usage with ability to merge pipeline metrics"""

    def __init__(self):
        self.start_time = time.time()
        self.metrics = {}

    def reset(self):
        """Reset collected metrics/time."""
        self.start_time = time.time()
        self.metrics.clear()

    def update_metrics(self, extra: Dict[str, Any] | None = None):
        """Update performance metrics; merge optional extra dict."""
        if _HAS_PSUTIL:
            base = {
                'cpu_percent': psutil.cpu_percent(),
                'memory_percent': psutil.virtual_memory().percent,
                'uptime_sec': time.time() - self.start_time,
                'timestamp': time.time()
            }
        else:
            base = {
                'cpu_percent': 0.0,
                'memory_percent': 0.0,
                'uptime_sec': time.time() - self.start_time,
                'timestamp': time.time(),
                'note': 'psutil not installed'
            }
        if extra:
            base.update(extra)
        self.metrics = base

    def get_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics (auto refresh)."""
        self.update_metrics()
        return self.metrics.copy()
