"""
Shared Memory Manager - Inter-process communication for enhanced prototype
"""

import threading
from typing import Dict, Any

class SharedMemoryManager:
    """Manage shared memory between processes"""
    
    def __init__(self):
        self.memory_blocks = {}
        self.lock = threading.Lock()
        self.latest_scan_data: Dict[str, Any] = {}
        
    def create_block(self, name: str, size: int) -> bool:
        """Create a shared memory block"""
        with self.lock:
            self.memory_blocks[name] = bytearray(size)
            return True
            
    def get_block(self, name: str) -> bytearray:
        """Get shared memory block"""
        return self.memory_blocks.get(name, bytearray())
        
    def cleanup(self):
        """Cleanup shared memory"""
        with self.lock:
            self.memory_blocks.clear()
            self.latest_scan_data.clear()

    # Added for scanning_service compatibility
    def update_scan_data(self, data: Dict[str, Any]):
        with self.lock:
            self.latest_scan_data = data.copy()

    def get_memory_usage(self) -> int:
        with self.lock:
            return sum(len(b) for b in self.memory_blocks.values())
