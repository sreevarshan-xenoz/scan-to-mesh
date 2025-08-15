#!/usr/bin/env python3
"""
Quick test of DLL export analysis on a single high-value DLL
"""

import subprocess
import sys
from pathlib import Path

def test_single_dll(dll_path):
    """Test export analysis on a single DLL"""
    print(f"Testing DLL: {dll_path}")
    
    # Test objdump
    try:
        result = subprocess.run(['objdump', '-p', str(dll_path)], 
                              capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            lines = result.stdout.split('\n')
            export_lines = [line for line in lines if 'Export' in line or 'DLL Name' in line]
            print(f"objdump found {len(export_lines)} export-related lines")
            for line in export_lines[:5]:
                print(f"  {line.strip()}")
        else:
            print(f"objdump failed: {result.stderr}")
    except Exception as e:
        print(f"objdump error: {e}")
    
    # Test strings for CUDA kernels
    try:
        result = subprocess.run(['strings', '-n', '10', str(dll_path)], 
                              capture_output=True, text=True, timeout=15)
        if result.returncode == 0:
            lines = result.stdout.split('\n')
            cuda_lines = [line for line in lines if any(term in line.lower() 
                         for term in ['kernel', 'cuda', 'sn3d', '__global__'])]
            print(f"strings found {len(cuda_lines)} CUDA-related lines")
            for line in cuda_lines[:3]:
                print(f"  {line.strip()[:100]}...")
        else:
            print(f"strings failed: {result.stderr}")
    except Exception as e:
        print(f"strings error: {e}")

if __name__ == "__main__":
    base_path = Path(".")
    dll_path = base_path / "IntraoralScan" / "Bin" / "Sn3DSpeckleFusion.dll"
    
    if dll_path.exists():
        test_single_dll(dll_path)
    else:
        print(f"DLL not found: {dll_path}")