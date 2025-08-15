#!/usr/bin/env python3
"""
Executable inventory and classification script for IntraoralScan analysis
"""
import os
import json
from pathlib import Path
import hashlib

def get_file_info(filepath):
    """Extract basic file information"""
    stat = os.stat(filepath)
    with open(filepath, 'rb') as f:
        file_hash = hashlib.md5(f.read()).hexdigest()
    
    return {
        'path': str(filepath),
        'size': stat.st_size,
        'md5': file_hash,
        'extension': filepath.suffix.lower()
    }

def classify_executable(filename):
    """Quick classification based on naming patterns"""
    name = filename.lower()
    
    if 'launcher' in name:
        return 'bootstrapper'
    elif 'scan' in name and 'logic' in name:
        return 'scanning_logic'
    elif 'algo' in name or 'service' in name:
        return 'algorithm_service'
    elif 'network' in name:
        return 'network_service'
    elif 'order' in name:
        return 'order_management'
    elif 'design' in name:
        return 'design_service'
    elif name.endswith('.exe') and 'intraoral' in name:
        return 'main_ui'
    else:
        return 'utility'

def scan_directory(base_path):
    """Scan directory for executables and DLLs"""
    results = {
        'executables': [],
        'libraries': [],
        'python_modules': [],
        'configs': [],
        'models': []
    }
    
    for root, dirs, files in os.walk(base_path):
        for file in files:
            filepath = Path(root) / file
            file_info = get_file_info(filepath)
            
            if file.endswith('.exe'):
                file_info['classification'] = classify_executable(file)
                file_info['confidence'] = 0.8 if file_info['classification'] != 'utility' else 0.5
                results['executables'].append(file_info)
            elif file.endswith('.dll'):
                results['libraries'].append(file_info)
            elif file.endswith(('.pyd', '.pyc', '.py')):
                results['python_modules'].append(file_info)
            elif file.endswith(('.json', '.ini', '.xml', '.yaml', '.yml', '.cfg')):
                results['configs'].append(file_info)
            elif file.endswith(('.onnx', '.pb', '.engine', '.trt', '.model')):
                results['models'].append(file_info)
    
    return results

if __name__ == "__main__":
    base_path = "IntraoralScan/Bin"
    if os.path.exists(base_path):
        results = scan_directory(base_path)
        
        # Save results
        with open('analysis_output/inventory_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Found {len(results['executables'])} executables")
        print(f"Found {len(results['libraries'])} libraries")
        print(f"Found {len(results['python_modules'])} Python modules")
        print(f"Found {len(results['configs'])} config files")
        print(f"Found {len(results['models'])} model files")
    else:
        print(f"Directory {base_path} not found")