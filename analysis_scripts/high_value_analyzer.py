#!/usr/bin/env python3
"""
High-value executable analyzer for IntraoralScan
"""
import json
import os
from pathlib import Path

def analyze_high_value_executables():
    """Analyze the key executables identified in our classification"""
    
    # Load inventory results
    with open('analysis_output/inventory_results.json', 'r') as f:
        data = json.load(f)
    
    # High-value targets based on our design
    high_value_targets = [
        'IntraoralScan.exe',
        'DentalAlgoService.exe', 
        'DentalScanAppLogic.exe',
        'DentalLauncher.exe',
        'DentalNetwork.exe',
        'DentalOrderAppLogic.exe',
        'DentalDesignAppLogic.exe'
    ]
    
    analysis_results = {
        'high_value_executables': [],
        'key_algorithm_dlls': [],
        'ai_models_found': [],
        'config_files_found': []
    }
    
    # Find high-value executables
    for exe in data['executables']:
        filename = Path(exe['path']).name
        if filename in high_value_targets:
            exe_info = {
                'name': filename,
                'path': exe['path'],
                'size_mb': round(exe['size'] / (1024*1024), 2),
                'classification': exe.get('classification', 'unknown'),
                'confidence': exe.get('confidence', 0.0),
                'md5': exe['md5']
            }
            analysis_results['high_value_executables'].append(exe_info)
    
    # Find key algorithm DLLs
    algorithm_patterns = ['Sn3D', 'algorithm', 'Dental', 'opencv', 'cuda']
    for dll in data['libraries']:
        filename = Path(dll['path']).name.lower()
        if any(pattern.lower() in filename for pattern in algorithm_patterns):
            if dll['size'] > 1024*1024:  # Only large DLLs (>1MB)
                dll_info = {
                    'name': Path(dll['path']).name,
                    'path': dll['path'],
                    'size_mb': round(dll['size'] / (1024*1024), 2),
                    'md5': dll['md5']
                }
                analysis_results['key_algorithm_dlls'].append(dll_info)
    
    # Sort by size (largest first)
    analysis_results['key_algorithm_dlls'].sort(key=lambda x: x['size_mb'], reverse=True)
    analysis_results['high_value_executables'].sort(key=lambda x: x['size_mb'], reverse=True)
    
    # AI models
    analysis_results['ai_models_found'] = data['models']
    
    # Config files (sample)
    analysis_results['config_files_found'] = data['configs'][:20]  # First 20 for brevity
    
    return analysis_results

def generate_file_analysis():
    """Generate basic file analysis using available tools"""
    results = analyze_high_value_executables()
    
    # Use 'file' command to identify file types
    for exe in results['high_value_executables']:
        try:
            import subprocess
            result = subprocess.run(['file', exe['path']], 
                                  capture_output=True, text=True)
            exe['file_type'] = result.stdout.strip()
        except:
            exe['file_type'] = 'Unknown'
    
    return results

if __name__ == "__main__":
    results = generate_file_analysis()
    
    # Save detailed analysis
    with open('analysis_output/high_value_analysis.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("=== HIGH-VALUE EXECUTABLE ANALYSIS ===")
    print(f"Found {len(results['high_value_executables'])} high-value executables")
    print(f"Found {len(results['key_algorithm_dlls'])} key algorithm DLLs")
    print(f"Found {len(results['ai_models_found'])} AI model files")
    print(f"Found {len(results['config_files_found'])} config files")
    
    print("\n=== TOP EXECUTABLES BY SIZE ===")
    for exe in results['high_value_executables'][:5]:
        print(f"{exe['name']}: {exe['size_mb']}MB - {exe['classification']}")
    
    print("\n=== TOP ALGORITHM DLLs BY SIZE ===")
    for dll in results['key_algorithm_dlls'][:10]:
        print(f"{dll['name']}: {dll['size_mb']}MB")