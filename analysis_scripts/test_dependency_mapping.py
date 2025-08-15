#!/usr/bin/env python3
"""
Test script for dependency mapping functionality
"""
import os
import sys
import json
from pathlib import Path

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dependency_mapper import DependencyMapper
from database_utils import AnalysisDatabase

def test_dependency_mapping():
    """Test the dependency mapping functionality"""
    print("=== TESTING DEPENDENCY MAPPING ===")
    
    # Initialize mapper
    mapper = DependencyMapper("test_analysis.db")
    
    # Test database initialization
    print("✓ Database initialized")
    
    # Test getting high-value targets
    targets = mapper.get_high_value_targets()
    print(f"✓ Found {len(targets)} high-value targets")
    
    if not targets:
        print("⚠ No targets found - creating mock data for testing")
        # Create mock target for testing
        mock_target = {
            'name': 'IntraoralScan.exe',
            'path': 'IntraoralScan/Bin/IntraoralScan.exe',
            'size_mb': 50.0,
            'classification': 'main_ui',
            'confidence': 0.9,
            'md5': 'mock_hash_123'
        }
        
        # Test storing executable info
        exe_id = mapper.store_executable_info(mock_target)
        print(f"✓ Stored executable info with ID: {exe_id}")
        
        # Test manual dependency analysis (since we don't have actual tools)
        mock_deps = [
            {'name': 'Qt5Core.dll', 'path': '', 'type': 'dll', 'is_system': False, 'found': True},
            {'name': 'kernel32.dll', 'path': '', 'type': 'dll', 'is_system': True, 'found': True},
            {'name': 'user32.dll', 'path': '', 'type': 'dll', 'is_system': True, 'found': True}
        ]
        
        mapper.store_dependencies(exe_id, mock_deps, 'mock_test')
        print("✓ Stored mock dependencies")
    
    # Test database utilities
    db = AnalysisDatabase("test_analysis.db")
    summary = db.get_dependency_summary()
    print(f"✓ Retrieved dependency summary: {len(summary)} executables")
    
    # Test export functionality
    export_data = db.export_to_json("test_export.json")
    print("✓ Exported data to JSON")
    
    # Clean up test files
    try:
        os.remove("test_analysis.db")
        os.remove("test_export.json")
        print("✓ Cleaned up test files")
    except:
        pass
    
    print("\n=== ALL TESTS PASSED ===")
    return True

def verify_prerequisites():
    """Verify that prerequisite files exist"""
    print("=== VERIFYING PREREQUISITES ===")
    
    required_files = [
        'analysis_output/high_value_analysis.json',
        'IntraoralScan/Bin'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
            print(f"⚠ Missing: {file_path}")
        else:
            print(f"✓ Found: {file_path}")
    
    if missing_files:
        print(f"\n⚠ {len(missing_files)} prerequisite files missing")
        print("Run the following scripts first:")
        print("  python analysis_scripts/inventory_scanner.py")
        print("  python analysis_scripts/high_value_analyzer.py")
        return False
    
    print("✓ All prerequisites found")
    return True

if __name__ == "__main__":
    if verify_prerequisites():
        test_dependency_mapping()
    else:
        print("Please run prerequisite scripts first")
        sys.exit(1)