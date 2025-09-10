#!/usr/bin/env python3
"""
Verification script for dependency mapping task completion
"""
import os
import sqlite3
import json
from pathlib import Path

def verify_task_completion():
    """Verify that all task requirements have been met"""
    print("=" * 60)
    print("TASK 2.2 VERIFICATION: Build dependency mapping for high-value targets")
    print("=" * 60)
    
    verification_results = {
        'dependencies_tool_integration': False,
        'unified_database_created': False,
        'quick_overview_generated': False,
        'fallback_implemented': False,
        'high_value_targets_analyzed': False
    }
    
    # Check 1: Dependencies tool integration (with fallback)
    print("\n1. Dependencies tool integration:")
    if os.path.exists('analysis_scripts/dependency_mapper.py'):
        with open('analysis_scripts/dependency_mapper.py', 'r') as f:
            content = f.read()
            if 'analyze_with_dependencies_tool' in content:
                print("   ✓ Dependencies tool integration implemented")
                verification_results['dependencies_tool_integration'] = True
            else:
                print("   ✗ Dependencies tool integration missing")
    
    # Check 2: Unified SQLite database
    print("\n2. Unified SQLite database:")
    if os.path.exists('analysis_results.db'):
        try:
            conn = sqlite3.connect('analysis_results.db')
            cursor = conn.cursor()
            
            # Check if required tables exist
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            
            required_tables = ['executables', 'dependencies', 'dependency_analysis']
            if all(table in tables for table in