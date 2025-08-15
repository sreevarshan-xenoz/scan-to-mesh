#!/usr/bin/env python3
"""
Database utilities for querying analysis results
"""
import sqlite3
import json
from pathlib import Path

class AnalysisDatabase:
    def __init__(self, db_path="analysis_results.db"):
        self.db_path = db_path
    
    def get_executable_dependencies(self, executable_name):
        """Get all dependencies for a specific executable"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT 
                d.dependency_name,
                d.dependency_path,
                d.dependency_type,
                d.is_system_library,
                d.is_found,
                d.analysis_method
            FROM dependencies d
            JOIN executables e ON d.executable_id = e.id
            WHERE e.name = ?
            ORDER BY d.is_system_library, d.dependency_name
        ''', (executable_name,))
        
        dependencies = []
        for row in cursor.fetchall():
            dependencies.append({
                'name': row[0],
                'path': row[1],
                'type': row[2],
                'is_system': bool(row[3]),
                'found': bool(row[4]),
                'analysis_method': row[5]
            })
        
        conn.close()
        return dependencies
    
    def get_dependency_summary(self):
        """Get summary of all dependency analysis"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT 
                e.name,
                e.classification,
                e.size_bytes,
                da.total_dependencies,
                da.system_dependencies,
                da.missing_dependencies,
                da.analysis_method,
                da.analysis_success
            FROM executables e
            LEFT JOIN dependency_analysis da ON e.id = da.executable_id
            ORDER BY e.size_bytes DESC
        ''')
        
        summary = []
        for row in cursor.fetchall():
            summary.append({
                'executable': row[0],
                'classification': row[1],
                'size_mb': round(row[2] / (1024*1024), 2) if row[2] else 0,
                'total_dependencies': row[3] or 0,
                'system_dependencies': row[4] or 0,
                'missing_dependencies': row[5] or 0,
                'analysis_method': row[6] or 'none',
                'analysis_success': bool(row[7]) if row[7] is not None else False
            })
        
        conn.close()
        return summary
    
    def find_common_dependencies(self):
        """Find dependencies shared across multiple executables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT 
                d.dependency_name,
                COUNT(DISTINCT d.executable_id) as usage_count,
                d.is_system_library
            FROM dependencies d
            GROUP BY d.dependency_name, d.is_system_library
            HAVING COUNT(DISTINCT d.executable_id) > 1
            ORDER BY usage_count DESC, d.dependency_name
        ''')
        
        common_deps = []
        for row in cursor.fetchall():
            common_deps.append({
                'dependency': row[0],
                'used_by_count': row[1],
                'is_system': bool(row[2])
            })
        
        conn.close()
        return common_deps
    
    def export_to_json(self, output_file="analysis_output/database_export.json"):
        """Export all database contents to JSON"""
        export_data = {
            'summary': self.get_dependency_summary(),
            'common_dependencies': self.find_common_dependencies(),
            'detailed_dependencies': {}
        }
        
        # Get detailed dependencies for each executable
        summary = self.get_dependency_summary()
        for item in summary:
            if item['analysis_success']:
                deps = self.get_executable_dependencies(item['executable'])
                export_data['detailed_dependencies'][item['executable']] = deps
        
        # Ensure output directory exists
        Path(output_file).parent.mkdir(exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        return export_data

if __name__ == "__main__":
    db = AnalysisDatabase()
    
    print("=== DEPENDENCY ANALYSIS SUMMARY ===")
    summary = db.get_dependency_summary()
    
    for item in summary:
        if item['analysis_success']:
            print(f"{item['executable']}: {item['total_dependencies']} deps "
                  f"({item['system_dependencies']} system, {item['missing_dependencies']} missing)")
    
    print("\n=== COMMON DEPENDENCIES ===")
    common = db.find_common_dependencies()
    for dep in common[:10]:  # Top 10
        print(f"{dep['dependency']}: used by {dep['used_by_count']} executables")
    
    # Export to JSON
    export_data = db.export_to_json()
    print(f"\nExported {len(export_data['detailed_dependencies'])} executable analyses to JSON")