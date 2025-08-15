#!/usr/bin/env python3
"""
Generate a comprehensive summary of dependency mapping results
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from database_utils import AnalysisDatabase

def generate_dependency_summary():
    """Generate comprehensive dependency analysis summary"""
    db = AnalysisDatabase()
    
    print("=" * 60)
    print("DEPENDENCY MAPPING ANALYSIS SUMMARY")
    print("=" * 60)
    
    # Get overall summary
    summary = db.get_dependency_summary()
    total_deps = sum(item['total_dependencies'] for item in summary)
    total_system = sum(item['system_dependencies'] for item in summary)
    total_app = total_deps - total_system
    
    print(f"\nOVERALL STATISTICS:")
    print(f"  Executables analyzed: {len(summary)}")
    print(f"  Total dependencies found: {total_deps}")
    print(f"  System libraries: {total_system} ({total_system/total_deps*100:.1f}%)")
    print(f"  Application libraries: {total_app} ({total_app/total_deps*100:.1f}%)")
    
    print(f"\nEXECUTABLE BREAKDOWN:")
    for item in summary:
        if item['analysis_success']:
            app_deps = item['total_dependencies'] - item['system_dependencies']
            print(f"  {item['executable']:<25} | {item['size_mb']:>6.1f}MB | "
                  f"{item['total_dependencies']:>3} deps ({item['system_dependencies']:>2} sys, {app_deps:>2} app)")
    
    # Common dependencies analysis
    common_deps = db.find_common_dependencies()
    
    print(f"\nCOMMON DEPENDENCIES (used by multiple executables):")
    system_common = [d for d in common_deps if d['is_system']]
    app_common = [d for d in common_deps if not d['is_system']]
    
    print(f"  System libraries shared across executables:")
    for dep in system_common[:10]:
        print(f"    {dep['dependency']:<30} | used by {dep['used_by_count']} executables")
    
    print(f"  Application libraries shared across executables:")
    for dep in app_common[:10]:
        print(f"    {dep['dependency']:<30} | used by {dep['used_by_count']} executables")
    
    # Key insights
    print(f"\nKEY INSIGHTS:")
    
    # Find the executable with most dependencies
    max_deps_exe = max(summary, key=lambda x: x['total_dependencies'])
    print(f"  • {max_deps_exe['executable']} has the most dependencies ({max_deps_exe['total_dependencies']})")
    
    # Find most commonly used application library
    if app_common:
        most_used_app_lib = app_common[0]
        print(f"  • {most_used_app_lib['dependency']} is used by all {most_used_app_lib['used_by_count']} executables")
    
    # Technology stack identification
    qt_deps = [d for d in common_deps if 'qt5' in d['dependency'].lower()]
    if qt_deps:
        print(f"  • Qt5 framework detected ({len(qt_deps)} Qt5 libraries found)")
    
    opencv_deps = [d for d in common_deps if 'opencv' in d['dependency'].lower()]
    if opencv_deps:
        print(f"  • OpenCV computer vision library detected")
    
    dental_deps = [d for d in common_deps if 'dental' in d['dependency'].lower() or 'sn' in d['dependency'].lower()]
    if dental_deps:
        print(f"  • {len(dental_deps)} custom dental/scanning libraries identified")
    
    print(f"\nANALYSIS METHOD:")
    print(f"  • Used manual string analysis (Dependencies tool not available)")
    print(f"  • Extracted DLL references from executable strings")
    print(f"  • All dependencies marked as 'missing' (paths not resolved)")
    print(f"  • Results stored in SQLite database: analysis_results.db")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    generate_dependency_summary()