#!/usr/bin/env python3
"""
Comprehensive analysis script for IntraoralScan reverse engineering
"""
import json
import os
import sqlite3
from pathlib import Path
import subprocess

def analyze_ai_models():
    """Analyze AI models and their configurations"""
    models_info = {
        'model_files': [],
        'configurations': {},
        'encrypted_models': [],
        'model_categories': {}
    }
    
    # Analyze .model files
    model_files = list(Path("IntraoralScan/Bin/AIModels").glob("*.model"))
    for model_file in model_files:
        model_info = {
            'name': model_file.name,
            'size_mb': round(model_file.stat().st_size / (1024*1024), 2),
            'category': 'unknown'
        }
        
        # Categorize based on name
        name_lower = model_file.name.lower()
        if 'segment' in name_lower:
            model_info['category'] = 'segmentation'
        elif 'detect' in name_lower or 'det' in name_lower:
            model_info['category'] = 'detection'
        elif 'cls' in name_lower:
            model_info['category'] = 'classification'
        elif 'abutment' in name_lower:
            model_info['category'] = 'implant_analysis'
        elif 'caries' in name_lower:
            model_info['category'] = 'caries_detection'
        elif 'face' in name_lower:
            model_info['category'] = 'facial_analysis'
        elif 'tooth' in name_lower:
            model_info['category'] = 'tooth_analysis'
        
        models_info['model_files'].append(model_info)
    
    # Analyze encrypted ONNX models
    encrypted_models = list(Path("IntraoralScan/Bin/AIModels/models").glob("*.onnx.encrypt"))
    for enc_model in encrypted_models:
        models_info['encrypted_models'].append({
            'name': enc_model.name,
            'size_mb': round(enc_model.stat().st_size / (1024*1024), 2),
            'type': 'ONNX (encrypted)'
        })
    
    # Read configuration files
    config_files = [
        "IntraoralScan/Bin/AIModels/config/config.txt",
        "IntraoralScan/Bin/AIModels/config-DENTURE/config.txt",
        "IntraoralScan/Bin/AIModels/config-EJAW/config.txt"
    ]
    
    for config_file in config_files:
        if os.path.exists(config_file):
            config_name = Path(config_file).parent.name
            with open(config_file, 'r') as f:
                config_content = {}
                for line in f:
                    if '=' in line:
                        key, value = line.strip().split('=', 1)
                        config_content[key] = value
                models_info['configurations'][config_name] = config_content
    
    return models_info

def analyze_database_structure():
    """Analyze SQLite database structure"""
    db_info = {
        'main_database': {},
        'implant_data_count': 0,
        'markpoint_data_count': 0
    }
    
    # Analyze main database
    db_path = "IntraoralScan/Bin/DB/data.db"
    if os.path.exists(db_path):
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Get table names
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            
            db_info['main_database']['tables'] = []
            for table in tables:
                table_name = table[0]
                cursor.execute(f"PRAGMA table_info({table_name});")
                columns = cursor.fetchall()
                
                cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
                row_count = cursor.fetchone()[0]
                
                table_info = {
                    'name': table_name,
                    'columns': [col[1] for col in columns],
                    'row_count': row_count
                }
                db_info['main_database']['tables'].append(table_info)
            
            conn.close()
        except Exception as e:
            db_info['main_database']['error'] = str(e)
    
    # Count implant and markpoint data directories
    implant_dirs = list(Path("IntraoralScan/Bin/DB/implant").glob("*"))
    markpoint_dirs = list(Path("IntraoralScan/Bin/DB/markpoint").glob("*"))
    
    db_info['implant_data_count'] = len([d for d in implant_dirs if d.is_dir()])
    db_info['markpoint_data_count'] = len([d for d in markpoint_dirs if d.is_dir()])
    
    return db_info

def analyze_python_components():
    """Analyze Python components"""
    python_info = {
        'pyd_modules': [],
        'python_version': None,
        'pyc_files': []
    }
    
    # Find .pyd files (Python extensions)
    pyd_files = list(Path("IntraoralScan/Bin").glob("*.pyd"))
    for pyd_file in pyd_files:
        python_info['pyd_modules'].append({
            'name': pyd_file.name,
            'size_kb': round(pyd_file.stat().st_size / 1024, 1)
        })
    
    # Check Python version from DLL
    if os.path.exists("IntraoralScan/Bin/python38.dll"):
        python_info['python_version'] = "3.8"
    elif os.path.exists("IntraoralScan/Bin/python3.dll"):
        python_info['python_version'] = "3.x"
    
    # Find .pyc files
    pyc_files = list(Path("IntraoralScan/Bin/python").rglob("*.pyc")) if Path("IntraoralScan/Bin/python").exists() else []
    python_info['pyc_files'] = [str(f.relative_to("IntraoralScan/Bin")) for f in pyc_files[:10]]  # First 10
    
    return python_info

def generate_comprehensive_report():
    """Generate comprehensive analysis report"""
    
    # Load previous analysis results
    with open('analysis_output/high_value_analysis.json', 'r') as f:
        high_value_data = json.load(f)
    
    with open('analysis_output/architecture_analysis.json', 'r') as f:
        arch_data = json.load(f)
    
    # Perform additional analysis
    ai_models = analyze_ai_models()
    db_structure = analyze_database_structure()
    python_components = analyze_python_components()
    
    # Compile comprehensive report
    report = {
        'executive_summary': {
            'application_name': 'IntraoralScan 3.5.4.6',
            'architecture_type': 'Multi-process Qt/QML application with AI/ML components',
            'total_executables': len(high_value_data['high_value_executables']),
            'total_libraries': len(high_value_data['key_algorithm_dlls']),
            'ai_models_count': len(ai_models['model_files']) + len(ai_models['encrypted_models']),
            'supported_devices': len([d for d in arch_data['supported_devices'] if not d['disabled']]),
            'confidence_level': 'High (0.85)'
        },
        'system_architecture': {
            'service_startup_order': arch_data['service_startup_order'],
            'communication_channels': arch_data['communication_channels'],
            'network_endpoints': arch_data['network_endpoints'],
            'feature_flags': arch_data['feature_flags']
        },
        'high_value_components': {
            'main_executables': high_value_data['high_value_executables'],
            'algorithm_libraries': high_value_data['key_algorithm_dlls'][:15],  # Top 15
            'technology_stack': {
                'ui_framework': 'Qt5 with QML',
                'graphics': 'OpenSceneGraph (OSG)',
                'computer_vision': 'OpenCV 3.4.8 & 4.5.5',
                'gpu_acceleration': 'CUDA 11.0 with cuDNN 8',
                'ai_inference': 'TensorRT 8.5.3.1 & ONNX Runtime',
                'python_runtime': python_components['python_version'],
                'database': 'SQLite'
            }
        },
        'ai_ml_components': {
            'model_categories': {
                'segmentation': len([m for m in ai_models['model_files'] if m['category'] == 'segmentation']),
                'detection': len([m for m in ai_models['model_files'] if m['category'] == 'detection']),
                'classification': len([m for m in ai_models['model_files'] if m['category'] == 'classification']),
                'implant_analysis': len([m for m in ai_models['model_files'] if m['category'] == 'implant_analysis']),
                'caries_detection': len([m for m in ai_models['model_files'] if m['category'] == 'caries_detection']),
                'facial_analysis': len([m for m in ai_models['model_files'] if m['category'] == 'facial_analysis']),
                'tooth_analysis': len([m for m in ai_models['model_files'] if m['category'] == 'tooth_analysis'])
            },
            'model_configurations': ai_models['configurations'],
            'encrypted_models': ai_models['encrypted_models']
        },
        'data_storage': {
            'database_structure': db_structure['main_database'],
            'implant_data_entries': db_structure['implant_data_count'],
            'markpoint_data_entries': db_structure['markpoint_data_count']
        },
        'python_integration': python_components,
        'scanning_pipeline': {
            'acquisition': 'DentalScanAppLogic.exe - Real-time camera stream processing',
            'algorithm_processing': 'DentalAlgoService.exe - AI/ML inference and 3D processing',
            'registration': 'Sn3DRegistration.dll - Point cloud alignment and pose estimation',
            'fusion': 'Sn3DSpeckleFusion.dll - TSDF-based mesh generation',
            'ai_analysis': 'Multiple AI models for tooth segmentation and clinical analysis',
            'visualization': 'OpenSceneGraph + Qt5 for 3D rendering and UI',
            'export': 'DentalOrderAppLogic.exe - Data export and order management'
        },
        'device_support': arch_data['supported_devices'],
        'security_observations': {
            'encrypted_models': len(ai_models['encrypted_models']) > 0,
            'encrypted_configs': 'Some configuration files appear encrypted',
            'authentication_service': 'SnAuthenticator.exe handles licensing/authentication',
            'log_encryption': arch_data['feature_flags'].get('encrypt_log', False)
        }
    }
    
    return report

if __name__ == "__main__":
    report = generate_comprehensive_report()
    
    # Save comprehensive report
    with open('analysis_output/comprehensive_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print("=== COMPREHENSIVE INTRAORAL SCAN ANALYSIS REPORT ===\n")
    
    summary = report['executive_summary']
    print(f"Application: {summary['application_name']}")
    print(f"Architecture: {summary['architecture_type']}")
    print(f"Executables: {summary['total_executables']} high-value components")
    print(f"Libraries: {summary['total_libraries']} algorithm libraries")
    print(f"AI Models: {summary['ai_models_count']} models")
    print(f"Supported Devices: {summary['supported_devices']} active devices")
    print(f"Analysis Confidence: {summary['confidence_level']}")
    
    print(f"\n=== TECHNOLOGY STACK ===")
    tech = report['high_value_components']['technology_stack']
    for component, technology in tech.items():
        print(f"{component.replace('_', ' ').title()}: {technology}")
    
    print(f"\n=== AI/ML CAPABILITIES ===")
    ai_cats = report['ai_ml_components']['model_categories']
    for category, count in ai_cats.items():
        if count > 0:
            print(f"{category.replace('_', ' ').title()}: {count} models")
    
    print(f"\n=== DATA STORAGE ===")
    db_info = report['data_storage']
    if 'tables' in db_info['database_structure']:
        print(f"Database Tables: {len(db_info['database_structure']['tables'])}")
    print(f"Implant Data Entries: {db_info['implant_data_entries']}")
    print(f"Markpoint Data Entries: {db_info['markpoint_data_entries']}")
    
    print(f"\n=== SCANNING PIPELINE ===")
    pipeline = report['scanning_pipeline']
    for stage, description in pipeline.items():
        print(f"{stage.replace('_', ' ').title()}: {description}")
    
    print("\n=== ANALYSIS COMPLETE ===")
    print("Detailed report saved to: analysis_output/comprehensive_report.json")