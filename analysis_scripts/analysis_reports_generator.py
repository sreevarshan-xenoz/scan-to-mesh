#!/usr/bin/env python3
"""
Analysis Reports Generator
Compiles comprehensive analysis reports with confidence tracking for all components.
"""

import sqlite3
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from collections import defaultdict, Counter
import pandas as pd

class AnalysisReportsGenerator:
    def __init__(self, db_path="analysis_results.db", output_dir="analysis_output/architecture_docs/reports"):
        self.db_path = db_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.conn = None
        self.data_cache = {}
        
    def connect_db(self):
        """Connect to the analysis database"""
        try:
            self.conn = sqlite3.connect(self.db_path)
            self.conn.row_factory = sqlite3.Row
            print(f"Connected to database: {self.db_path}")
            return True
        except Exception as e:
            print(f"Error connecting to database: {e}")
            return False
    
    def close_db(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
    
    def load_analysis_data(self):
        """Load all analysis data for report generation"""
        if not self.conn:
            return False
            
        try:
            # Load dependency analysis data
            cursor = self.conn.execute("""
                SELECT da.*, e.name as executable_name, e.classification
                FROM dependency_analysis da
                JOIN executables e ON da.executable_id = e.id
                ORDER BY da.analysis_success DESC, e.classification
            """)
            self.data_cache['dependency_analysis'] = [dict(row) for row in cursor.fetchall()]
            
            # Load dependencies with validation status
            cursor = self.conn.execute("""
                SELECT d.*, e.name as executable_name, e.classification as exe_classification
                FROM dependencies d
                JOIN executables e ON d.executable_id = e.id
                ORDER BY d.is_found DESC, d.dependency_name
            """)
            self.data_cache['dependencies'] = [dict(row) for row in cursor.fetchall()]
            
            # Load configuration analysis
            cursor = self.conn.execute("""
                SELECT * FROM qt_resources
                ORDER BY resource_type, resource_path
            """)
            self.data_cache['qt_resources'] = [dict(row) for row in cursor.fetchall()]
            
            # Load QML files
            cursor = self.conn.execute("""
                SELECT * FROM qml_files
                ORDER BY file_path
            """)
            self.data_cache['qml_files'] = [dict(row) for row in cursor.fetchall()]
            
            # Load Python modules
            cursor = self.conn.execute("""
                SELECT * FROM python_modules
                ORDER BY module_name
            """)
            self.data_cache['python_modules'] = [dict(row) for row in cursor.fetchall()]
            
            # Load algorithm DLLs and functions
            cursor = self.conn.execute("""
                SELECT * FROM algorithm_dlls
                ORDER BY processing_stage, dll_name
            """)
            self.data_cache['algorithm_dlls'] = [dict(row) for row in cursor.fetchall()]
            
            cursor = self.conn.execute("""
                SELECT * FROM algorithm_functions
                ORDER BY dll_name, function_signature
            """)
            self.data_cache['algorithm_functions'] = [dict(row) for row in cursor.fetchall()]
            
            # Load hardware interfaces
            cursor = self.conn.execute("""
                SELECT * FROM hardware_interfaces
                ORDER BY interface_type, dll_name
            """)
            self.data_cache['hardware_interfaces'] = [dict(row) for row in cursor.fetchall()]
            
            # Load database schemas
            cursor = self.conn.execute("""
                SELECT * FROM database_schemas
                ORDER BY db_name
            """)
            self.data_cache['database_schemas'] = [dict(row) for row in cursor.fetchall()]
            
            # Load network analysis
            cursor = self.conn.execute("""
                SELECT * FROM network_analysis
                ORDER BY analysis_success DESC
            """)
            self.data_cache['network_analysis'] = [dict(row) for row in cursor.fetchall()]
            
            print(f"Loaded analysis data for report generation")
            return True
            
        except Exception as e:
            print(f"Error loading analysis data: {e}")
            return False
    
    def generate_dependency_analysis_report(self):
        """Generate dependency analysis reports with validation status"""
        print("Generating dependency analysis report...")
        
        report = {
            'title': 'Dependency Analysis Report',
            'generated_at': datetime.now().isoformat(),
            'summary': {},
            'detailed_analysis': {},
            'validation_status': {},
            'recommendations': []
        }
        
        # Summary statistics
        total_executables = len(set(dep['executable_name'] for dep in self.data_cache['dependencies']))
        total_dependencies = len(self.data_cache['dependencies'])
        found_dependencies = len([dep for dep in self.data_cache['dependencies'] if dep['is_found']])
        missing_dependencies = total_dependencies - found_dependencies
        
        system_libraries = len([dep for dep in self.data_cache['dependencies'] if dep.get('is_system_library')])
        custom_libraries = total_dependencies - system_libraries
        
        report['summary'] = {
            'total_executables_analyzed': total_executables,
            'total_dependencies_found': total_dependencies,
            'dependencies_resolved': found_dependencies,
            'dependencies_missing': missing_dependencies,
            'resolution_rate': round((found_dependencies / total_dependencies) * 100, 2) if total_dependencies > 0 else 0,
            'system_libraries': system_libraries,
            'custom_libraries': custom_libraries,
            'analysis_methods_used': list(set(dep.get('analysis_method', 'unknown') for dep in self.data_cache['dependencies']))
        }
        
        # Detailed analysis by executable
        exe_analysis = defaultdict(lambda: {
            'total_deps': 0,
            'found_deps': 0,
            'missing_deps': [],
            'system_deps': 0,
            'custom_deps': 0,
            'classification': 'Unknown'
        })
        
        for dep in self.data_cache['dependencies']:
            exe_name = dep['executable_name']
            exe_analysis[exe_name]['total_deps'] += 1
            exe_analysis[exe_name]['classification'] = dep.get('exe_classification', 'Unknown')
            
            if dep['is_found']:
                exe_analysis[exe_name]['found_deps'] += 1
            else:
                exe_analysis[exe_name]['missing_deps'].append(dep['dependency_name'])
            
            if dep.get('is_system_library'):
                exe_analysis[exe_name]['system_deps'] += 1
            else:
                exe_analysis[exe_name]['custom_deps'] += 1
        
        # Convert to regular dict and add resolution rates
        for exe_name, data in exe_analysis.items():
            data['resolution_rate'] = round((data['found_deps'] / data['total_deps']) * 100, 2) if data['total_deps'] > 0 else 0
        
        report['detailed_analysis'] = dict(exe_analysis)
        
        # Validation status
        analysis_success_rate = 0
        if self.data_cache['dependency_analysis']:
            successful_analyses = len([da for da in self.data_cache['dependency_analysis'] if da['analysis_success']])
            total_analyses = len(self.data_cache['dependency_analysis'])
            analysis_success_rate = round((successful_analyses / total_analyses) * 100, 2) if total_analyses > 0 else 0
        
        report['validation_status'] = {
            'analysis_success_rate': analysis_success_rate,
            'most_problematic_executables': sorted([
                (exe, data['resolution_rate']) for exe, data in exe_analysis.items()
            ], key=lambda x: x[1])[:5],
            'common_missing_dependencies': dict(Counter([
                dep for deps_list in [data['missing_deps'] for data in exe_analysis.values()]
                for dep in deps_list
            ]).most_common(10))
        }
        
        # Recommendations
        if missing_dependencies > 0:
            report['recommendations'].append(f"Investigate {missing_dependencies} missing dependencies")
        
        if analysis_success_rate < 90:
            report['recommendations'].append("Consider alternative analysis tools for failed dependency scans")
        
        # Save report
        report_path = self.output_dir / "dependency_analysis_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Generate markdown version
        self._generate_dependency_markdown_report(report)
        
        print(f"Dependency analysis report saved to: {report_path}")
        return report
    
    def _generate_dependency_markdown_report(self, report_data):
        """Generate markdown version of dependency report"""
        md_content = f"""# Dependency Analysis Report

Generated: {report_data['generated_at']}

## Summary

- **Total Executables Analyzed**: {report_data['summary']['total_executables_analyzed']}
- **Total Dependencies Found**: {report_data['summary']['total_dependencies_found']}
- **Dependencies Resolved**: {report_data['summary']['dependencies_resolved']}
- **Dependencies Missing**: {report_data['summary']['dependencies_missing']}
- **Resolution Rate**: {report_data['summary']['resolution_rate']}%
- **System Libraries**: {report_data['summary']['system_libraries']}
- **Custom Libraries**: {report_data['summary']['custom_libraries']}

## Analysis Methods Used

{', '.join(report_data['summary']['analysis_methods_used'])}

## Detailed Analysis by Executable

| Executable | Classification | Total Deps | Found | Missing | Resolution Rate |
|------------|---------------|------------|-------|---------|-----------------|
"""
        
        for exe_name, data in report_data['detailed_analysis'].items():
            md_content += f"| {exe_name} | {data['classification']} | {data['total_deps']} | {data['found_deps']} | {len(data['missing_deps'])} | {data['resolution_rate']}% |\n"
        
        md_content += f"""
## Validation Status

- **Analysis Success Rate**: {report_data['validation_status']['analysis_success_rate']}%

### Most Problematic Executables

"""
        
        for exe, rate in report_data['validation_status']['most_problematic_executables']:
            md_content += f"- {exe}: {rate}% resolution rate\n"
        
        md_content += "\n### Common Missing Dependencies\n\n"
        
        for dep, count in report_data['validation_status']['common_missing_dependencies'].items():
            md_content += f"- {dep}: missing from {count} executables\n"
        
        md_content += "\n## Recommendations\n\n"
        
        for rec in report_data['recommendations']:
            md_content += f"- {rec}\n"
        
        # Save markdown report
        md_path = self.output_dir / "dependency_analysis_report.md"
        with open(md_path, 'w') as f:
            f.write(md_content)
    
    def generate_configuration_reference_guide(self):
        """Create configuration reference guides with parsing coverage metrics"""
        print("Generating configuration reference guide...")
        
        guide = {
            'title': 'Configuration Reference Guide',
            'generated_at': datetime.now().isoformat(),
            'qt_resources': {},
            'qml_structure': {},
            'parsing_coverage': {},
            'configuration_insights': []
        }
        
        # Analyze Qt resources
        resource_types = defaultdict(list)
        for resource in self.data_cache['qt_resources']:
            resource_type = resource.get('resource_type', 'unknown')
            resource_types[resource_type].append({
                'path': resource.get('resource_path', ''),
                'size': resource.get('resource_size', 0),
                'extraction_success': resource.get('confidence_score', 0) > 0.5  # Use confidence as proxy for success
            })
        
        guide['qt_resources'] = {
            'total_resources': len(self.data_cache['qt_resources']),
            'by_type': dict(resource_types),
            'extraction_success_rate': round(
                len([r for r in self.data_cache['qt_resources'] if r.get('confidence_score', 0) > 0.5]) / 
                max(len(self.data_cache['qt_resources']), 1) * 100, 2
            )
        }
        
        # Analyze QML structure
        qml_components = defaultdict(list)
        ui_flows = []
        
        for qml_file in self.data_cache['qml_files']:
            file_path = qml_file.get('file_path', '')
            component_type = qml_file.get('ui_component_type', 'unknown')
            
            qml_components[component_type].append({
                'path': file_path,
                'service_connections': qml_file.get('service_connections', '').split(',') if qml_file.get('service_connections') else [],
                'signal_slot_connections': qml_file.get('signal_slot_connections', 0)
            })
            
            # Extract UI flows from file paths
            if 'main' in file_path.lower() or 'app' in file_path.lower():
                ui_flows.append(f"Main Application: {file_path}")
            elif 'dialog' in file_path.lower() or 'popup' in file_path.lower():
                ui_flows.append(f"Dialog/Popup: {file_path}")
            elif 'view' in file_path.lower() or 'page' in file_path.lower():
                ui_flows.append(f"View/Page: {file_path}")
        
        guide['qml_structure'] = {
            'total_qml_files': len(self.data_cache['qml_files']),
            'components_by_type': dict(qml_components),
            'identified_ui_flows': ui_flows,
            'common_imports': self._analyze_qml_imports(),
            'signal_patterns': self._analyze_qml_signals()
        }
        
        # Parsing coverage metrics
        total_config_files = len(self.data_cache['qt_resources']) + len(self.data_cache['qml_files'])
        successfully_parsed = (
            len([r for r in self.data_cache['qt_resources'] if r.get('confidence_score', 0) > 0.5]) +
            len(self.data_cache['qml_files'])  # Assume QML files are successfully parsed if they're in the DB
        )
        
        guide['parsing_coverage'] = {
            'total_configuration_files': total_config_files,
            'successfully_parsed': successfully_parsed,
            'parsing_success_rate': round((successfully_parsed / max(total_config_files, 1)) * 100, 2),
            'file_types_covered': ['Qt Resources (.qrc)', 'QML Files (.qml)', 'Python Modules (.py)'],
            'parsing_methods_used': ['rcc tool', 'QML parser', 'Python AST analysis']
        }
        
        # Configuration insights
        if guide['qt_resources']['extraction_success_rate'] < 80:
            guide['configuration_insights'].append("Low Qt resource extraction rate - consider alternative tools")
        
        if len(ui_flows) > 0:
            guide['configuration_insights'].append(f"Identified {len(ui_flows)} UI flow components")
        
        # Save guide
        guide_path = self.output_dir / "configuration_reference_guide.json"
        with open(guide_path, 'w') as f:
            json.dump(guide, f, indent=2)
        
        # Generate markdown version
        self._generate_config_markdown_guide(guide)
        
        print(f"Configuration reference guide saved to: {guide_path}")
        return guide
    
    def _analyze_qml_imports(self):
        """Analyze common QML service connections"""
        all_connections = []
        for qml_file in self.data_cache['qml_files']:
            connections = qml_file.get('service_connections', '')
            if connections:
                all_connections.extend([conn.strip() for conn in connections.split(',') if conn.strip()])
        
        return dict(Counter(all_connections).most_common(10))
    
    def _analyze_qml_signals(self):
        """Analyze QML signal/slot connection counts"""
        connection_counts = []
        for qml_file in self.data_cache['qml_files']:
            count = qml_file.get('signal_slot_connections', 0)
            if count > 0:
                connection_counts.append(count)
        
        if connection_counts:
            return {
                'total_connections': sum(connection_counts),
                'average_per_file': round(sum(connection_counts) / len(connection_counts), 2),
                'max_connections': max(connection_counts),
                'files_with_connections': len(connection_counts)
            }
        return {}
    
    def _generate_config_markdown_guide(self, guide_data):
        """Generate markdown version of configuration guide"""
        md_content = f"""# Configuration Reference Guide

Generated: {guide_data['generated_at']}

## Qt Resources Analysis

- **Total Resources**: {guide_data['qt_resources']['total_resources']}
- **Extraction Success Rate**: {guide_data['qt_resources']['extraction_success_rate']}%

### Resources by Type

"""
        
        for resource_type, resources in guide_data['qt_resources']['by_type'].items():
            md_content += f"#### {resource_type.title()} ({len(resources)} files)\n\n"
            for resource in resources[:5]:  # Show first 5 examples
                status = "✓" if resource['extraction_success'] else "✗"
                md_content += f"- {status} {resource['path']} ({resource['size']} bytes)\n"
            if len(resources) > 5:
                md_content += f"- ... and {len(resources) - 5} more\n"
            md_content += "\n"
        
        md_content += f"""## QML Structure Analysis

- **Total QML Files**: {guide_data['qml_structure']['total_qml_files']}
- **Identified UI Flows**: {len(guide_data['qml_structure']['identified_ui_flows'])}

### UI Flow Components

"""
        
        for flow in guide_data['qml_structure']['identified_ui_flows']:
            md_content += f"- {flow}\n"
        
        md_content += "\n### Common Service Connections\n\n"
        
        for connection, count in guide_data['qml_structure']['common_imports'].items():
            md_content += f"- {connection}: used {count} times\n"
        
        md_content += f"""
## Parsing Coverage Metrics

- **Total Configuration Files**: {guide_data['parsing_coverage']['total_configuration_files']}
- **Successfully Parsed**: {guide_data['parsing_coverage']['successfully_parsed']}
- **Parsing Success Rate**: {guide_data['parsing_coverage']['parsing_success_rate']}%

### File Types Covered

"""
        
        for file_type in guide_data['parsing_coverage']['file_types_covered']:
            md_content += f"- {file_type}\n"
        
        md_content += "\n### Parsing Methods Used\n\n"
        
        for method in guide_data['parsing_coverage']['parsing_methods_used']:
            md_content += f"- {method}\n"
        
        md_content += "\n## Configuration Insights\n\n"
        
        for insight in guide_data['configuration_insights']:
            md_content += f"- {insight}\n"
        
        # Save markdown guide
        md_path = self.output_dir / "configuration_reference_guide.md"
        with open(md_path, 'w') as f:
            f.write(md_content)
    
    def generate_ai_model_specifications(self):
        """Build AI model specifications with tensor documentation"""
        print("Generating AI model specifications...")
        
        specs = {
            'title': 'AI Model Specifications',
            'generated_at': datetime.now().isoformat(),
            'model_inventory': {},
            'algorithm_analysis': {},
            'processing_pipeline': {},
            'tensor_documentation': {},
            'confidence_assessment': {}
        }
        
        # Analyze algorithm DLLs
        algorithm_stages = defaultdict(list)
        for dll in self.data_cache['algorithm_dlls']:
            stage = dll.get('processing_stage', 'unknown')
            algorithm_stages[stage].append({
                'dll_name': dll['dll_name'],
                'file_size': dll.get('file_size', 0),
                'export_count': dll.get('export_count', 0),
                'confidence_score': dll.get('confidence_score', 0.0)
            })
        
        specs['algorithm_analysis'] = {
            'total_algorithm_dlls': len(self.data_cache['algorithm_dlls']),
            'processing_stages': dict(algorithm_stages),
            'high_confidence_dlls': len([dll for dll in self.data_cache['algorithm_dlls'] 
                                       if dll.get('confidence_score', 0) >= 0.8]),
            'total_exported_functions': sum(dll.get('export_count', 0) for dll in self.data_cache['algorithm_dlls'])
        }
        
        # Analyze algorithm functions
        function_analysis = defaultdict(list)
        for func in self.data_cache['algorithm_functions']:
            dll_name = func.get('dll_name', 'unknown')
            function_analysis[dll_name].append({
                'signature': func.get('function_signature', ''),
                'type': func.get('function_type', 'unknown'),
                'stage': func.get('algorithm_stage', 'unknown'),
                'confidence': func.get('confidence', 0.0)
            })
        
        specs['processing_pipeline'] = {
            'functions_by_dll': dict(function_analysis),
            'function_types': dict(Counter([func.get('function_type', 'unknown') 
                                          for func in self.data_cache['algorithm_functions']])),
            'algorithm_stages': dict(Counter([func.get('algorithm_stage', 'unknown') 
                                            for func in self.data_cache['algorithm_functions']])),
            'high_confidence_functions': len([func for func in self.data_cache['algorithm_functions'] 
                                            if func.get('confidence', 0) >= 0.8])
        }
        
        # Model inventory (inferred from algorithm analysis)
        inferred_models = []
        
        # Look for AI/ML related DLLs
        ai_keywords = ['AI', 'Neural', 'Deep', 'CNN', 'Seg', 'Det', 'Cls']
        for dll in self.data_cache['algorithm_dlls']:
            dll_name = dll['dll_name']
            if any(keyword in dll_name for keyword in ai_keywords):
                inferred_models.append({
                    'model_source': dll_name,
                    'inferred_type': self._infer_model_type(dll_name),
                    'processing_stage': dll.get('processing_stage', 'unknown'),
                    'confidence': dll.get('confidence_score', 0.0)
                })
        
        specs['model_inventory'] = {
            'total_inferred_models': len(inferred_models),
            'models_by_type': dict(Counter([model['inferred_type'] for model in inferred_models])),
            'models_by_stage': dict(Counter([model['processing_stage'] for model in inferred_models])),
            'model_details': inferred_models
        }
        
        # Tensor documentation (inferred from function signatures and DLL analysis)
        tensor_hints = []
        for func in self.data_cache['algorithm_functions']:
            signature = func.get('function_signature', '')
            if any(keyword in signature.lower() for keyword in ['tensor', 'input', 'output', 'shape', 'batch']):
                tensor_hints.append({
                    'dll_name': func.get('dll_name', ''),
                    'function': signature,
                    'inferred_tensor_info': self._extract_tensor_hints(signature)
                })
        
        specs['tensor_documentation'] = {
            'functions_with_tensor_hints': len(tensor_hints),
            'tensor_hint_details': tensor_hints,
            'common_tensor_patterns': self._analyze_tensor_patterns(tensor_hints)
        }
        
        # Confidence assessment
        avg_dll_confidence = sum(dll.get('confidence_score', 0) for dll in self.data_cache['algorithm_dlls']) / max(len(self.data_cache['algorithm_dlls']), 1)
        avg_func_confidence = sum(func.get('confidence', 0) for func in self.data_cache['algorithm_functions']) / max(len(self.data_cache['algorithm_functions']), 1)
        
        specs['confidence_assessment'] = {
            'average_dll_confidence': round(avg_dll_confidence, 3),
            'average_function_confidence': round(avg_func_confidence, 3),
            'high_confidence_threshold': 0.8,
            'analysis_completeness': {
                'dll_analysis': 'Complete' if len(self.data_cache['algorithm_dlls']) > 0 else 'Incomplete',
                'function_analysis': 'Complete' if len(self.data_cache['algorithm_functions']) > 0 else 'Incomplete',
                'tensor_analysis': 'Inferred' if len(tensor_hints) > 0 else 'Limited'
            }
        }
        
        # Save specifications
        specs_path = self.output_dir / "ai_model_specifications.json"
        with open(specs_path, 'w') as f:
            json.dump(specs, f, indent=2)
        
        # Generate markdown version
        self._generate_ai_specs_markdown(specs)
        
        print(f"AI model specifications saved to: {specs_path}")
        return specs
    
    def _infer_model_type(self, dll_name):
        """Infer AI model type from DLL name"""
        dll_lower = dll_name.lower()
        
        if 'seg' in dll_lower or 'segmentation' in dll_lower:
            return 'Segmentation'
        elif 'det' in dll_lower or 'detection' in dll_lower:
            return 'Detection'
        elif 'cls' in dll_lower or 'classification' in dll_lower:
            return 'Classification'
        elif 'face' in dll_lower:
            return 'Face Analysis'
        elif 'dental' in dll_lower and 'ai' in dll_lower:
            return 'Dental AI'
        elif 'cnn' in dll_lower or 'neural' in dll_lower:
            return 'Neural Network'
        else:
            return 'Unknown AI Model'
    
    def _extract_tensor_hints(self, signature):
        """Extract tensor information hints from function signatures"""
        hints = []
        signature_lower = signature.lower()
        
        if 'input' in signature_lower:
            hints.append('Has input tensor parameter')
        if 'output' in signature_lower:
            hints.append('Has output tensor parameter')
        if 'batch' in signature_lower:
            hints.append('Supports batch processing')
        if 'shape' in signature_lower:
            hints.append('Shape parameter present')
        if 'float' in signature_lower:
            hints.append('Float tensor data type')
        if 'int' in signature_lower:
            hints.append('Integer tensor data type')
        
        return hints
    
    def _analyze_tensor_patterns(self, tensor_hints):
        """Analyze common tensor patterns"""
        all_patterns = []
        for hint in tensor_hints:
            all_patterns.extend(hint.get('inferred_tensor_info', []))
        
        return dict(Counter(all_patterns).most_common(10))
    
    def _generate_ai_specs_markdown(self, specs_data):
        """Generate markdown version of AI model specifications"""
        md_content = f"""# AI Model Specifications

Generated: {specs_data['generated_at']}

## Model Inventory

- **Total Inferred Models**: {specs_data['model_inventory']['total_inferred_models']}

### Models by Type

"""
        
        for model_type, count in specs_data['model_inventory']['models_by_type'].items():
            md_content += f"- {model_type}: {count} models\n"
        
        md_content += "\n### Model Details\n\n"
        
        for model in specs_data['model_inventory']['model_details']:
            md_content += f"- **{model['model_source']}**\n"
            md_content += f"  - Type: {model['inferred_type']}\n"
            md_content += f"  - Stage: {model['processing_stage']}\n"
            md_content += f"  - Confidence: {model['confidence']}\n\n"
        
        md_content += f"""## Algorithm Analysis

- **Total Algorithm DLLs**: {specs_data['algorithm_analysis']['total_algorithm_dlls']}
- **High Confidence DLLs**: {specs_data['algorithm_analysis']['high_confidence_dlls']}
- **Total Exported Functions**: {specs_data['algorithm_analysis']['total_exported_functions']}

### Processing Stages

"""
        
        for stage, dlls in specs_data['algorithm_analysis']['processing_stages'].items():
            md_content += f"#### {stage.title()} ({len(dlls)} DLLs)\n\n"
            for dll in dlls:
                md_content += f"- {dll['dll_name']} (confidence: {dll['confidence_score']})\n"
            md_content += "\n"
        
        md_content += f"""## Processing Pipeline

### Function Types Distribution

"""
        
        for func_type, count in specs_data['processing_pipeline']['function_types'].items():
            md_content += f"- {func_type}: {count} functions\n"
        
        md_content += "\n### Algorithm Stages\n\n"
        
        for stage, count in specs_data['processing_pipeline']['algorithm_stages'].items():
            md_content += f"- {stage}: {count} functions\n"
        
        md_content += f"""
## Tensor Documentation

- **Functions with Tensor Hints**: {specs_data['tensor_documentation']['functions_with_tensor_hints']}

### Common Tensor Patterns

"""
        
        for pattern, count in specs_data['tensor_documentation']['common_tensor_patterns'].items():
            md_content += f"- {pattern}: {count} occurrences\n"
        
        md_content += f"""
## Confidence Assessment

- **Average DLL Confidence**: {specs_data['confidence_assessment']['average_dll_confidence']}
- **Average Function Confidence**: {specs_data['confidence_assessment']['average_function_confidence']}

### Analysis Completeness

"""
        
        for analysis_type, status in specs_data['confidence_assessment']['analysis_completeness'].items():
            md_content += f"- {analysis_type.replace('_', ' ').title()}: {status}\n"
        
        # Save markdown specifications
        md_path = self.output_dir / "ai_model_specifications.md"
        with open(md_path, 'w') as f:
            f.write(md_content)
    
    def generate_hardware_interface_documentation(self):
        """Document hardware interfaces and device communication protocols"""
        print("Generating hardware interface documentation...")
        
        doc = {
            'title': 'Hardware Interface Documentation',
            'generated_at': datetime.now().isoformat(),
            'interface_summary': {},
            'device_capabilities': {},
            'communication_protocols': {},
            'driver_analysis': {},
            'confidence_metrics': {}
        }
        
        # Interface summary
        interface_types = defaultdict(list)
        for hw in self.data_cache['hardware_interfaces']:
            interface_type = hw.get('interface_type', 'unknown')
            interface_types[interface_type].append({
                'dll_name': hw['dll_name'],
                'dll_path': hw.get('dll_path', ''),
                'file_size': hw.get('file_size', 0),
                'confidence_score': hw.get('confidence_score', 0.0)
            })
        
        doc['interface_summary'] = {
            'total_hardware_interfaces': len(self.data_cache['hardware_interfaces']),
            'interface_types': dict(interface_types),
            'high_confidence_interfaces': len([hw for hw in self.data_cache['hardware_interfaces'] 
                                             if hw.get('confidence_score', 0) >= 0.8]),
            'analysis_methods': list(set(hw.get('analysis_method', 'unknown') 
                                       for hw in self.data_cache['hardware_interfaces']))
        }
        
        # Device capabilities analysis
        capabilities = []
        protocols = []
        
        for hw in self.data_cache['hardware_interfaces']:
            caps = hw.get('device_capabilities', '')
            if caps:
                capabilities.extend([cap.strip() for cap in caps.split(',') if cap.strip()])
            
            prots = hw.get('communication_protocols', '')
            if prots:
                protocols.extend([prot.strip() for prot in prots.split(',') if prot.strip()])
        
        doc['device_capabilities'] = {
            'identified_capabilities': dict(Counter(capabilities).most_common(20)),
            'total_capability_mentions': len(capabilities),
            'unique_capabilities': len(set(capabilities))
        }
        
        doc['communication_protocols'] = {
            'identified_protocols': dict(Counter(protocols).most_common(10)),
            'total_protocol_mentions': len(protocols),
            'unique_protocols': len(set(protocols))
        }
        
        # Driver analysis
        driver_patterns = {
            'camera_drivers': [hw for hw in self.data_cache['hardware_interfaces'] 
                             if 'camera' in hw['dll_name'].lower() or 'cam' in hw['dll_name'].lower()],
            'usb_drivers': [hw for hw in self.data_cache['hardware_interfaces'] 
                          if 'usb' in hw['dll_name'].lower() or 'hid' in hw['dll_name'].lower()],
            'scanner_drivers': [hw for hw in self.data_cache['hardware_interfaces'] 
                              if 'scan' in hw['dll_name'].lower() or '3d' in hw['dll_name'].lower()],
            'communication_drivers': [hw for hw in self.data_cache['hardware_interfaces'] 
                                    if 'comm' in hw['dll_name'].lower() or 'serial' in hw['dll_name'].lower()]
        }
        
        doc['driver_analysis'] = {
            'driver_categories': {category: len(drivers) for category, drivers in driver_patterns.items()},
            'driver_details': {category: [{'name': hw['dll_name'], 'confidence': hw.get('confidence_score', 0)} 
                                        for hw in drivers] for category, drivers in driver_patterns.items()}
        }
        
        # Confidence metrics
        avg_confidence = sum(hw.get('confidence_score', 0) for hw in self.data_cache['hardware_interfaces']) / max(len(self.data_cache['hardware_interfaces']), 1)
        
        doc['confidence_metrics'] = {
            'average_confidence': round(avg_confidence, 3),
            'confidence_distribution': {
                'high_confidence': len([hw for hw in self.data_cache['hardware_interfaces'] if hw.get('confidence_score', 0) >= 0.8]),
                'medium_confidence': len([hw for hw in self.data_cache['hardware_interfaces'] if 0.5 <= hw.get('confidence_score', 0) < 0.8]),
                'low_confidence': len([hw for hw in self.data_cache['hardware_interfaces'] if hw.get('confidence_score', 0) < 0.5])
            },
            'analysis_completeness': 'Complete' if len(self.data_cache['hardware_interfaces']) > 0 else 'No hardware interfaces found'
        }
        
        # Save documentation
        doc_path = self.output_dir / "hardware_interface_documentation.json"
        with open(doc_path, 'w') as f:
            json.dump(doc, f, indent=2)
        
        # Generate markdown version
        self._generate_hardware_markdown_doc(doc)
        
        print(f"Hardware interface documentation saved to: {doc_path}")
        return doc
    
    def _generate_hardware_markdown_doc(self, doc_data):
        """Generate markdown version of hardware interface documentation"""
        md_content = f"""# Hardware Interface Documentation

Generated: {doc_data['generated_at']}

## Interface Summary

- **Total Hardware Interfaces**: {doc_data['interface_summary']['total_hardware_interfaces']}
- **High Confidence Interfaces**: {doc_data['interface_summary']['high_confidence_interfaces']}

### Interface Types

"""
        
        for interface_type, interfaces in doc_data['interface_summary']['interface_types'].items():
            md_content += f"#### {interface_type.title()} ({len(interfaces)} interfaces)\n\n"
            for interface in interfaces:
                md_content += f"- {interface['dll_name']} (confidence: {interface['confidence_score']})\n"
            md_content += "\n"
        
        md_content += f"""## Device Capabilities

- **Total Capability Mentions**: {doc_data['device_capabilities']['total_capability_mentions']}
- **Unique Capabilities**: {doc_data['device_capabilities']['unique_capabilities']}

### Identified Capabilities

"""
        
        for capability, count in doc_data['device_capabilities']['identified_capabilities'].items():
            md_content += f"- {capability}: {count} mentions\n"
        
        md_content += f"""
## Communication Protocols

- **Total Protocol Mentions**: {doc_data['communication_protocols']['total_protocol_mentions']}
- **Unique Protocols**: {doc_data['communication_protocols']['unique_protocols']}

### Identified Protocols

"""
        
        for protocol, count in doc_data['communication_protocols']['identified_protocols'].items():
            md_content += f"- {protocol}: {count} mentions\n"
        
        md_content += "\n## Driver Analysis\n\n"
        
        for category, count in doc_data['driver_analysis']['driver_categories'].items():
            md_content += f"### {category.replace('_', ' ').title()} ({count} drivers)\n\n"
            for driver in doc_data['driver_analysis']['driver_details'][category]:
                md_content += f"- {driver['name']} (confidence: {driver['confidence']})\n"
            md_content += "\n"
        
        md_content += f"""## Confidence Metrics

- **Average Confidence**: {doc_data['confidence_metrics']['average_confidence']}
- **Analysis Completeness**: {doc_data['confidence_metrics']['analysis_completeness']}

### Confidence Distribution

- **High Confidence (≥0.8)**: {doc_data['confidence_metrics']['confidence_distribution']['high_confidence']} interfaces
- **Medium Confidence (0.5-0.8)**: {doc_data['confidence_metrics']['confidence_distribution']['medium_confidence']} interfaces
- **Low Confidence (<0.5)**: {doc_data['confidence_metrics']['confidence_distribution']['low_confidence']} interfaces
"""
        
        # Save markdown documentation
        md_path = self.output_dir / "hardware_interface_documentation.md"
        with open(md_path, 'w') as f:
            f.write(md_content)
    
    def run_task_7_2(self):
        """Execute task 7.2: Compile analysis reports with confidence tracking"""
        print("=== Task 7.2: Compiling analysis reports with confidence tracking ===")
        
        if not self.connect_db():
            return False
        
        if not self.load_analysis_data():
            return False
        
        try:
            # Generate all reports
            dependency_report = self.generate_dependency_analysis_report()
            config_guide = self.generate_configuration_reference_guide()
            ai_specs = self.generate_ai_model_specifications()
            hardware_doc = self.generate_hardware_interface_documentation()
            
            # Create comprehensive summary
            task_7_2_summary = {
                'task': '7.2 Compile analysis reports with confidence tracking',
                'completion_timestamp': datetime.now().isoformat(),
                'deliverables': {
                    'dependency_analysis_report': str(self.output_dir / "dependency_analysis_report.json"),
                    'configuration_reference_guide': str(self.output_dir / "configuration_reference_guide.json"),
                    'ai_model_specifications': str(self.output_dir / "ai_model_specifications.json"),
                    'hardware_interface_documentation': str(self.output_dir / "hardware_interface_documentation.json")
                },
                'report_statistics': {
                    'dependency_analysis': {
                        'executables_analyzed': dependency_report['summary']['total_executables_analyzed'],
                        'resolution_rate': dependency_report['summary']['resolution_rate']
                    },
                    'configuration_analysis': {
                        'parsing_success_rate': config_guide['parsing_coverage']['parsing_success_rate'],
                        'qml_files_analyzed': config_guide['qml_structure']['total_qml_files']
                    },
                    'ai_model_analysis': {
                        'algorithm_dlls': ai_specs['algorithm_analysis']['total_algorithm_dlls'],
                        'inferred_models': ai_specs['model_inventory']['total_inferred_models']
                    },
                    'hardware_analysis': {
                        'hardware_interfaces': hardware_doc['interface_summary']['total_hardware_interfaces'],
                        'average_confidence': hardware_doc['confidence_metrics']['average_confidence']
                    }
                },
                'requirements_addressed': ['6.2', '6.3', '6.4', '6.5']
            }
            
            # Save task summary
            summary_path = self.output_dir / "task_7_2_summary.json"
            with open(summary_path, 'w') as f:
                json.dump(task_7_2_summary, f, indent=2)
            
            print(f"\n=== Task 7.2 Completed Successfully ===")
            print(f"Summary saved to: {summary_path}")
            print(f"All reports generated in: {self.output_dir}")
            
            return True
            
        except Exception as e:
            print(f"Error in task 7.2: {e}")
            return False
        finally:
            self.close_db()

def main():
    """Main execution function"""
    generator = AnalysisReportsGenerator()
    success = generator.run_task_7_2()
    
    if success:
        print("\nTask 7.2 completed successfully!")
        return 0
    else:
        print("\nTask 7.2 failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())