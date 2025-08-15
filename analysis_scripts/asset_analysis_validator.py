#!/usr/bin/env python3
"""
Asset Analysis Validation Checkpoint for IntraoralScan Analysis
Implements task 4.4: Asset analysis validation checkpoint
"""
import os
import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Tuple, Any
import subprocess

class AssetAnalysisValidator:
    def __init__(self, db_path="analysis_results.db"):
        self.db_path = db_path
        self.validation_results = {}
        
    def validate_configuration_parsing(self) -> Dict:
        """Verify configuration parsing coverage rate"""
        validation = {
            'total_config_files': 0,
            'successfully_parsed': 0,
            'parsing_coverage_rate': 0.0,
            'failed_files': [],
            'parsing_methods': {},
            'confidence_scores': []
        }
        
        # Check if config analyzer results exist
        config_report_path = "analysis_output/config_analysis.json"
        if os.path.exists(config_report_path):
            try:
                with open(config_report_path, 'r') as f:
                    config_data = json.load(f)
                
                validation['total_config_files'] = config_data.get('summary', {}).get('total_files', 0)
                validation['successfully_parsed'] = config_data.get('summary', {}).get('successfully_parsed', 0)
                
                if validation['total_config_files'] > 0:
                    validation['parsing_coverage_rate'] = validation['successfully_parsed'] / validation['total_config_files']
                
                # Get parsing methods
                for file_info in config_data.get('parsed_files', []):
                    method = file_info.get('parser_used', 'unknown')
                    validation['parsing_methods'][method] = validation['parsing_methods'].get(method, 0) + 1
                    validation['confidence_scores'].append(file_info.get('confidence', 0.0))
                
                # Get failed files
                validation['failed_files'] = [
                    f['file_path'] for f in config_data.get('failed_files', [])
                ]
                
            except Exception as e:
                validation['error'] = f"Failed to load config analysis: {str(e)}"
        else:
            # Fallback: scan config directories manually
            config_dirs = [
                "IntraoralScan/Bin/config",
                "IntraoralScan/Bin/AIModels/config",
                "IntraoralScan/Bin/AIModels/config-DENTURE",
                "IntraoralScan/Bin/AIModels/config-EJAW"
            ]
            
            total_files = 0
            for config_dir in config_dirs:
                if os.path.exists(config_dir):
                    for ext in ['*.json', '*.ini', '*.xml', '*.yaml', '*.yml', '*.txt']:
                        total_files += len(list(Path(config_dir).rglob(ext)))
            
            validation['total_config_files'] = total_files
            validation['parsing_coverage_rate'] = 0.0  # No parsing done
            validation['note'] = "Config analysis not found - manual count performed"
        
        return validation
    
    def validate_qml_extraction(self) -> Dict:
        """Verify QML and Qt resource extraction completeness"""
        validation = {
            'qml_files_found': 0,
            'qt_resources_found': 0,
            'ui_components_identified': 0,
            'extraction_methods': {},
            'average_confidence': 0.0,
            'coverage_assessment': 'unknown'
        }
        
        # Check QML analysis results
        qml_report_path = "analysis_output/qml_qt_analysis.json"
        if os.path.exists(qml_report_path):
            try:
                with open(qml_report_path, 'r') as f:
                    qml_data = json.load(f)
                
                summary = qml_data.get('summary', {})
                validation['qml_files_found'] = summary.get('qml_files_found', 0)
                validation['qt_resources_found'] = summary.get('qt_resources_found', 0)
                validation['ui_components_identified'] = summary.get('ui_components_identified', 0)
                validation['average_confidence'] = (
                    summary.get('average_qml_confidence', 0.0) + 
                    summary.get('average_resource_confidence', 0.0)
                ) / 2
                
                # Get extraction methods
                validation['extraction_methods'] = qml_data.get('extraction_methods', {})
                
                # Assess coverage
                if validation['qml_files_found'] > 100:
                    validation['coverage_assessment'] = 'excellent'
                elif validation['qml_files_found'] > 50:
                    validation['coverage_assessment'] = 'good'
                elif validation['qml_files_found'] > 10:
                    validation['coverage_assessment'] = 'moderate'
                else:
                    validation['coverage_assessment'] = 'limited'
                    
            except Exception as e:
                validation['error'] = f"Failed to load QML analysis: {str(e)}"
        else:
            validation['coverage_assessment'] = 'not_performed'
            validation['note'] = "QML analysis not found"
        
        return validation
    
    def validate_python_analysis(self) -> Dict:
        """Verify Python module analysis and service mappings"""
        validation = {
            'python_modules_found': 0,
            'pyd_modules_found': 0,
            'decompilation_success_rate': 0.0,
            'service_mappings_found': 0,
            'python_version_detected': 'unknown',
            'analysis_methods': {},
            'confidence_assessment': 'unknown'
        }
        
        # Check Python analysis results
        python_report_path = "analysis_output/python_runtime_analysis.json"
        if os.path.exists(python_report_path):
            try:
                with open(python_report_path, 'r') as f:
                    python_data = json.load(f)
                
                summary = python_data.get('summary', {})
                validation['python_modules_found'] = summary.get('python_modules_found', 0)
                validation['pyd_modules_found'] = summary.get('pyd_modules_found', 0)
                validation['decompilation_success_rate'] = summary.get('decompilation_success_rate', 0.0)
                validation['service_mappings_found'] = summary.get('service_mappings_found', 0)
                validation['python_version_detected'] = summary.get('python_version', 'unknown')
                
                # Get analysis methods
                validation['analysis_methods'] = python_data.get('decompilation_methods', {})
                
                # Assess confidence
                avg_python_conf = summary.get('average_python_confidence', 0.0)
                avg_pyd_conf = summary.get('average_pyd_confidence', 0.0)
                overall_confidence = (avg_python_conf + avg_pyd_conf) / 2
                
                if overall_confidence > 0.8:
                    validation['confidence_assessment'] = 'high'
                elif overall_confidence > 0.6:
                    validation['confidence_assessment'] = 'medium'
                else:
                    validation['confidence_assessment'] = 'low'
                    
            except Exception as e:
                validation['error'] = f"Failed to load Python analysis: {str(e)}"
        else:
            validation['confidence_assessment'] = 'not_performed'
            validation['note'] = "Python analysis not found"
        
        return validation
    
    def cross_validate_dependencies(self) -> Dict:
        """Cross-validate Python module mappings with process dependencies"""
        validation = {
            'dependency_consistency': 'unknown',
            'python_process_mappings': [],
            'inconsistencies_found': [],
            'validation_confidence': 0.0
        }
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check if we have both dependency and Python analysis data
            cursor.execute('SELECT COUNT(*) FROM python_modules')
            python_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='dependencies'")
            has_dependencies = cursor.fetchone() is not None
            
            if python_count > 0 and has_dependencies:
                # Get Python modules that might be related to processes
                cursor.execute('''
                    SELECT module_name, imports, service_mappings 
                    FROM python_modules 
                    WHERE service_mappings != '[]' AND service_mappings IS NOT NULL
                ''')
                
                python_mappings = []
                for row in cursor.fetchall():
                    module_name = row[0]
                    imports = json.loads(row[1]) if row[1] else []
                    service_mappings = json.loads(row[2]) if row[2] else []
                    
                    python_mappings.append({
                        'module': module_name,
                        'imports': imports,
                        'service_mappings': service_mappings
                    })
                
                validation['python_process_mappings'] = python_mappings
                
                # Simple consistency check
                if len(python_mappings) > 0:
                    validation['dependency_consistency'] = 'partial'
                    validation['validation_confidence'] = 0.6
                else:
                    validation['dependency_consistency'] = 'limited'
                    validation['validation_confidence'] = 0.3
            else:
                validation['dependency_consistency'] = 'insufficient_data'
                validation['validation_confidence'] = 0.1
            
            conn.close()
            
        except Exception as e:
            validation['error'] = f"Database validation failed: {str(e)}"
            validation['validation_confidence'] = 0.0
        
        return validation
    
    def assess_asset_analysis_gaps(self) -> Dict:
        """Document asset analysis confidence scores and gaps"""
        gaps = {
            'configuration_gaps': [],
            'qml_extraction_gaps': [],
            'python_analysis_gaps': [],
            'overall_confidence_score': 0.0,
            'recommendations': []
        }
        
        # Analyze configuration gaps
        config_validation = self.validation_results.get('configuration_parsing', {})
        if config_validation.get('parsing_coverage_rate', 0) < 0.8:
            gaps['configuration_gaps'].append({
                'issue': 'Low configuration parsing coverage',
                'coverage': config_validation.get('parsing_coverage_rate', 0),
                'impact': 'medium'
            })
        
        failed_files = config_validation.get('failed_files', [])
        if len(failed_files) > 0:
            gaps['configuration_gaps'].append({
                'issue': f'{len(failed_files)} configuration files failed to parse',
                'files': failed_files[:5],  # Show first 5
                'impact': 'low'
            })
        
        # Analyze QML extraction gaps
        qml_validation = self.validation_results.get('qml_extraction', {})
        if qml_validation.get('coverage_assessment') in ['limited', 'not_performed']:
            gaps['qml_extraction_gaps'].append({
                'issue': 'Limited QML file extraction',
                'coverage': qml_validation.get('coverage_assessment'),
                'impact': 'medium'
            })
        
        if qml_validation.get('average_confidence', 0) < 0.7:
            gaps['qml_extraction_gaps'].append({
                'issue': 'Low QML extraction confidence',
                'confidence': qml_validation.get('average_confidence', 0),
                'impact': 'medium'
            })
        
        # Analyze Python analysis gaps
        python_validation = self.validation_results.get('python_analysis', {})
        if python_validation.get('decompilation_success_rate', 0) < 0.5:
            gaps['python_analysis_gaps'].append({
                'issue': 'Low Python decompilation success rate',
                'success_rate': python_validation.get('decompilation_success_rate', 0),
                'impact': 'high'
            })
        
        if python_validation.get('service_mappings_found', 0) < 5:
            gaps['python_analysis_gaps'].append({
                'issue': 'Few Python-to-service mappings found',
                'mappings_found': python_validation.get('service_mappings_found', 0),
                'impact': 'medium'
            })
        
        # Calculate overall confidence
        confidence_scores = []
        
        if 'configuration_parsing' in self.validation_results:
            config_conf = 1.0 - len(gaps['configuration_gaps']) * 0.2
            confidence_scores.append(max(0.0, config_conf))
        
        if 'qml_extraction' in self.validation_results:
            qml_conf = qml_validation.get('average_confidence', 0.5)
            confidence_scores.append(qml_conf)
        
        if 'python_analysis' in self.validation_results:
            python_conf = python_validation.get('decompilation_success_rate', 0.5)
            confidence_scores.append(python_conf)
        
        if confidence_scores:
            gaps['overall_confidence_score'] = sum(confidence_scores) / len(confidence_scores)
        
        # Generate recommendations
        if gaps['overall_confidence_score'] < 0.6:
            gaps['recommendations'].append("Consider manual inspection of failed analysis components")
        
        if len(gaps['configuration_gaps']) > 0:
            gaps['recommendations'].append("Implement additional configuration file parsers")
        
        if len(gaps['python_analysis_gaps']) > 0:
            gaps['recommendations'].append("Try alternative Python decompilation tools")
        
        if len(gaps['qml_extraction_gaps']) > 0:
            gaps['recommendations'].append("Use Resource Hacker for additional QML extraction")
        
        return gaps
    
    def run_validation(self) -> Dict:
        """Run complete asset analysis validation"""
        print("=== ASSET ANALYSIS VALIDATION CHECKPOINT ===")
        
        # Run all validations
        print("Validating configuration parsing...")
        self.validation_results['configuration_parsing'] = self.validate_configuration_parsing()
        
        print("Validating QML extraction...")
        self.validation_results['qml_extraction'] = self.validate_qml_extraction()
        
        print("Validating Python analysis...")
        self.validation_results['python_analysis'] = self.validate_python_analysis()
        
        print("Cross-validating dependencies...")
        self.validation_results['dependency_cross_validation'] = self.cross_validate_dependencies()
        
        print("Assessing analysis gaps...")
        self.validation_results['gap_analysis'] = self.assess_asset_analysis_gaps()
        
        return self.validation_results
    
    def generate_validation_report(self) -> Dict:
        """Generate comprehensive validation report"""
        report = {
            'validation_summary': {
                'timestamp': subprocess.run(['date'], capture_output=True, text=True).stdout.strip(),
                'overall_status': 'unknown',
                'overall_confidence': 0.0,
                'critical_issues': 0,
                'recommendations_count': 0
            },
            'component_validations': self.validation_results,
            'executive_summary': {},
            'detailed_findings': {},
            'action_items': []
        }
        
        # Calculate overall status
        gap_analysis = self.validation_results.get('gap_analysis', {})
        overall_confidence = gap_analysis.get('overall_confidence_score', 0.0)
        
        if overall_confidence > 0.8:
            report['validation_summary']['overall_status'] = 'excellent'
        elif overall_confidence > 0.6:
            report['validation_summary']['overall_status'] = 'good'
        elif overall_confidence > 0.4:
            report['validation_summary']['overall_status'] = 'moderate'
        else:
            report['validation_summary']['overall_status'] = 'needs_improvement'
        
        report['validation_summary']['overall_confidence'] = round(overall_confidence, 2)
        
        # Count critical issues
        critical_issues = 0
        for gap_type in ['configuration_gaps', 'qml_extraction_gaps', 'python_analysis_gaps']:
            gaps = gap_analysis.get(gap_type, [])
            critical_issues += len([g for g in gaps if g.get('impact') == 'high'])
        
        report['validation_summary']['critical_issues'] = critical_issues
        report['validation_summary']['recommendations_count'] = len(gap_analysis.get('recommendations', []))
        
        # Executive summary
        config_val = self.validation_results.get('configuration_parsing', {})
        qml_val = self.validation_results.get('qml_extraction', {})
        python_val = self.validation_results.get('python_analysis', {})
        
        report['executive_summary'] = {
            'configuration_analysis': {
                'files_parsed': config_val.get('successfully_parsed', 0),
                'total_files': config_val.get('total_config_files', 0),
                'coverage_rate': f"{config_val.get('parsing_coverage_rate', 0):.1%}"
            },
            'qml_analysis': {
                'qml_files_found': qml_val.get('qml_files_found', 0),
                'ui_components': qml_val.get('ui_components_identified', 0),
                'coverage_assessment': qml_val.get('coverage_assessment', 'unknown')
            },
            'python_analysis': {
                'modules_analyzed': python_val.get('python_modules_found', 0),
                'pyd_modules': python_val.get('pyd_modules_found', 0),
                'decompilation_rate': f"{python_val.get('decompilation_success_rate', 0):.1%}",
                'service_mappings': python_val.get('service_mappings_found', 0)
            }
        }
        
        # Action items
        recommendations = gap_analysis.get('recommendations', [])
        for i, rec in enumerate(recommendations, 1):
            report['action_items'].append({
                'priority': 'high' if critical_issues > 0 else 'medium',
                'action': rec,
                'category': 'asset_analysis_improvement'
            })
        
        return report

if __name__ == "__main__":
    validator = AssetAnalysisValidator()
    
    # Run validation
    validation_results = validator.run_validation()
    
    # Generate report
    report = validator.generate_validation_report()
    
    # Save report
    os.makedirs('analysis_output', exist_ok=True)
    with open('analysis_output/asset_analysis_validation.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print("\n=== VALIDATION RESULTS ===\n")
    
    summary = report['validation_summary']
    print(f"Overall Status: {summary['overall_status'].upper()}")
    print(f"Overall Confidence: {summary['overall_confidence']:.2f}")
    print(f"Critical Issues: {summary['critical_issues']}")
    print(f"Recommendations: {summary['recommendations_count']}")
    
    print("\n=== COMPONENT ANALYSIS ===\n")
    
    exec_summary = report['executive_summary']
    
    print("Configuration Analysis:")
    config = exec_summary['configuration_analysis']
    print(f"  - Files parsed: {config['files_parsed']}/{config['total_files']} ({config['coverage_rate']})")
    
    print("\nQML Analysis:")
    qml = exec_summary['qml_analysis']
    print(f"  - QML files found: {qml['qml_files_found']}")
    print(f"  - UI components: {qml['ui_components']}")
    print(f"  - Coverage: {qml['coverage_assessment']}")
    
    print("\nPython Analysis:")
    python = exec_summary['python_analysis']
    print(f"  - Modules analyzed: {python['modules_analyzed']}")
    print(f"  - PYD modules: {python['pyd_modules']}")
    print(f"  - Decompilation rate: {python['decompilation_rate']}")
    print(f"  - Service mappings: {python['service_mappings']}")
    
    if report['action_items']:
        print("\n=== ACTION ITEMS ===\n")
        for item in report['action_items']:
            print(f"- [{item['priority'].upper()}] {item['action']}")
    
    print("\nDetailed validation report saved to: analysis_output/asset_analysis_validation.json")
    print("=== ASSET ANALYSIS VALIDATION COMPLETE ===")