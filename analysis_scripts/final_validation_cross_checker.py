#!/usr/bin/env python3
"""
Final Validation Cross-Checker
Cross-validates all findings for consistency across different analysis methods
"""

import json
import sqlite3
import os
import sys
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Set, Tuple, Any
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FinalValidationCrossChecker:
    def __init__(self, analysis_output_dir: str = "analysis_output", db_path: str = "analysis_results.db"):
        self.analysis_output_dir = Path(analysis_output_dir)
        self.db_path = db_path
        self.findings = {}
        self.inconsistencies = []
        self.confidence_scores = {}
        self.validation_results = {
            'process_consistency': {},
            'communication_consistency': {},
            'pipeline_consistency': {},
            'dependency_consistency': {},
            'overall_confidence': 0.0
        }
    
    def load_all_findings(self):
        """Load all analysis findings from JSON files and database"""
        logger.info("Loading all analysis findings...")
        
        # Load JSON findings
        json_files = [
            'inventory_results.json',
            'high_value_analysis.json',
            'dependency_overview.json',
            'architecture_analysis.json',
            'communication_validation.json',
            'asset_analysis_validation.json',
            'pipeline_reconstruction.json',
            'algorithm_dll_analysis.json',
            'hardware_interface_report.json',
            'python_runtime_analysis.json',
            'qml_qt_analysis.json',
            'network_endpoints.json',
            'ipc_endpoints.json'
        ]
        
        for json_file in json_files:
            file_path = self.analysis_output_dir / json_file
            if file_path.exists():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        self.findings[json_file.replace('.json', '')] = json.load(f)
                    logger.info(f"Loaded {json_file}")
                except Exception as e:
                    logger.error(f"Failed to load {json_file}: {e}")
        
        # Load database findings if available
        if os.path.exists(self.db_path):
            self.load_database_findings()
    
    def load_database_findings(self):
        """Load findings from SQLite database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get all table names
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            
            self.findings['database'] = {}
            for table_name in tables:
                table = table_name[0]
                cursor.execute(f"SELECT * FROM {table}")
                rows = cursor.fetchall()
                cursor.execute(f"PRAGMA table_info({table})")
                columns = [col[1] for col in cursor.fetchall()]
                
                self.findings['database'][table] = []
                for row in rows:
                    self.findings['database'][table].append(dict(zip(columns, row)))
            
            conn.close()
            logger.info("Loaded database findings")
        except Exception as e:
            logger.error(f"Failed to load database: {e}")
    
    def cross_validate_process_analysis(self):
        """Cross-validate process analysis findings"""
        logger.info("Cross-validating process analysis...")
        
        # Get process lists from different sources
        inventory_processes = set()
        if 'inventory_results' in self.findings:
            executables = self.findings['inventory_results'].get('executables', [])
            if isinstance(executables, list):
                inventory_processes = set(exe.get('path', '').split('/')[-1] for exe in executables if exe.get('path'))
            elif isinstance(executables, dict):
                inventory_processes = set(executables.keys())
        
        high_value_processes = set()
        if 'high_value_analysis' in self.findings:
            targets = self.findings['high_value_analysis'].get('high_value_targets', {})
            if isinstance(targets, dict):
                high_value_processes = set(targets.keys())
            elif isinstance(targets, list):
                high_value_processes = set(target.get('name', '') for target in targets if target.get('name'))
        
        dependency_processes = set()
        if 'dependency_overview' in self.findings:
            data = self.findings['dependency_overview']
            if isinstance(data, list):
                dependency_processes = set(proc.get('executable', '').split('/')[-1] for proc in data if proc.get('executable'))
            elif isinstance(data, dict):
                processes = data.get('processes', {})
                if isinstance(processes, dict):
                    dependency_processes = set(processes.keys())
                elif isinstance(processes, list):
                    dependency_processes = set(proc.get('name', '') for proc in processes if proc.get('name'))
        
        # Check consistency
        all_processes = inventory_processes | high_value_processes | dependency_processes
        
        consistency_score = 0.0
        if all_processes:
            # Calculate overlap percentages
            inventory_coverage = len(inventory_processes & all_processes) / len(all_processes) if all_processes else 0
            high_value_coverage = len(high_value_processes & all_processes) / len(all_processes) if all_processes else 0
            dependency_coverage = len(dependency_processes & all_processes) / len(all_processes) if all_processes else 0
            
            consistency_score = (inventory_coverage + high_value_coverage + dependency_coverage) / 3
        
        self.validation_results['process_consistency'] = {
            'total_processes': len(all_processes),
            'inventory_processes': len(inventory_processes),
            'high_value_processes': len(high_value_processes),
            'dependency_processes': len(dependency_processes),
            'consistency_score': consistency_score,
            'missing_from_inventory': list(all_processes - inventory_processes),
            'missing_from_high_value': list(all_processes - high_value_processes),
            'missing_from_dependency': list(all_processes - dependency_processes)
        }
        
        if consistency_score < 0.8:
            self.inconsistencies.append(f"Process analysis consistency low: {consistency_score:.2f}")
    
    def cross_validate_communication_patterns(self):
        """Cross-validate communication pattern findings"""
        logger.info("Cross-validating communication patterns...")
        
        # Get communication endpoints from different sources
        ipc_endpoints = set()
        if 'ipc_endpoints' in self.findings:
            endpoints = self.findings['ipc_endpoints'].get('endpoints', [])
            if isinstance(endpoints, list):
                for endpoint in endpoints:
                    if isinstance(endpoint, dict):
                        ipc_endpoints.add(endpoint.get('name', ''))
                    elif isinstance(endpoint, str):
                        ipc_endpoints.add(endpoint)
        
        network_endpoints = set()
        if 'network_endpoints' in self.findings:
            endpoints = self.findings['network_endpoints'].get('endpoints', [])
            if isinstance(endpoints, list):
                for endpoint in endpoints:
                    if isinstance(endpoint, dict):
                        network_endpoints.add(f"{endpoint.get('host', '')}:{endpoint.get('port', '')}")
                    elif isinstance(endpoint, str):
                        network_endpoints.add(endpoint)
        
        communication_validation = set()
        if 'communication_validation' in self.findings:
            patterns = self.findings['communication_validation'].get('patterns', [])
            if isinstance(patterns, list):
                for pattern in patterns:
                    if isinstance(pattern, dict):
                        communication_validation.add(pattern.get('endpoint', ''))
                    elif isinstance(pattern, str):
                        communication_validation.add(pattern)
        
        # Check consistency
        all_endpoints = ipc_endpoints | network_endpoints | communication_validation
        
        consistency_score = 0.0
        if all_endpoints:
            ipc_coverage = len(ipc_endpoints & all_endpoints) / len(all_endpoints) if all_endpoints else 0
            network_coverage = len(network_endpoints & all_endpoints) / len(all_endpoints) if all_endpoints else 0
            validation_coverage = len(communication_validation & all_endpoints) / len(all_endpoints) if all_endpoints else 0
            
            consistency_score = (ipc_coverage + network_coverage + validation_coverage) / 3
        
        self.validation_results['communication_consistency'] = {
            'total_endpoints': len(all_endpoints),
            'ipc_endpoints': len(ipc_endpoints),
            'network_endpoints': len(network_endpoints),
            'validation_endpoints': len(communication_validation),
            'consistency_score': consistency_score
        }
        
        if consistency_score < 0.7:
            self.inconsistencies.append(f"Communication pattern consistency low: {consistency_score:.2f}")
    
    def cross_validate_pipeline_reconstruction(self):
        """Cross-validate pipeline reconstruction against component dependencies"""
        logger.info("Cross-validating pipeline reconstruction...")
        
        pipeline_components = set()
        if 'pipeline_reconstruction' in self.findings:
            stages = self.findings['pipeline_reconstruction'].get('stages', [])
            if isinstance(stages, list):
                for stage in stages:
                    if isinstance(stage, dict):
                        components = stage.get('components', [])
                        if isinstance(components, list):
                            pipeline_components.update(components)
        
        algorithm_components = set()
        if 'algorithm_dll_analysis' in self.findings:
            dlls = self.findings['algorithm_dll_analysis'].get('dlls', {})
            if isinstance(dlls, dict):
                algorithm_components = set(dlls.keys())
            elif isinstance(dlls, list):
                algorithm_components = set(dll.get('name', '') for dll in dlls if dll.get('name'))
        
        dependency_components = set()
        if 'dependency_overview' in self.findings:
            data = self.findings['dependency_overview']
            if isinstance(data, list):
                for proc in data:
                    if isinstance(proc, dict):
                        deps = proc.get('dependencies', [])
                        if isinstance(deps, list):
                            dependency_components.update(deps)
            elif isinstance(data, dict):
                processes = data.get('processes', {})
                if isinstance(processes, dict):
                    for process_deps in processes.values():
                        if isinstance(process_deps, dict):
                            deps = process_deps.get('dependencies', [])
                            if isinstance(deps, list):
                                dependency_components.update(deps)
        
        # Check pipeline component coverage
        all_components = pipeline_components | algorithm_components | dependency_components
        
        consistency_score = 0.0
        if all_components:
            pipeline_coverage = len(pipeline_components & all_components) / len(all_components) if all_components else 0
            algorithm_coverage = len(algorithm_components & all_components) / len(all_components) if all_components else 0
            dependency_coverage = len(dependency_components & all_components) / len(all_components) if all_components else 0
            
            consistency_score = (pipeline_coverage + algorithm_coverage + dependency_coverage) / 3
        
        self.validation_results['pipeline_consistency'] = {
            'total_components': len(all_components),
            'pipeline_components': len(pipeline_components),
            'algorithm_components': len(algorithm_components),
            'dependency_components': len(dependency_components),
            'consistency_score': consistency_score,
            'missing_from_pipeline': list(all_components - pipeline_components)
        }
        
        if consistency_score < 0.6:
            self.inconsistencies.append(f"Pipeline reconstruction consistency low: {consistency_score:.2f}")
    
    def calculate_confidence_scores(self):
        """Calculate and document confidence scores for all findings"""
        logger.info("Calculating confidence scores...")
        
        # Define confidence scoring criteria
        confidence_criteria = {
            'direct_evidence': 1.0,      # Configuration files, exports, strings
            'strong_inference': 0.8,     # Naming patterns, dependencies
            'medium_inference': 0.6,     # Domain knowledge, partial evidence
            'weak_inference': 0.4,       # Speculation, incomplete data
            'unknown': 0.2               # No evidence
        }
        
        # Score different analysis types
        analysis_confidence = {}
        
        # Process analysis confidence
        if 'inventory_results' in self.findings:
            analysis_confidence['process_inventory'] = 0.9  # Direct file system evidence
        
        if 'dependency_overview' in self.findings:
            analysis_confidence['dependency_mapping'] = 0.8  # Tool-based analysis
        
        # Communication analysis confidence
        if 'ipc_endpoints' in self.findings:
            analysis_confidence['ipc_detection'] = 0.7  # String-based inference
        
        if 'network_endpoints' in self.findings:
            analysis_confidence['network_analysis'] = 0.6  # Configuration-based
        
        # Algorithm analysis confidence
        if 'algorithm_dll_analysis' in self.findings:
            analysis_confidence['algorithm_analysis'] = 0.8  # Export-based analysis
        
        # Pipeline reconstruction confidence
        if 'pipeline_reconstruction' in self.findings:
            analysis_confidence['pipeline_reconstruction'] = 0.6  # Inference-heavy
        
        # Calculate overall confidence
        if analysis_confidence:
            overall_confidence = sum(analysis_confidence.values()) / len(analysis_confidence)
        else:
            overall_confidence = 0.0
        
        self.confidence_scores = analysis_confidence
        self.validation_results['overall_confidence'] = overall_confidence
        
        # Flag low-confidence assumptions
        low_confidence_items = []
        for analysis, score in analysis_confidence.items():
            if score < 0.5:
                low_confidence_items.append(f"{analysis}: {score:.2f}")
        
        if low_confidence_items:
            self.inconsistencies.extend([f"Low confidence: {item}" for item in low_confidence_items])
    
    def validate_dependency_consistency(self):
        """Validate dependency relationships across different analysis methods"""
        logger.info("Validating dependency consistency...")
        
        # Collect dependencies from different sources
        dependency_sources = {}
        
        if 'dependency_overview' in self.findings:
            data = self.findings['dependency_overview']
            if isinstance(data, dict):
                dependency_sources['dependency_tool'] = data.get('processes', {})
            elif isinstance(data, list):
                # Convert list to dict format for consistency
                dependency_sources['dependency_tool'] = {
                    item.get('executable', '').split('/')[-1]: item.get('dependencies', [])
                    for item in data if item.get('executable')
                }
        
        if 'algorithm_dll_analysis' in self.findings:
            dependency_sources['algorithm_analysis'] = {}
            for dll, info in self.findings['algorithm_dll_analysis'].get('dlls', {}).items():
                dependency_sources['algorithm_analysis'][dll] = info.get('dependencies', [])
        
        # Cross-validate dependencies
        common_processes = set()
        for source in dependency_sources.values():
            if isinstance(source, dict):
                common_processes.update(source.keys())
            elif isinstance(source, list):
                common_processes.update(item.get('executable', '').split('/')[-1] for item in source if item.get('executable'))
        
        consistency_issues = []
        for process in common_processes:
            deps_by_source = {}
            for source_name, source_data in dependency_sources.items():
                if isinstance(source_data, dict) and process in source_data:
                    deps = source_data[process]
                    if isinstance(deps, dict):
                        deps = deps.get('dependencies', [])
                    deps_by_source[source_name] = set(deps) if deps else set()
                elif isinstance(source_data, list):
                    for item in source_data:
                        if item.get('executable', '').split('/')[-1] == process:
                            deps = item.get('dependencies', [])
                            deps_by_source[source_name] = set(deps) if deps else set()
                            break
            
            # Check for major discrepancies
            if len(deps_by_source) > 1:
                all_deps = set()
                for deps in deps_by_source.values():
                    all_deps.update(deps)
                
                # Calculate agreement percentage
                agreements = []
                for source_deps in deps_by_source.values():
                    if all_deps:
                        agreement = len(source_deps & all_deps) / len(all_deps)
                        agreements.append(agreement)
                
                if agreements and min(agreements) < 0.5:
                    consistency_issues.append(f"Dependency mismatch for {process}: {deps_by_source}")
        
        self.validation_results['dependency_consistency'] = {
            'total_processes_checked': len(common_processes),
            'consistency_issues': len(consistency_issues),
            'issues': consistency_issues[:10]  # Limit to first 10 for readability
        }
        
        if consistency_issues:
            self.inconsistencies.extend([f"Dependency inconsistency: {issue}" for issue in consistency_issues[:5]])
    
    def generate_validation_report(self):
        """Generate comprehensive validation report"""
        logger.info("Generating validation report...")
        
        report = {
            'validation_summary': {
                'total_inconsistencies': len(self.inconsistencies),
                'overall_confidence': self.validation_results['overall_confidence'],
                'validation_date': str(Path().cwd()),
                'findings_sources': len(self.findings)
            },
            'consistency_checks': self.validation_results,
            'confidence_scores': self.confidence_scores,
            'inconsistencies': self.inconsistencies,
            'recommendations': self.generate_recommendations()
        }
        
        # Save validation report
        output_file = self.analysis_output_dir / 'final_validation_report.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # Generate markdown summary
        self.generate_markdown_summary(report)
        
        logger.info(f"Validation report saved to {output_file}")
        return report
    
    def generate_recommendations(self):
        """Generate recommendations based on validation results"""
        recommendations = []
        
        # Process consistency recommendations
        if self.validation_results['process_consistency'].get('consistency_score', 0) < 0.8:
            recommendations.append("Re-examine process inventory for completeness")
        
        # Communication consistency recommendations
        if self.validation_results['communication_consistency'].get('consistency_score', 0) < 0.7:
            recommendations.append("Validate communication patterns with runtime analysis")
        
        # Pipeline consistency recommendations
        if self.validation_results['pipeline_consistency'].get('consistency_score', 0) < 0.6:
            recommendations.append("Refine pipeline reconstruction with additional component analysis")
        
        # Overall confidence recommendations
        if self.validation_results['overall_confidence'] < 0.7:
            recommendations.append("Increase analysis depth for low-confidence findings")
        
        # Dependency consistency recommendations
        if self.validation_results['dependency_consistency'].get('consistency_issues', 0) > 5:
            recommendations.append("Resolve dependency mapping inconsistencies")
        
        return recommendations
    
    def generate_markdown_summary(self, report):
        """Generate markdown summary of validation results"""
        markdown_content = f"""# Final Validation Cross-Check Report

## Summary
- **Total Inconsistencies**: {report['validation_summary']['total_inconsistencies']}
- **Overall Confidence**: {report['validation_summary']['overall_confidence']:.2f}
- **Findings Sources**: {report['validation_summary']['findings_sources']}

## Consistency Checks

### Process Analysis Consistency
- **Consistency Score**: {report['consistency_checks']['process_consistency'].get('consistency_score', 0):.2f}
- **Total Processes**: {report['consistency_checks']['process_consistency'].get('total_processes', 0)}

### Communication Pattern Consistency
- **Consistency Score**: {report['consistency_checks']['communication_consistency'].get('consistency_score', 0):.2f}
- **Total Endpoints**: {report['consistency_checks']['communication_consistency'].get('total_endpoints', 0)}

### Pipeline Reconstruction Consistency
- **Consistency Score**: {report['consistency_checks']['pipeline_consistency'].get('consistency_score', 0):.2f}
- **Total Components**: {report['consistency_checks']['pipeline_consistency'].get('total_components', 0)}

### Dependency Consistency
- **Processes Checked**: {report['consistency_checks']['dependency_consistency'].get('total_processes_checked', 0)}
- **Consistency Issues**: {report['consistency_checks']['dependency_consistency'].get('consistency_issues', 0)}

## Confidence Scores
"""
        
        for analysis, score in report['confidence_scores'].items():
            markdown_content += f"- **{analysis}**: {score:.2f}\n"
        
        markdown_content += f"""
## Inconsistencies Found
"""
        
        for inconsistency in report['inconsistencies']:
            markdown_content += f"- {inconsistency}\n"
        
        markdown_content += f"""
## Recommendations
"""
        
        for recommendation in report['recommendations']:
            markdown_content += f"- {recommendation}\n"
        
        # Save markdown report
        output_file = self.analysis_output_dir / 'final_validation_report.md'
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
    
    def run_validation(self):
        """Run complete validation process"""
        logger.info("Starting final validation cross-check...")
        
        # Load all findings
        self.load_all_findings()
        
        # Run cross-validation checks
        self.cross_validate_process_analysis()
        self.cross_validate_communication_patterns()
        self.cross_validate_pipeline_reconstruction()
        self.validate_dependency_consistency()
        self.calculate_confidence_scores()
        
        # Generate report
        report = self.generate_validation_report()
        
        logger.info("Final validation cross-check completed")
        return report

def main():
    """Main execution function"""
    validator = FinalValidationCrossChecker()
    report = validator.run_validation()
    
    print(f"\nValidation Summary:")
    print(f"- Total Inconsistencies: {len(validator.inconsistencies)}")
    print(f"- Overall Confidence: {report['validation_summary']['overall_confidence']:.2f}")
    print(f"- Findings Sources: {report['validation_summary']['findings_sources']}")
    
    if validator.inconsistencies:
        print(f"\nTop Inconsistencies:")
        for i, inconsistency in enumerate(validator.inconsistencies[:5], 1):
            print(f"{i}. {inconsistency}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())