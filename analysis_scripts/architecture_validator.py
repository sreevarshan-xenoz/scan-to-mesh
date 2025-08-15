#!/usr/bin/env python3
"""
Architecture validation checkpoint for IntraoralScan reverse engineering
Cross-validates process roles against dependency scans and applies domain knowledge
"""
import json
import sqlite3
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple
from database_utils import AnalysisDatabase

@dataclass
class ValidationResult:
    component: str
    expected_role: str
    inferred_role: str
    confidence_score: float
    inconsistencies: List[str]
    supporting_evidence: List[str]
    domain_validation: str

class ArchitectureValidator:
    def __init__(self, db_path="analysis_results.db"):
        self.db = AnalysisDatabase(db_path)
        
        # Domain knowledge for dental scanning workflow
        self.dental_workflow_stages = [
            "patient_management",
            "device_initialization", 
            "calibration",
            "real_time_scanning",
            "point_cloud_processing",
            "mesh_generation",
            "ai_analysis",
            "visualization",
            "export_processing",
            "order_management"
        ]
        
        # Expected process roles based on domain knowledge
        self.expected_roles = {
            'IntraoralScan.exe': {
                'role': 'UI_ORCHESTRATOR',
                'workflow_stages': ['patient_management', 'device_initialization', 'visualization'],
                'expected_deps': ['Qt5', 'QML', 'OpenGL', 'database'],
                'size_range': (50, 200)  # MB
            },
            'DentalLauncher.exe': {
                'role': 'LAUNCHER',
                'workflow_stages': ['device_initialization'],
                'expected_deps': ['system_libs'],
                'size_range': (1, 20)
            },
            'DentalAlgoService.exe': {
                'role': 'ALGORITHM_ENGINE',
                'workflow_stages': ['point_cloud_processing', 'mesh_generation', 'ai_analysis'],
                'expected_deps': ['CUDA', 'OpenCV', 'AI_frameworks', 'Sn3D'],
                'size_range': (20, 100)
            },
            'DentalScanAppLogic.exe': {
                'role': 'REALTIME_SCANNER',
                'workflow_stages': ['real_time_scanning', 'calibration'],
                'expected_deps': ['camera_drivers', 'CUDA', 'OpenCV'],
                'size_range': (10, 80)
            },
            'DentalNetwork.exe': {
                'role': 'NETWORK_SERVICE',
                'workflow_stages': ['order_management'],
                'expected_deps': ['network_libs', 'crypto'],
                'size_range': (5, 50)
            },
            'DentalOrderAppLogic.exe': {
                'role': 'ORDER_MANAGER',
                'workflow_stages': ['order_management', 'export_processing'],
                'expected_deps': ['database', 'network_libs'],
                'size_range': (10, 60)
            },
            'DentalDesignAppLogic.exe': {
                'role': 'DESIGN_PROCESSOR',
                'workflow_stages': ['mesh_generation', 'export_processing'],
                'expected_deps': ['3D_processing', 'CAD_libs'],
                'size_range': (15, 80)
            }
        }
        
        # Dependency patterns that indicate specific roles
        self.dependency_indicators = {
            'UI_ORCHESTRATOR': ['Qt5', 'QML', 'OpenGL', 'osg', 'WebEngine'],
            'ALGORITHM_ENGINE': ['cuda', 'opencv', 'Sn3D', 'MNN', 'onnx', 'tensorrt'],
            'REALTIME_SCANNER': ['camera', 'Vimba', 'opencv', 'cuda'],
            'NETWORK_SERVICE': ['ssl', 'crypto', 'network', 'curl'],
            'ORDER_MANAGER': ['sqlite', 'database', 'zip'],
            'DESIGN_PROCESSOR': ['mesh', '3D', 'CAD', 'geometry'],
            'LAUNCHER': ['system', 'kernel32', 'user32']
        }

    def validate_process_role(self, executable_name: str) -> ValidationResult:
        """Validate a single process role against dependencies and domain knowledge"""
        
        # Get dependency data
        dependencies = self.db.get_executable_dependencies(executable_name)
        summary = self.db.get_dependency_summary()
        
        # Find this executable in summary
        exe_data = next((item for item in summary if item['executable'] == executable_name), None)
        if not exe_data:
            return ValidationResult(
                component=executable_name,
                expected_role="UNKNOWN",
                inferred_role="NOT_FOUND",
                confidence_score=0.0,
                inconsistencies=["Executable not found in analysis"],
                supporting_evidence=[],
                domain_validation="FAILED - Not analyzed"
            )
        
        expected = self.expected_roles.get(executable_name, {})
        expected_role = expected.get('role', 'UNKNOWN')
        
        # Infer role from dependencies
        inferred_role = self._infer_role_from_dependencies(dependencies)
        
        # Calculate confidence score
        confidence = self._calculate_confidence(executable_name, dependencies, exe_data)
        
        # Find inconsistencies
        inconsistencies = self._find_inconsistencies(executable_name, dependencies, exe_data, expected)
        
        # Gather supporting evidence
        evidence = self._gather_supporting_evidence(dependencies, exe_data)
        
        # Domain validation
        domain_validation = self._validate_against_domain_knowledge(executable_name, dependencies, expected)
        
        return ValidationResult(
            component=executable_name,
            expected_role=expected_role,
            inferred_role=inferred_role,
            confidence_score=confidence,
            inconsistencies=inconsistencies,
            supporting_evidence=evidence,
            domain_validation=domain_validation
        )

    def _infer_role_from_dependencies(self, dependencies: List[Dict]) -> str:
        """Infer process role based on dependency patterns"""
        dep_names = [dep['name'].lower() for dep in dependencies]
        
        role_scores = {}
        for role, indicators in self.dependency_indicators.items():
            score = 0
            for indicator in indicators:
                if any(indicator.lower() in dep_name for dep_name in dep_names):
                    score += 1
            role_scores[role] = score / len(indicators)  # Normalize
        
        if not role_scores or max(role_scores.values()) == 0:
            return "UNKNOWN"
        
        return max(role_scores, key=role_scores.get)

    def _calculate_confidence(self, executable_name: str, dependencies: List[Dict], exe_data: Dict) -> float:
        """Calculate confidence score for role assignment"""
        confidence_factors = []
        
        # Factor 1: Analysis success (dependency extraction worked)
        if exe_data['analysis_success'] and exe_data.get('total_dependencies', 0) > 0:
            confidence_factors.append(0.4)
        elif exe_data['analysis_success']:
            confidence_factors.append(0.2)  # Analysis worked but no deps found
        
        # Factor 2: Size validation (more lenient ranges)
        expected = self.expected_roles.get(executable_name, {})
        size_range = expected.get('size_range', (0, 1000))
        exe_size_mb = exe_data['size_mb']
        if exe_size_mb > 0:
            # More lenient size validation - within 50% of range
            expanded_min = size_range[0] * 0.5
            expanded_max = size_range[1] * 1.5
            if expanded_min <= exe_size_mb <= expanded_max:
                confidence_factors.append(0.2)
            else:
                confidence_factors.append(0.1)  # Size available but way off
        
        # Factor 3: Dependency pattern match (ignore missing status for static analysis)
        expected_deps = expected.get('expected_deps', [])
        if expected_deps and dependencies:
            dep_names = [dep['name'].lower() for dep in dependencies]
            matches = sum(1 for exp_dep in expected_deps 
                         if any(exp_dep.lower() in dep_name for dep_name in dep_names))
            confidence_factors.append(0.3 * matches / len(expected_deps))
        
        # Factor 4: Dependency count indicates complexity
        total_count = exe_data.get('total_dependencies', 0)
        if total_count > 20:  # Complex application
            confidence_factors.append(0.1)
        elif total_count > 10:  # Moderate complexity
            confidence_factors.append(0.05)
        
        return min(1.0, sum(confidence_factors))

    def _find_inconsistencies(self, executable_name: str, dependencies: List[Dict], 
                            exe_data: Dict, expected: Dict) -> List[str]:
        """Find inconsistencies between expected and actual analysis"""
        inconsistencies = []
        
        # Size inconsistency (more lenient for static analysis)
        size_range = expected.get('size_range', (0, 1000))
        exe_size_mb = exe_data['size_mb']
        if exe_size_mb > 0:
            # Allow 2x variance for static analysis
            expanded_min = size_range[0] * 0.3
            expanded_max = size_range[1] * 2.0
            if not (expanded_min <= exe_size_mb <= expanded_max):
                inconsistencies.append(f"Size {exe_size_mb}MB significantly outside expected range {size_range}")
        
        # Missing critical expected dependencies (only flag if no related deps found)
        expected_deps = expected.get('expected_deps', [])
        dep_names = [dep['name'].lower() for dep in dependencies]
        critical_missing = []
        for exp_dep in expected_deps:
            if not any(exp_dep.lower() in dep_name for dep_name in dep_names):
                critical_missing.append(exp_dep)
        
        # Only flag as inconsistency if more than half of expected deps are missing
        if len(critical_missing) > len(expected_deps) / 2:
            inconsistencies.append(f"Missing most expected dependencies: {', '.join(critical_missing[:3])}")
        
        # Analysis failure
        if not exe_data['analysis_success']:
            inconsistencies.append("Dependency analysis failed")
        
        # No dependencies found (suspicious for complex executables)
        total_count = exe_data.get('total_dependencies', 0)
        if total_count == 0 and exe_data['size_mb'] > 5:
            inconsistencies.append("No dependencies found for large executable")
        
        return inconsistencies

    def _gather_supporting_evidence(self, dependencies: List[Dict], exe_data: Dict) -> List[str]:
        """Gather evidence supporting the role assignment"""
        evidence = []
        
        # Key dependencies found
        key_deps = []
        for dep in dependencies:
            dep_name = dep['name'].lower()
            if any(indicator in dep_name for indicators in self.dependency_indicators.values() 
                   for indicator in indicators):
                key_deps.append(dep['name'])
        
        if key_deps:
            evidence.append(f"Key dependencies found: {', '.join(key_deps[:5])}")
        
        # Size evidence
        if exe_data['size_mb'] > 0:
            evidence.append(f"Executable size: {exe_data['size_mb']}MB")
        
        # Dependency count
        total_deps = exe_data.get('total_dependencies', 0)
        if total_deps > 0:
            evidence.append(f"Total dependencies: {total_deps}")
        
        # Analysis method
        method = exe_data.get('analysis_method', 'unknown')
        if method != 'unknown':
            evidence.append(f"Analysis method: {method}")
        
        return evidence

    def _validate_against_domain_knowledge(self, executable_name: str, dependencies: List[Dict], 
                                         expected: Dict) -> str:
        """Validate against dental scanning domain knowledge"""
        
        workflow_stages = expected.get('workflow_stages', [])
        if not workflow_stages:
            return "UNKNOWN - No domain mapping defined"
        
        # Check if dependencies align with workflow stages
        dep_names = [dep['name'].lower() for dep in dependencies]
        
        validation_results = []
        for stage in workflow_stages:
            if stage == 'real_time_scanning':
                has_camera = any('camera' in dep or 'vimba' in dep for dep in dep_names)
                has_opencv = any('opencv' in dep for dep in dep_names)
                if has_camera and has_opencv:
                    validation_results.append(f"✓ {stage}: camera + opencv found")
                else:
                    validation_results.append(f"? {stage}: missing camera/opencv deps")
            
            elif stage == 'ai_analysis':
                has_ai = any(ai_lib in dep for dep in dep_names 
                           for ai_lib in ['cuda', 'mnn', 'onnx', 'tensorrt'])
                if has_ai:
                    validation_results.append(f"✓ {stage}: AI framework found")
                else:
                    validation_results.append(f"? {stage}: no AI frameworks detected")
            
            elif stage == 'visualization':
                has_graphics = any(gfx_lib in dep for dep in dep_names 
                                 for gfx_lib in ['opengl', 'qt5', 'osg'])
                if has_graphics:
                    validation_results.append(f"✓ {stage}: graphics libs found")
                else:
                    validation_results.append(f"? {stage}: no graphics libs detected")
            
            elif stage == 'order_management':
                has_db = any('sqlite' in dep or 'database' in dep for dep in dep_names)
                has_network = any('ssl' in dep or 'crypto' in dep for dep in dep_names)
                if has_db or has_network:
                    validation_results.append(f"✓ {stage}: db/network libs found")
                else:
                    validation_results.append(f"? {stage}: no db/network libs detected")
        
        return "; ".join(validation_results)

    def validate_all_processes(self) -> List[ValidationResult]:
        """Validate all high-value processes"""
        results = []
        
        for executable_name in self.expected_roles.keys():
            result = self.validate_process_role(executable_name)
            results.append(result)
        
        return results

    def generate_validation_report(self) -> Dict:
        """Generate comprehensive validation report"""
        results = self.validate_all_processes()
        
        # Calculate overall statistics
        total_processes = len(results)
        high_confidence = sum(1 for r in results if r.confidence_score >= 0.7)
        medium_confidence = sum(1 for r in results if 0.4 <= r.confidence_score < 0.7)
        low_confidence = sum(1 for r in results if r.confidence_score < 0.4)
        
        role_matches = sum(1 for r in results if r.expected_role == r.inferred_role)
        
        # Identify critical inconsistencies
        critical_issues = []
        for result in results:
            if result.confidence_score < 0.3:
                critical_issues.append(f"{result.component}: Very low confidence ({result.confidence_score:.2f})")
            if len(result.inconsistencies) >= 3:
                critical_issues.append(f"{result.component}: Multiple inconsistencies ({len(result.inconsistencies)})")
        
        report = {
            'validation_summary': {
                'total_processes_analyzed': total_processes,
                'high_confidence_count': high_confidence,
                'medium_confidence_count': medium_confidence,
                'low_confidence_count': low_confidence,
                'role_match_count': role_matches,
                'role_match_percentage': round(role_matches / total_processes * 100, 1) if total_processes > 0 else 0,
                'critical_issues_count': len(critical_issues)
            },
            'detailed_results': [
                {
                    'component': r.component,
                    'expected_role': r.expected_role,
                    'inferred_role': r.inferred_role,
                    'confidence_score': round(r.confidence_score, 3),
                    'role_match': r.expected_role == r.inferred_role,
                    'inconsistency_count': len(r.inconsistencies),
                    'inconsistencies': r.inconsistencies,
                    'supporting_evidence': r.supporting_evidence,
                    'domain_validation': r.domain_validation
                }
                for r in results
            ],
            'critical_issues': critical_issues,
            'recommendations': self._generate_recommendations(results)
        }
        
        return report

    def _generate_recommendations(self, results: List[ValidationResult]) -> List[str]:
        """Generate recommendations for further analysis"""
        recommendations = []
        
        # Low confidence processes need deeper analysis
        low_confidence = [r for r in results if r.confidence_score < 0.4]
        if low_confidence:
            recommendations.append(f"Perform deeper binary analysis on {len(low_confidence)} low-confidence processes")
        
        # Processes with many inconsistencies
        inconsistent = [r for r in results if len(r.inconsistencies) >= 2]
        if inconsistent:
            recommendations.append(f"Investigate {len(inconsistent)} processes with multiple inconsistencies")
        
        # Missing processes
        analyzed_count = len([r for r in results if r.inferred_role != "NOT_FOUND"])
        if analyzed_count < len(self.expected_roles):
            recommendations.append("Some expected high-value processes were not found in analysis")
        
        # Role mismatches
        mismatches = [r for r in results if r.expected_role != r.inferred_role and r.inferred_role != "NOT_FOUND"]
        if mismatches:
            recommendations.append(f"Validate {len(mismatches)} processes with role mismatches through runtime analysis")
        
        return recommendations

def main():
    """Main execution function"""
    print("=== ARCHITECTURE VALIDATION CHECKPOINT ===")
    print("Cross-validating process roles against dependency analysis...")
    
    validator = ArchitectureValidator()
    report = validator.generate_validation_report()
    
    # Save detailed report
    Path("analysis_output").mkdir(exist_ok=True)
    with open("analysis_output/architecture_validation.json", "w") as f:
        json.dump(report, f, indent=2)
    
    # Print summary
    summary = report['validation_summary']
    print(f"\n=== VALIDATION SUMMARY ===")
    print(f"Total processes analyzed: {summary['total_processes_analyzed']}")
    print(f"High confidence (≥0.7): {summary['high_confidence_count']}")
    print(f"Medium confidence (0.4-0.7): {summary['medium_confidence_count']}")
    print(f"Low confidence (<0.4): {summary['low_confidence_count']}")
    print(f"Role matches: {summary['role_match_count']}/{summary['total_processes_analyzed']} ({summary['role_match_percentage']}%)")
    print(f"Critical issues: {summary['critical_issues_count']}")
    
    # Print detailed results
    print(f"\n=== DETAILED VALIDATION RESULTS ===")
    for result in report['detailed_results']:
        status = "✓" if result['role_match'] else "✗"
        confidence_level = "HIGH" if result['confidence_score'] >= 0.7 else "MED" if result['confidence_score'] >= 0.4 else "LOW"
        
        print(f"{status} {result['component']}")
        print(f"   Expected: {result['expected_role']} | Inferred: {result['inferred_role']}")
        print(f"   Confidence: {result['confidence_score']:.3f} ({confidence_level})")
        
        if result['inconsistencies']:
            print(f"   Issues: {'; '.join(result['inconsistencies'][:2])}")
        
        if result['domain_validation']:
            domain_summary = result['domain_validation'][:100] + "..." if len(result['domain_validation']) > 100 else result['domain_validation']
            print(f"   Domain: {domain_summary}")
        print()
    
    # Print critical issues
    if report['critical_issues']:
        print("=== CRITICAL ISSUES REQUIRING ATTENTION ===")
        for issue in report['critical_issues']:
            print(f"⚠ {issue}")
        print()
    
    # Print recommendations
    print("=== RECOMMENDATIONS FOR FURTHER ANALYSIS ===")
    for i, rec in enumerate(report['recommendations'], 1):
        print(f"{i}. {rec}")
    
    print(f"\nDetailed validation report saved to: analysis_output/architecture_validation.json")
    
    return report

if __name__ == "__main__":
    main()