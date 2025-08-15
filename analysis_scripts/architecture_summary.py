#!/usr/bin/env python3
"""
Architecture summary generator with confidence scoring
Provides high-level architectural insights from validation results
"""
import json
from pathlib import Path
from architecture_validator import ArchitectureValidator

def generate_architecture_summary():
    """Generate high-level architecture summary with confidence scores"""
    
    # Load validation results
    validator = ArchitectureValidator()
    report = validator.generate_validation_report()
    
    # Architecture insights
    insights = {
        'system_architecture': {
            'type': 'Multi-process dental scanning application',
            'ui_framework': 'Qt5/QML',
            'processing_model': 'Service-oriented with dedicated algorithm engines',
            'confidence': 0.8
        },
        'key_components': [],
        'communication_patterns': {
            'primary_ipc': 'Qt-based inter-process communication',
            'data_flow': 'UI → Scanner → Algorithm Service → Visualization',
            'confidence': 0.6
        },
        'technology_stack': {
            'ui': ['Qt5', 'QML', 'OpenSceneGraph'],
            'processing': ['Custom Sn3D libraries', 'OpenCV', 'CUDA (likely)'],
            'data': ['SQLite', 'Custom binary formats'],
            'confidence': 0.7
        },
        'architectural_patterns': [],
        'validation_summary': report['validation_summary']
    }
    
    # Analyze each component
    for result in report['detailed_results']:
        if result['confidence_score'] > 0.3:  # Only include reasonably confident results
            component_info = {
                'name': result['component'],
                'role': result['inferred_role'],
                'confidence': result['confidence_score'],
                'size_mb': 0,  # Will be filled from database
                'key_dependencies': [],
                'architectural_significance': 'Unknown'
            }
            
            # Determine architectural significance
            if 'IntraoralScan.exe' in result['component']:
                component_info['architectural_significance'] = 'Main UI orchestrator and entry point'
            elif 'AlgoService' in result['component']:
                component_info['architectural_significance'] = 'Core 3D processing and AI engine'
            elif 'ScanAppLogic' in result['component']:
                component_info['architectural_significance'] = 'Real-time scanning and data acquisition'
            elif 'Launcher' in result['component']:
                component_info['architectural_significance'] = 'System initialization and process management'
            elif 'Network' in result['component']:
                component_info['architectural_significance'] = 'Cloud connectivity and data synchronization'
            
            insights['key_components'].append(component_info)
    
    # Identify architectural patterns
    patterns = []
    
    # Service-oriented pattern
    service_count = len([c for c in insights['key_components'] if 'Service' in c['name']])
    if service_count > 0:
        patterns.append({
            'pattern': 'Service-Oriented Architecture',
            'evidence': f'{service_count} dedicated service processes identified',
            'confidence': 0.7
        })
    
    # Separation of concerns
    ui_components = len([c for c in insights['key_components'] if c['role'] in ['UI_ORCHESTRATOR', 'LAUNCHER']])
    algo_components = len([c for c in insights['key_components'] if c['role'] == 'ALGORITHM_ENGINE'])
    if ui_components > 0 and algo_components > 0:
        patterns.append({
            'pattern': 'Separation of UI and Processing Logic',
            'evidence': f'{ui_components} UI components, {algo_components} algorithm components',
            'confidence': 0.8
        })
    
    # Qt-based architecture
    qt_usage = len([c for c in insights['key_components'] if any('qt5' in dep.lower() for dep in c.get('key_dependencies', []))])
    if qt_usage > 2:
        patterns.append({
            'pattern': 'Qt-based Cross-Platform Architecture',
            'evidence': f'Qt5 dependencies found in {qt_usage} components',
            'confidence': 0.9
        })
    
    insights['architectural_patterns'] = patterns
    
    return insights

def print_architecture_summary(insights):
    """Print formatted architecture summary"""
    print("=" * 60)
    print("INTRAORAL SCAN ARCHITECTURE SUMMARY")
    print("=" * 60)
    
    # System overview
    arch = insights['system_architecture']
    print(f"\nSYSTEM TYPE: {arch['type']}")
    print(f"UI FRAMEWORK: {arch['ui_framework']}")
    print(f"PROCESSING MODEL: {arch['processing_model']}")
    print(f"CONFIDENCE: {arch['confidence']:.1f}/1.0")
    
    # Key components
    print(f"\nKEY COMPONENTS ({len(insights['key_components'])} identified):")
    print("-" * 50)
    for comp in sorted(insights['key_components'], key=lambda x: x['confidence'], reverse=True):
        conf_indicator = "●" if comp['confidence'] >= 0.7 else "◐" if comp['confidence'] >= 0.4 else "○"
        print(f"{conf_indicator} {comp['name']}")
        print(f"   Role: {comp['role']}")
        print(f"   Significance: {comp['architectural_significance']}")
        print(f"   Confidence: {comp['confidence']:.2f}")
        print()
    
    # Architectural patterns
    print("ARCHITECTURAL PATTERNS:")
    print("-" * 30)
    for pattern in insights['architectural_patterns']:
        conf_indicator = "●" if pattern['confidence'] >= 0.7 else "◐" if pattern['confidence'] >= 0.4 else "○"
        print(f"{conf_indicator} {pattern['pattern']}")
        print(f"   Evidence: {pattern['evidence']}")
        print(f"   Confidence: {pattern['confidence']:.1f}")
        print()
    
    # Technology stack
    tech = insights['technology_stack']
    print(f"TECHNOLOGY STACK (Confidence: {tech['confidence']:.1f}):")
    print("-" * 40)
    print(f"UI Technologies: {', '.join(tech['ui'])}")
    print(f"Processing: {', '.join(tech['processing'])}")
    print(f"Data Storage: {', '.join(tech['data'])}")
    
    # Communication patterns
    comm = insights['communication_patterns']
    print(f"\nCOMMUNICATION PATTERNS (Confidence: {comm['confidence']:.1f}):")
    print("-" * 45)
    print(f"Primary IPC: {comm['primary_ipc']}")
    print(f"Data Flow: {comm['data_flow']}")
    
    # Validation summary
    val_summary = insights['validation_summary']
    print(f"\nVALIDATION SUMMARY:")
    print("-" * 20)
    print(f"Processes Analyzed: {val_summary['total_processes_analyzed']}")
    print(f"High Confidence: {val_summary['high_confidence_count']}")
    print(f"Medium Confidence: {val_summary['medium_confidence_count']}")
    print(f"Low Confidence: {val_summary['low_confidence_count']}")
    print(f"Critical Issues: {val_summary['critical_issues_count']}")
    
    # Overall assessment
    total_confidence = sum(p['confidence'] for p in insights['architectural_patterns']) / len(insights['architectural_patterns']) if insights['architectural_patterns'] else 0.5
    
    print(f"\nOVERALL ARCHITECTURE CONFIDENCE: {total_confidence:.2f}/1.0")
    
    if total_confidence >= 0.7:
        print("✓ HIGH - Architecture well understood, ready for detailed analysis")
    elif total_confidence >= 0.5:
        print("◐ MEDIUM - Good understanding, some areas need deeper investigation")
    else:
        print("○ LOW - Significant gaps in understanding, requires more analysis")

def main():
    """Main execution"""
    insights = generate_architecture_summary()
    
    # Save detailed insights
    Path("analysis_output").mkdir(exist_ok=True)
    with open("analysis_output/architecture_summary.json", "w") as f:
        json.dump(insights, f, indent=2)
    
    # Print summary
    print_architecture_summary(insights)
    
    print(f"\nDetailed architecture summary saved to: analysis_output/architecture_summary.json")

if __name__ == "__main__":
    main()