#!/usr/bin/env python3
"""
Validation checkpoint summary - consolidates all validation findings
"""
import json
from pathlib import Path

def print_validation_checkpoint_summary():
    """Print a comprehensive summary of the validation checkpoint"""
    
    print("=" * 70)
    print("ARCHITECTURE VALIDATION CHECKPOINT - FINAL SUMMARY")
    print("=" * 70)
    
    # Load validation results
    validation_path = Path("analysis_output/architecture_validation.json")
    if not validation_path.exists():
        validation_path = Path("../analysis_output/architecture_validation.json")
    
    with open(validation_path, "r") as f:
        validation_data = json.load(f)
    
    # Load architecture summary
    summary_path = Path("analysis_output/architecture_summary.json")
    if not summary_path.exists():
        summary_path = Path("../analysis_output/architecture_summary.json")
        
    with open(summary_path, "r") as f:
        arch_data = json.load(f)
    
    summary = validation_data['validation_summary']
    
    print(f"\n📊 VALIDATION STATISTICS")
    print("-" * 30)
    print(f"Total Processes Analyzed: {summary['total_processes_analyzed']}")
    print(f"High Confidence (≥0.7):   {summary['high_confidence_count']} ({summary['high_confidence_count']/summary['total_processes_analyzed']*100:.0f}%)")
    print(f"Medium Confidence (0.4-0.7): {summary['medium_confidence_count']} ({summary['medium_confidence_count']/summary['total_processes_analyzed']*100:.0f}%)")
    print(f"Low Confidence (<0.4):    {summary['low_confidence_count']} ({summary['low_confidence_count']/summary['total_processes_analyzed']*100:.0f}%)")
    print(f"Critical Issues:          {summary['critical_issues_count']}")
    
    # Overall confidence assessment
    high_conf_ratio = summary['high_confidence_count'] / summary['total_processes_analyzed']
    if high_conf_ratio >= 0.6:
        status = "✅ EXCELLENT"
        color = "🟢"
    elif high_conf_ratio >= 0.4:
        status = "✅ GOOD"
        color = "🟡"
    elif high_conf_ratio >= 0.2:
        status = "⚠️  FAIR"
        color = "🟠"
    else:
        status = "❌ POOR"
        color = "🔴"
    
    print(f"\n{color} OVERALL VALIDATION STATUS: {status}")
    print(f"Architecture Understanding: {high_conf_ratio*100:.0f}% of key components well understood")
    
    print(f"\n🏗️  ARCHITECTURAL INSIGHTS")
    print("-" * 35)
    arch_summary = arch_data['system_architecture']
    print(f"System Type: {arch_summary['type']}")
    print(f"UI Framework: {arch_summary['ui_framework']}")
    print(f"Processing Model: {arch_summary['processing_model']}")
    print(f"Overall Architecture Confidence: {arch_summary['confidence']:.1f}/1.0")
    
    print(f"\n🔍 KEY FINDINGS")
    print("-" * 20)
    
    # High confidence components
    high_conf_components = [r for r in validation_data['detailed_results'] if r['confidence_score'] >= 0.7]
    print(f"✅ {len(high_conf_components)} High-Confidence Components:")
    for comp in high_conf_components:
        print(f"   • {comp['component']} ({comp['confidence_score']:.2f}) - {comp['inferred_role']}")
    
    # Critical issues
    critical_issues = validation_data.get('critical_issues', [])
    if critical_issues:
        print(f"\n⚠️  {len(critical_issues)} Critical Issues:")
        for issue in critical_issues[:3]:  # Show top 3
            print(f"   • {issue}")
        if len(critical_issues) > 3:
            print(f"   • ... and {len(critical_issues) - 3} more")
    
    # Architectural patterns
    patterns = arch_data.get('architectural_patterns', [])
    if patterns:
        print(f"\n🏛️  Identified Architectural Patterns:")
        for pattern in patterns:
            conf_indicator = "●" if pattern['confidence'] >= 0.7 else "◐" if pattern['confidence'] >= 0.4 else "○"
            print(f"   {conf_indicator} {pattern['pattern']} ({pattern['confidence']:.1f})")
    
    print(f"\n📋 RECOMMENDATIONS")
    print("-" * 20)
    recommendations = validation_data.get('recommendations', [])
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec}")
    
    print(f"\n📁 GENERATED ARTIFACTS")
    print("-" * 25)
    artifacts = [
        "analysis_output/architecture_validation.json",
        "analysis_output/architecture_summary.json", 
        "analysis_output/validation_checkpoint_report.md"
    ]
    
    for artifact in artifacts:
        if Path(artifact).exists():
            size_kb = Path(artifact).stat().st_size / 1024
            print(f"✓ {artifact} ({size_kb:.1f} KB)")
        else:
            print(f"✗ {artifact} (missing)")
    
    print(f"\n🎯 CHECKPOINT STATUS")
    print("-" * 22)
    
    # Determine if checkpoint passes
    passes_validation = (
        high_conf_ratio >= 0.3 and  # At least 30% high confidence
        summary['critical_issues_count'] <= 3 and  # No more than 3 critical issues
        summary['total_processes_analyzed'] >= 5  # Analyzed at least 5 processes
    )
    
    if passes_validation:
        print("✅ CHECKPOINT PASSED")
        print("   Ready to proceed with detailed binary analysis")
        print("   Architecture sufficiently understood for next phase")
    else:
        print("❌ CHECKPOINT REQUIRES ATTENTION")
        print("   Additional validation needed before proceeding")
        print("   Consider deeper analysis of critical issues")
    
    print("\n" + "=" * 70)

def main():
    """Main execution"""
    try:
        print_validation_checkpoint_summary()
    except FileNotFoundError as e:
        print(f"Error: Required validation files not found: {e}")
        print("Please run architecture_validator.py first")
    except Exception as e:
        print(f"Error generating summary: {e}")

if __name__ == "__main__":
    main()