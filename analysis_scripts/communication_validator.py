#!/usr/bin/env python3
"""
Communication Validation Checkpoint
Cross-validates IPC and network findings with configuration and dependency analysis
"""
import sqlite3
import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

class CommunicationValidator:
    def __init__(self, db_path="analysis_results.db"):
        self.db_path = db_path
        self.validation_results = {
            'ipc_validation': {},
            'network_validation': {},
            'cross_validation': {},
            'confidence_scores': {},
            'inconsistencies': [],
            'recommendations': []
        }
    
    def load_ipc_findings(self) -> Dict:
        """Load IPC endpoint findings from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get IPC summary
        cursor.execute('''
            SELECT 
                e.name,
                ia.total_endpoints,
                ia.named_pipes,
                ia.shared_memory,
                ia.sockets,
                ia.message_queues,
                ia.events,
                ia.analysis_success
            FROM ipc_analysis ia
            JOIN executables e ON ia.executable_id = e.id
            WHERE ia.analysis_success = 1
            ORDER BY ia.total_endpoints DESC
        ''')
        
        ipc_summary = {}
        for row in cursor.fetchall():
            ipc_summary[row[0]] = {
                'total_endpoints': row[1],
                'named_pipes': row[2],
                'shared_memory': row[3],
                'sockets': row[4],
                'message_queues': row[5],
                'events': row[6]
            }
        
        # Get detailed IPC endpoints
        cursor.execute('''
            SELECT 
                e.name,
                ie.endpoint_type,
                ie.endpoint_name,
                ie.confidence_score,
                ie.context
            FROM ipc_endpoints ie
            JOIN executables e ON ie.executable_id = e.id
            ORDER BY e.name, ie.confidence_score DESC
        ''')
        
        ipc_details = {}
        for row in cursor.fetchall():
            executable = row[0]
            if executable not in ipc_details:
                ipc_details[executable] = []
            
            ipc_details[executable].append({
                'type': row[1],
                'name': row[2],
                'confidence': row[3],
                'context': row[4]
            })
        
        conn.close()
        return {'summary': ipc_summary, 'details': ipc_details}
    
    def load_network_findings(self) -> Dict:
        """Load network endpoint findings from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get network summary
        cursor.execute('''
            SELECT 
                e.name,
                na.total_endpoints,
                na.urls,
                na.ip_addresses,
                na.ports,
                na.hostnames,
                na.protocols_detected,
                na.analysis_success
            FROM network_analysis na
            JOIN executables e ON na.executable_id = e.id
            WHERE na.analysis_success = 1
            ORDER BY na.total_endpoints DESC
        ''')
        
        network_summary = {}
        for row in cursor.fetchall():
            protocols = json.loads(row[6]) if row[6] else []
            network_summary[row[0]] = {
                'total_endpoints': row[1],
                'urls': row[2],
                'ip_addresses': row[3],
                'ports': row[4],
                'hostnames': row[5],
                'protocols_detected': protocols
            }
        
        # Get detailed network endpoints
        cursor.execute('''
            SELECT 
                e.name,
                ne.endpoint_type,
                ne.endpoint_value,
                ne.protocol,
                ne.port,
                ne.confidence_score,
                ne.detection_method,
                ne.context
            FROM network_endpoints ne
            JOIN executables e ON ne.executable_id = e.id
            ORDER BY e.name, ne.confidence_score DESC
        ''')
        
        network_details = {}
        for row in cursor.fetchall():
            executable = row[0]
            if executable not in network_details:
                network_details[executable] = []
            
            network_details[executable].append({
                'type': row[1],
                'value': row[2],
                'protocol': row[3],
                'port': row[4],
                'confidence': row[5],
                'detection_method': row[6],
                'context': row[7]
            })
        
        conn.close()
        return {'summary': network_summary, 'details': network_details}
    
    def load_config_data(self) -> Dict:
        """Load configuration analysis data"""
        config_data = {}
        
        # Load architecture analysis if available
        if os.path.exists('analysis_output/architecture_analysis.json'):
            with open('analysis_output/architecture_analysis.json', 'r') as f:
                config_data = json.load(f)
        
        return config_data
    
    def load_dependency_data(self) -> Dict:
        """Load dependency analysis data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get dependency summary
        cursor.execute('''
            SELECT 
                e.name,
                da.total_dependencies,
                da.system_dependencies,
                da.missing_dependencies,
                da.analysis_success
            FROM dependency_analysis da
            JOIN executables e ON da.executable_id = e.id
            WHERE da.analysis_success = 1
            ORDER BY da.total_dependencies DESC
        ''')
        
        dependency_summary = {}
        for row in cursor.fetchall():
            dependency_summary[row[0]] = {
                'total_dependencies': row[1],
                'system_dependencies': row[2],
                'missing_dependencies': row[3]
            }
        
        conn.close()
        return dependency_summary
    
    def validate_ipc_consistency(self, ipc_data: Dict, config_data: Dict) -> Dict:
        """Validate IPC findings against configuration and expected patterns"""
        validation = {
            'consistent_findings': [],
            'inconsistent_findings': [],
            'missing_expected': [],
            'confidence_adjustments': {}
        }
        
        # Expected IPC patterns based on config
        expected_shared_memory = config_data.get('communication_channels', {}).get('main_tunnel', {}).get('shared_memory', 'DentalShared')
        expected_service_hub = config_data.get('communication_channels', {}).get('main_tunnel', {}).get('service_name', 'DentalHub')
        
        for executable, details in ipc_data['details'].items():
            for endpoint in details:
                # Validate shared memory findings
                if endpoint['type'] == 'shared_memory':
                    if expected_shared_memory.lower() in endpoint['name'].lower():
                        validation['consistent_findings'].append({
                            'executable': executable,
                            'endpoint': endpoint,
                            'validation_reason': f"Matches expected shared memory: {expected_shared_memory}"
                        })
                        # Boost confidence
                        validation['confidence_adjustments'][f"{executable}:{endpoint['name']}"] = min(1.0, endpoint['confidence'] + 0.2)
                    
                    # Check for calibration-related shared memory (domain-specific validation)
                    if any(keyword in endpoint['name'].lower() for keyword in ['calib', 'image', 'shm']):
                        validation['consistent_findings'].append({
                            'executable': executable,
                            'endpoint': endpoint,
                            'validation_reason': "Matches expected calibration/imaging shared memory pattern"
                        })
                
                # Validate events (Windows synchronization objects)
                if endpoint['type'] == 'events':
                    if any(api in endpoint['context'] for api in ['CreateEvent', 'SetEvent', 'WaitForSingleObject']):
                        validation['consistent_findings'].append({
                            'executable': executable,
                            'endpoint': endpoint,
                            'validation_reason': "Valid Windows event synchronization pattern"
                        })
        
        # Check for missing expected IPC mechanisms
        main_executables = ['IntraoralScan.exe', 'DentalAlgoService.exe', 'DentalScanAppLogic.exe']
        for executable in main_executables:
            if executable in ipc_data['summary']:
                summary = ipc_data['summary'][executable]
                
                # Main executable should have shared memory for IPC
                if executable == 'IntraoralScan.exe' and summary['shared_memory'] == 0:
                    validation['missing_expected'].append({
                        'executable': executable,
                        'missing_type': 'shared_memory',
                        'reason': 'Main executable should use shared memory for IPC coordination'
                    })
                
                # Algorithm service should have events for synchronization
                if executable == 'DentalAlgoService.exe' and summary['events'] == 0:
                    validation['missing_expected'].append({
                        'executable': executable,
                        'missing_type': 'events',
                        'reason': 'Algorithm service should use events for processing synchronization'
                    })
        
        return validation
    
    def validate_network_consistency(self, network_data: Dict, config_data: Dict) -> Dict:
        """Validate network findings against configuration"""
        validation = {
            'consistent_findings': [],
            'inconsistent_findings': [],
            'missing_expected': [],
            'confidence_adjustments': {}
        }
        
        # Expected network endpoints from config
        expected_tunnel_port = config_data.get('communication_channels', {}).get('main_tunnel', {}).get('port', 18830)
        expected_http_sockets = config_data.get('network_endpoints', {}).get('http_sockets', [':3000', ':3001'])
        
        for executable, details in network_data['details'].items():
            for endpoint in details:
                # Validate against config ports
                if endpoint['type'] == 'port' and endpoint['port']:
                    if endpoint['port'] == expected_tunnel_port:
                        validation['consistent_findings'].append({
                            'executable': executable,
                            'endpoint': endpoint,
                            'validation_reason': f"Matches expected main tunnel port: {expected_tunnel_port}"
                        })
                        validation['confidence_adjustments'][f"{executable}:{endpoint['value']}"] = min(1.0, endpoint['confidence'] + 0.2)
                    
                    if f":{endpoint['port']}" in expected_http_sockets:
                        validation['consistent_findings'].append({
                            'executable': executable,
                            'endpoint': endpoint,
                            'validation_reason': f"Matches expected HTTP socket: :{endpoint['port']}"
                        })
                        validation['confidence_adjustments'][f"{executable}:{endpoint['value']}"] = min(1.0, endpoint['confidence'] + 0.2)
                
                # Validate protocol consistency
                if endpoint['type'] == 'hostname' and endpoint['protocol'] == 'tcp':
                    if 'localhost' in endpoint['value'].lower() or '127.0.0.1' in endpoint['value']:
                        validation['consistent_findings'].append({
                            'executable': executable,
                            'endpoint': endpoint,
                            'validation_reason': "Valid local TCP communication pattern"
                        })
        
        # Check for missing expected network endpoints
        network_executable = 'DentalNetwork.exe'
        if network_executable in network_data['summary']:
            summary = network_data['summary'][network_executable]
            
            # Network service should have MQTT protocol
            if 'mqtt' not in summary['protocols_detected']:
                validation['missing_expected'].append({
                    'executable': network_executable,
                    'missing_type': 'mqtt_protocol',
                    'reason': 'Network service should support MQTT for cloud communication'
                })
        
        return validation
    
    def cross_validate_communication_patterns(self, ipc_data: Dict, network_data: Dict, dependency_data: Dict) -> Dict:
        """Cross-validate communication patterns across different analysis types"""
        cross_validation = {
            'communication_matrix': {},
            'service_interactions': [],
            'protocol_consistency': {},
            'data_flow_patterns': []
        }
        
        # Build communication matrix
        all_executables = set(list(ipc_data['summary'].keys()) + list(network_data['summary'].keys()))
        
        for executable in all_executables:
            cross_validation['communication_matrix'][executable] = {
                'ipc_capabilities': ipc_data['summary'].get(executable, {}),
                'network_capabilities': network_data['summary'].get(executable, {}),
                'dependency_info': dependency_data.get(executable, {}),
                'communication_role': self.infer_communication_role(executable, ipc_data, network_data)
            }
        
        # Identify service interactions
        for executable in all_executables:
            ipc_summary = ipc_data['summary'].get(executable, {})
            network_summary = network_data['summary'].get(executable, {})
            
            # Infer interaction patterns
            if ipc_summary.get('shared_memory', 0) > 0 and network_summary.get('ports', 0) > 0:
                cross_validation['service_interactions'].append({
                    'service': executable,
                    'interaction_type': 'hub_service',
                    'description': 'Uses both shared memory (local IPC) and network ports (remote communication)',
                    'confidence': 0.8
                })
            elif ipc_summary.get('events', 0) > 0 and ipc_summary.get('shared_memory', 0) > 0:
                cross_validation['service_interactions'].append({
                    'service': executable,
                    'interaction_type': 'processing_service',
                    'description': 'Uses events and shared memory for synchronized processing',
                    'confidence': 0.7
                })
            elif network_summary.get('protocols_detected', []):
                protocols = network_summary['protocols_detected']
                if 'mqtt' in protocols:
                    cross_validation['service_interactions'].append({
                        'service': executable,
                        'interaction_type': 'cloud_service',
                        'description': 'Handles cloud communication via MQTT',
                        'confidence': 0.9
                    })
        
        # Analyze protocol consistency
        for executable in all_executables:
            network_summary = network_data['summary'].get(executable, {})
            protocols = network_summary.get('protocols_detected', [])
            
            if protocols:
                # Check for protocol consistency
                if 'http' in protocols and 'rest' in protocols:
                    cross_validation['protocol_consistency'][executable] = {
                        'status': 'consistent',
                        'reason': 'HTTP and REST protocols are complementary'
                    }
                elif 'tcp' in protocols and 'udp' in protocols:
                    cross_validation['protocol_consistency'][executable] = {
                        'status': 'mixed',
                        'reason': 'Both TCP and UDP suggest different communication needs'
                    }
        
        return cross_validation
    
    def infer_communication_role(self, executable: str, ipc_data: Dict, network_data: Dict) -> str:
        """Infer the communication role of an executable"""
        ipc_summary = ipc_data['summary'].get(executable, {})
        network_summary = network_data['summary'].get(executable, {})
        
        # Main UI application
        if executable == 'IntraoralScan.exe':
            return 'ui_coordinator'
        
        # Network service
        if executable == 'DentalNetwork.exe':
            return 'network_gateway'
        
        # Algorithm service with high shared memory usage
        if (executable == 'DentalAlgoService.exe' and 
            ipc_summary.get('shared_memory', 0) > 10):
            return 'processing_engine'
        
        # Scanning logic with mixed IPC/network
        if (executable == 'DentalScanAppLogic.exe' and 
            ipc_summary.get('total_endpoints', 0) > 5 and 
            network_summary.get('total_endpoints', 0) > 5):
            return 'scan_controller'
        
        # Default classification
        if ipc_summary.get('total_endpoints', 0) > network_summary.get('total_endpoints', 0):
            return 'local_service'
        elif network_summary.get('total_endpoints', 0) > 0:
            return 'network_service'
        else:
            return 'utility_service'
    
    def calculate_overall_confidence(self, ipc_validation: Dict, network_validation: Dict, cross_validation: Dict) -> Dict:
        """Calculate overall confidence scores for communication analysis"""
        confidence_scores = {
            'ipc_analysis_confidence': 0.0,
            'network_analysis_confidence': 0.0,
            'cross_validation_confidence': 0.0,
            'overall_confidence': 0.0
        }
        
        # IPC analysis confidence
        total_ipc_consistent = len(ipc_validation['consistent_findings'])
        total_ipc_inconsistent = len(ipc_validation['inconsistent_findings'])
        total_ipc_findings = total_ipc_consistent + total_ipc_inconsistent
        
        if total_ipc_findings > 0:
            confidence_scores['ipc_analysis_confidence'] = total_ipc_consistent / total_ipc_findings
        else:
            confidence_scores['ipc_analysis_confidence'] = 0.5  # Neutral if no findings
        
        # Network analysis confidence
        total_network_consistent = len(network_validation['consistent_findings'])
        total_network_inconsistent = len(network_validation['inconsistent_findings'])
        total_network_findings = total_network_consistent + total_network_inconsistent
        
        if total_network_findings > 0:
            confidence_scores['network_analysis_confidence'] = total_network_consistent / total_network_findings
        else:
            confidence_scores['network_analysis_confidence'] = 0.5
        
        # Cross-validation confidence
        total_interactions = len(cross_validation['service_interactions'])
        high_confidence_interactions = len([i for i in cross_validation['service_interactions'] if i['confidence'] > 0.7])
        
        if total_interactions > 0:
            confidence_scores['cross_validation_confidence'] = high_confidence_interactions / total_interactions
        else:
            confidence_scores['cross_validation_confidence'] = 0.5
        
        # Overall confidence (weighted average)
        confidence_scores['overall_confidence'] = (
            confidence_scores['ipc_analysis_confidence'] * 0.3 +
            confidence_scores['network_analysis_confidence'] * 0.3 +
            confidence_scores['cross_validation_confidence'] * 0.4
        )
        
        return confidence_scores
    
    def generate_recommendations(self, validation_results: Dict) -> List[str]:
        """Generate recommendations based on validation results"""
        recommendations = []
        
        # IPC recommendations
        ipc_missing = validation_results['ipc_validation']['missing_expected']
        if ipc_missing:
            recommendations.append(
                f"Consider deeper analysis of {len(ipc_missing)} executables with missing expected IPC mechanisms"
            )
        
        # Network recommendations
        network_missing = validation_results['network_validation']['missing_expected']
        if network_missing:
            recommendations.append(
                f"Investigate {len(network_missing)} network services with missing expected protocols"
            )
        
        # Cross-validation recommendations
        service_interactions = validation_results['cross_validation']['service_interactions']
        hub_services = [s for s in service_interactions if s['interaction_type'] == 'hub_service']
        if len(hub_services) > 1:
            recommendations.append(
                "Multiple hub services detected - verify service coordination architecture"
            )
        
        # Confidence-based recommendations
        overall_confidence = validation_results['confidence_scores']['overall_confidence']
        if overall_confidence < 0.7:
            recommendations.append(
                f"Overall confidence ({overall_confidence:.2f}) is below threshold - consider additional validation methods"
            )
        
        if overall_confidence > 0.8:
            recommendations.append(
                "High confidence in communication analysis - proceed with pipeline reconstruction"
            )
        
        return recommendations
    
    def run_validation_checkpoint(self) -> Dict:
        """Run complete communication validation checkpoint"""
        print("=== COMMUNICATION VALIDATION CHECKPOINT ===")
        
        # Load all analysis data
        print("Loading analysis data...")
        ipc_data = self.load_ipc_findings()
        network_data = self.load_network_findings()
        config_data = self.load_config_data()
        dependency_data = self.load_dependency_data()
        
        print(f"Loaded IPC data for {len(ipc_data['summary'])} executables")
        print(f"Loaded network data for {len(network_data['summary'])} executables")
        
        # Run validations
        print("Validating IPC consistency...")
        ipc_validation = self.validate_ipc_consistency(ipc_data, config_data)
        
        print("Validating network consistency...")
        network_validation = self.validate_network_consistency(network_data, config_data)
        
        print("Cross-validating communication patterns...")
        cross_validation = self.cross_validate_communication_patterns(ipc_data, network_data, dependency_data)
        
        # Calculate confidence scores
        print("Calculating confidence scores...")
        confidence_scores = self.calculate_overall_confidence(ipc_validation, network_validation, cross_validation)
        
        # Compile results
        self.validation_results = {
            'ipc_validation': ipc_validation,
            'network_validation': network_validation,
            'cross_validation': cross_validation,
            'confidence_scores': confidence_scores,
            'recommendations': self.generate_recommendations({
                'ipc_validation': ipc_validation,
                'network_validation': network_validation,
                'cross_validation': cross_validation,
                'confidence_scores': confidence_scores
            })
        }
        
        return self.validation_results
    
    def export_validation_report(self, output_file: str = "analysis_output/communication_validation.json"):
        """Export validation results to JSON report"""
        # Ensure output directory exists
        Path(output_file).parent.mkdir(exist_ok=True)
        
        # Add metadata
        export_data = {
            'validation_timestamp': datetime.now().isoformat(),
            'validation_results': self.validation_results
        }
        
        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        return export_data
    
    def print_validation_summary(self):
        """Print a human-readable validation summary"""
        results = self.validation_results
        
        print("\n=== VALIDATION SUMMARY ===")
        
        # IPC validation summary
        ipc_consistent = len(results['ipc_validation']['consistent_findings'])
        ipc_missing = len(results['ipc_validation']['missing_expected'])
        print(f"IPC Validation: {ipc_consistent} consistent findings, {ipc_missing} missing expected")
        
        # Network validation summary
        network_consistent = len(results['network_validation']['consistent_findings'])
        network_missing = len(results['network_validation']['missing_expected'])
        print(f"Network Validation: {network_consistent} consistent findings, {network_missing} missing expected")
        
        # Cross-validation summary
        service_interactions = len(results['cross_validation']['service_interactions'])
        print(f"Service Interactions: {service_interactions} identified patterns")
        
        # Confidence scores
        confidence = results['confidence_scores']
        print(f"\nConfidence Scores:")
        print(f"  IPC Analysis: {confidence['ipc_analysis_confidence']:.2f}")
        print(f"  Network Analysis: {confidence['network_analysis_confidence']:.2f}")
        print(f"  Cross-Validation: {confidence['cross_validation_confidence']:.2f}")
        print(f"  Overall: {confidence['overall_confidence']:.2f}")
        
        # Recommendations
        print(f"\nRecommendations:")
        for i, rec in enumerate(results['recommendations'], 1):
            print(f"  {i}. {rec}")
        
        # Communication matrix summary
        print(f"\nCommunication Roles:")
        for executable, info in results['cross_validation']['communication_matrix'].items():
            role = info['communication_role']
            ipc_endpoints = info['ipc_capabilities'].get('total_endpoints', 0)
            network_endpoints = info['network_capabilities'].get('total_endpoints', 0)
            print(f"  {executable}: {role} (IPC: {ipc_endpoints}, Network: {network_endpoints})")

if __name__ == "__main__":
    validator = CommunicationValidator()
    
    # Run validation checkpoint
    results = validator.run_validation_checkpoint()
    
    # Export results
    export_data = validator.export_validation_report()
    print(f"\nExported validation report to analysis_output/communication_validation.json")
    
    # Print summary
    validator.print_validation_summary()