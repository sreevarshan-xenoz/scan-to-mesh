#!/usr/bin/env python3
"""
Network Endpoint Analyzer
Analyzes network endpoints using configuration data and string analysis
"""
import sqlite3
import json
import re
import os
import subprocess
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional

class NetworkAnalyzer:
    def __init__(self, db_path="analysis_results.db"):
        self.db_path = db_path
        self.init_database()
        
        # Network pattern definitions
        self.network_patterns = {
            'urls': [
                r'https?://([a-zA-Z0-9\-\.]+(?:\.[a-zA-Z]{2,})?(?::\d+)?(?:/[^\s]*)?)',
                r'ws://([a-zA-Z0-9\-\.]+(?:\.[a-zA-Z]{2,})?(?::\d+)?(?:/[^\s]*)?)',
                r'wss://([a-zA-Z0-9\-\.]+(?:\.[a-zA-Z]{2,})?(?::\d+)?(?:/[^\s]*)?)',
            ],
            'ip_addresses': [
                r'(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}):(\d+)',
                r'(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})',
            ],
            'ports': [
                r'port[:\s]*(\d+)',
                r'listen[:\s]*(\d+)',
                r'bind[:\s]*(\d+)',
                r':(\d{4,5})',  # Common port format
            ],
            'hostnames': [
                r'localhost:(\d+)',
                r'127\.0\.0\.1:(\d+)',
                r'0\.0\.0\.0:(\d+)',
                r'([a-zA-Z0-9\-]+\.(?:com|net|org|io|cn|local)):(\d+)',
            ]
        }
        
        # Protocol keywords
        self.protocol_keywords = {
            'http': ['http://', 'https://', 'HttpClient', 'HttpRequest', 'HttpResponse', 'GET ', 'POST ', 'PUT ', 'DELETE '],
            'websocket': ['ws://', 'wss://', 'WebSocket', 'websocket', 'socket.io'],
            'tcp': ['TCP', 'tcp', 'socket', 'connect', 'bind', 'listen', 'accept'],
            'udp': ['UDP', 'udp', 'sendto', 'recvfrom'],
            'mqtt': ['mqtt', 'MQTT', 'publish', 'subscribe', 'broker'],
            'grpc': ['grpc', 'gRPC', 'protobuf', '.proto'],
            'rest': ['REST', 'api/', '/api', 'json', 'application/json'],
            'soap': ['SOAP', 'soap', 'wsdl', 'xml']
        }
    
    def init_database(self):
        """Initialize database tables for network endpoint data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create network endpoints table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS network_endpoints (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                executable_id INTEGER,
                endpoint_type TEXT NOT NULL, -- 'url', 'ip_address', 'port', 'hostname'
                endpoint_value TEXT NOT NULL,
                protocol TEXT, -- 'http', 'https', 'tcp', 'udp', 'websocket', etc.
                port INTEGER,
                confidence_score REAL,
                detection_method TEXT, -- 'config_analysis', 'string_analysis', 'cross_validation'
                context TEXT,
                analysis_timestamp TEXT,
                FOREIGN KEY (executable_id) REFERENCES executables (id)
            )
        ''')
        
        # Create network analysis summary table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS network_analysis (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                executable_id INTEGER,
                total_endpoints INTEGER,
                urls INTEGER,
                ip_addresses INTEGER,
                ports INTEGER,
                hostnames INTEGER,
                protocols_detected TEXT, -- JSON array of detected protocols
                analysis_method TEXT,
                analysis_success BOOLEAN,
                error_message TEXT,
                analysis_timestamp TEXT,
                FOREIGN KEY (executable_id) REFERENCES executables (id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def load_config_endpoints(self) -> Dict:
        """Load network endpoints from configuration analysis"""
        config_endpoints = {
            'main_tunnel': {},
            'http_sockets': [],
            'cloud_config': {},
            'service_ports': {}
        }
        
        try:
            # Load from config analyzer results
            if os.path.exists('analysis_output/architecture_analysis.json'):
                with open('analysis_output/architecture_analysis.json', 'r') as f:
                    config_data = json.load(f)
                
                # Extract main tunnel info
                tunnel = config_data.get('communication_channels', {}).get('main_tunnel', {})
                config_endpoints['main_tunnel'] = {
                    'host': tunnel.get('host', 'localhost'),
                    'port': tunnel.get('port', 18830),
                    'channel': tunnel.get('channel', 'Dental'),
                    'service_name': tunnel.get('service_name', 'DentalHub'),
                    'shared_memory': tunnel.get('shared_memory', 'DentalShared')
                }
                
                # Extract HTTP sockets
                http_sockets = config_data.get('network_endpoints', {}).get('http_sockets', [])
                config_endpoints['http_sockets'] = http_sockets
                
                # Extract cloud config
                cloud_config = config_data.get('network_endpoints', {}).get('cloud_config', {})
                config_endpoints['cloud_config'] = cloud_config
                
            else:
                # Fallback: analyze config files directly
                config_endpoints = self.analyze_config_files_direct()
                
        except Exception as e:
            print(f"Error loading config endpoints: {e}")
            config_endpoints = self.analyze_config_files_direct()
        
        return config_endpoints
    
    def analyze_config_files_direct(self) -> Dict:
        """Direct analysis of configuration files as fallback"""
        config_endpoints = {
            'main_tunnel': {},
            'http_sockets': [],
            'cloud_config': {},
            'service_ports': {}
        }
        
        # Analyze main system config
        main_cfg_path = "IntraoralScan/Bin/config/IntraoralScan/SystemCfg.json"
        if os.path.exists(main_cfg_path):
            try:
                with open(main_cfg_path, 'r') as f:
                    main_cfg = json.load(f)
                
                tunnel_info = main_cfg.get('Tunnel', {})
                config_endpoints['main_tunnel'] = {
                    'host': tunnel_info.get('HostAddress', 'localhost'),
                    'port': tunnel_info.get('Port', 18830),
                    'channel': tunnel_info.get('Channel', 'Dental'),
                    'service_name': tunnel_info.get('ServiceName', 'DentalHub')
                }
                
                server_info = main_cfg.get('Server', {})
                config_endpoints['http_sockets'] = server_info.get('httpSocket', [])
                
            except Exception as e:
                print(f"Error parsing main config: {e}")
        
        return config_endpoints
    
    def extract_strings_for_network(self, binary_path: str) -> List[str]:
        """Extract strings from binary focusing on network-related content"""
        try:
            # Use strings command with focus on network patterns
            result = subprocess.run([
                'strings', '-n', '6', binary_path
            ], capture_output=True, text=True, timeout=120)
            
            if result.returncode == 0:
                all_strings = result.stdout.split('\n')
                # Filter for network-related strings
                network_strings = []
                for string in all_strings:
                    if any(keyword in string.lower() for keyword in 
                          ['http', 'tcp', 'udp', 'socket', 'port', 'localhost', 'server', 'client', 'api', 'url']):
                        network_strings.append(string)
                return network_strings
            else:
                return self.extract_strings_manual_network(binary_path)
                
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return self.extract_strings_manual_network(binary_path)
    
    def extract_strings_manual_network(self, binary_path: str) -> List[str]:
        """Manual network-focused string extraction"""
        network_strings = []
        try:
            with open(binary_path, 'rb') as f:
                data = f.read()
            
            # Extract ASCII strings with network keywords
            current_string = ""
            for byte in data:
                if 32 <= byte <= 126:  # Printable ASCII
                    current_string += chr(byte)
                else:
                    if len(current_string) >= 6:
                        # Check if string contains network-related keywords
                        if any(keyword in current_string.lower() for keyword in 
                              ['http', 'tcp', 'udp', 'socket', 'port', 'localhost', 'server', 'api']):
                            network_strings.append(current_string)
                    current_string = ""
            
            # Don't forget the last string
            if len(current_string) >= 6:
                if any(keyword in current_string.lower() for keyword in 
                      ['http', 'tcp', 'udp', 'socket', 'port', 'localhost', 'server', 'api']):
                    network_strings.append(current_string)
                    
        except Exception as e:
            print(f"Manual network string extraction failed for {binary_path}: {e}")
        
        return network_strings
    
    def analyze_strings_for_network(self, strings: List[str]) -> Dict[str, List[Dict]]:
        """Analyze strings for network endpoint patterns"""
        network_findings = {
            'urls': [],
            'ip_addresses': [],
            'ports': [],
            'hostnames': []
        }
        
        for string in strings:
            # Check each pattern type
            for endpoint_type, patterns in self.network_patterns.items():
                for pattern in patterns:
                    try:
                        matches = re.finditer(pattern, string, re.IGNORECASE)
                        for match in matches:
                            confidence = self.calculate_network_confidence(pattern, string, match)
                            
                            finding = {
                                'endpoint_value': match.group(0),
                                'extracted_part': match.group(1) if match.groups() else match.group(0),
                                'context': string.strip()[:200],
                                'confidence_score': confidence,
                                'pattern_used': pattern
                            }
                            
                            # Extract port if available
                            if len(match.groups()) >= 2 and match.group(2).isdigit():
                                finding['port'] = int(match.group(2))
                            
                            network_findings[endpoint_type].append(finding)
                    except re.error:
                        continue  # Skip invalid regex patterns
        
        return network_findings
    
    def detect_protocols(self, strings: List[str]) -> List[str]:
        """Detect network protocols from string analysis"""
        detected_protocols = set()
        
        for string in strings:
            string_lower = string.lower()
            for protocol, keywords in self.protocol_keywords.items():
                if any(keyword.lower() in string_lower for keyword in keywords):
                    detected_protocols.add(protocol)
        
        return list(detected_protocols)
    
    def calculate_network_confidence(self, pattern: str, context: str, match) -> float:
        """Calculate confidence score for network endpoint detection"""
        confidence = 0.6  # Base confidence
        
        # Higher confidence for well-formed URLs
        if 'http' in pattern:
            confidence += 0.2
        
        # Higher confidence for specific IP patterns
        if r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}' in pattern:
            confidence += 0.1
        
        # Higher confidence for common ports
        endpoint_value = match.group(0)
        if any(port in endpoint_value for port in ['80', '443', '8080', '3000', '18830']):
            confidence += 0.1
        
        # Context analysis
        context_lower = context.lower()
        if any(keyword in context_lower for keyword in ['server', 'client', 'api', 'endpoint', 'url']):
            confidence += 0.1
        
        # Lower confidence for very generic patterns
        if ':' in pattern and len(endpoint_value) < 5:
            confidence -= 0.2
        
        return min(1.0, max(0.1, confidence))
    
    def cross_validate_with_config(self, string_findings: Dict, config_endpoints: Dict) -> Dict:
        """Cross-validate string findings with configuration data"""
        validated_findings = {
            'validated_endpoints': [],
            'config_only_endpoints': [],
            'string_only_endpoints': []
        }
        
        # Add config endpoints with high confidence
        main_tunnel = config_endpoints.get('main_tunnel', {})
        if main_tunnel.get('host') and main_tunnel.get('port'):
            validated_findings['config_only_endpoints'].append({
                'endpoint_type': 'hostname',
                'endpoint_value': f"{main_tunnel['host']}:{main_tunnel['port']}",
                'protocol': 'tcp',
                'port': main_tunnel['port'],
                'confidence_score': 0.9,
                'detection_method': 'config_analysis',
                'context': f"Main IPC tunnel: {main_tunnel.get('channel', 'Unknown')}"
            })
        
        # Add HTTP sockets from config
        for socket in config_endpoints.get('http_sockets', []):
            if socket.startswith(':'):
                port = socket[1:]
                if port.isdigit():
                    validated_findings['config_only_endpoints'].append({
                        'endpoint_type': 'port',
                        'endpoint_value': socket,
                        'protocol': 'http',
                        'port': int(port),
                        'confidence_score': 0.9,
                        'detection_method': 'config_analysis',
                        'context': 'HTTP socket from system configuration'
                    })
        
        # Process string findings and check for validation
        for endpoint_type, findings in string_findings.items():
            for finding in findings:
                # Check if this finding matches config data
                is_validated = False
                
                if endpoint_type == 'ports':
                    # Check against config ports
                    try:
                        port_num = int(finding['extracted_part'])
                        if (port_num == main_tunnel.get('port') or 
                            f":{port_num}" in config_endpoints.get('http_sockets', [])):
                            finding['confidence_score'] = min(1.0, finding['confidence_score'] + 0.2)
                            finding['detection_method'] = 'cross_validation'
                            is_validated = True
                    except ValueError:
                        pass
                
                if endpoint_type == 'hostnames':
                    # Check against config hostnames
                    if main_tunnel.get('host', '').lower() in finding['endpoint_value'].lower():
                        finding['confidence_score'] = min(1.0, finding['confidence_score'] + 0.2)
                        finding['detection_method'] = 'cross_validation'
                        is_validated = True
                
                if is_validated:
                    validated_findings['validated_endpoints'].append(finding)
                else:
                    validated_findings['string_only_endpoints'].append(finding)
        
        return validated_findings
    
    def analyze_network_in_executable(self, executable_path: str) -> Dict:
        """Analyze network endpoints in a single executable"""
        print(f"Analyzing network endpoints in: {executable_path}")
        
        # Load configuration endpoints
        config_endpoints = self.load_config_endpoints()
        
        # Extract network-related strings
        strings = self.extract_strings_for_network(executable_path)
        print(f"Extracted {len(strings)} network-related strings")
        
        # Analyze strings for network patterns
        string_findings = self.analyze_strings_for_network(strings)
        
        # Detect protocols
        protocols = self.detect_protocols(strings)
        
        # Cross-validate with configuration
        validated_findings = self.cross_validate_with_config(string_findings, config_endpoints)
        
        # Combine all findings
        all_endpoints = (validated_findings['validated_endpoints'] + 
                        validated_findings['config_only_endpoints'] + 
                        validated_findings['string_only_endpoints'])
        
        # Calculate summary statistics
        summary = {
            'total_endpoints': len(all_endpoints),
            'urls': len([e for e in all_endpoints if e.get('endpoint_type') == 'url']),
            'ip_addresses': len([e for e in all_endpoints if e.get('endpoint_type') == 'ip_address']),
            'ports': len([e for e in all_endpoints if e.get('endpoint_type') == 'port']),
            'hostnames': len([e for e in all_endpoints if e.get('endpoint_type') == 'hostname']),
            'protocols_detected': protocols,
            'findings': all_endpoints,
            'validation_summary': {
                'validated_count': len(validated_findings['validated_endpoints']),
                'config_only_count': len(validated_findings['config_only_endpoints']),
                'string_only_count': len(validated_findings['string_only_endpoints'])
            }
        }
        
        return summary
    
    def store_network_analysis(self, executable_id: int, analysis_result: Dict, analysis_method: str = "hybrid_analysis"):
        """Store network analysis results in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        timestamp = datetime.now().isoformat()
        
        # Store summary
        cursor.execute('''
            INSERT INTO network_analysis (
                executable_id, total_endpoints, urls, ip_addresses,
                ports, hostnames, protocols_detected, analysis_method,
                analysis_success, analysis_timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            executable_id,
            analysis_result['total_endpoints'],
            analysis_result['urls'],
            analysis_result['ip_addresses'],
            analysis_result['ports'],
            analysis_result['hostnames'],
            json.dumps(analysis_result['protocols_detected']),
            analysis_method,
            True,
            timestamp
        ))
        
        # Store individual endpoints
        for finding in analysis_result['findings']:
            cursor.execute('''
                INSERT INTO network_endpoints (
                    executable_id, endpoint_type, endpoint_value,
                    protocol, port, confidence_score, detection_method,
                    context, analysis_timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                executable_id,
                finding.get('endpoint_type', 'unknown'),
                finding.get('endpoint_value', ''),
                finding.get('protocol', ''),
                finding.get('port'),
                finding.get('confidence_score', 0.5),
                finding.get('detection_method', analysis_method),
                finding.get('context', '')[:500],  # Limit context length
                timestamp
            ))
        
        conn.commit()
        conn.close()
    
    def get_executable_id(self, executable_name: str) -> Optional[int]:
        """Get executable ID from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT id FROM executables WHERE name = ?', (executable_name,))
        result = cursor.fetchone()
        
        conn.close()
        return result[0] if result else None
    
    def analyze_high_value_targets(self) -> Dict:
        """Analyze network endpoints in high-value target executables"""
        high_value_targets = [
            'IntraoralScan.exe',
            'DentalAlgoService.exe',
            'DentalScanAppLogic.exe',
            'DentalNetwork.exe',
            'DentalOrderDataMgr.exe'
        ]
        
        results = {}
        
        for target in high_value_targets:
            executable_path = f"IntraoralScan/Bin/{target}"
            
            if not os.path.exists(executable_path):
                print(f"Target not found: {executable_path}")
                continue
            
            # Get executable ID
            executable_id = self.get_executable_id(target)
            if not executable_id:
                print(f"Executable {target} not found in database")
                continue
            
            # Analyze network endpoints
            try:
                analysis_result = self.analyze_network_in_executable(executable_path)
                
                # Store results
                self.store_network_analysis(executable_id, analysis_result)
                
                results[target] = analysis_result
                print(f"Found {analysis_result['total_endpoints']} network endpoints in {target}")
                print(f"  Protocols: {', '.join(analysis_result['protocols_detected'])}")
                
            except Exception as e:
                print(f"Error analyzing {target}: {e}")
                # Store error in database
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO network_analysis (
                        executable_id, analysis_method, analysis_success,
                        error_message, analysis_timestamp
                    ) VALUES (?, ?, ?, ?, ?)
                ''', (executable_id, "hybrid_analysis", False, str(e), datetime.now().isoformat()))
                conn.commit()
                conn.close()
        
        return results
    
    def export_network_findings(self, output_file: str = "analysis_output/network_endpoints.json"):
        """Export network findings to JSON"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get all network analysis results
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
            ORDER BY na.total_endpoints DESC
        ''')
        
        summary = []
        for row in cursor.fetchall():
            protocols = json.loads(row[6]) if row[6] else []
            summary.append({
                'executable': row[0],
                'total_endpoints': row[1],
                'urls': row[2],
                'ip_addresses': row[3],
                'ports': row[4],
                'hostnames': row[5],
                'protocols_detected': protocols,
                'analysis_success': bool(row[7])
            })
        
        # Get detailed endpoints
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
        
        detailed_endpoints = {}
        for row in cursor.fetchall():
            executable = row[0]
            if executable not in detailed_endpoints:
                detailed_endpoints[executable] = []
            
            detailed_endpoints[executable].append({
                'type': row[1],
                'value': row[2],
                'protocol': row[3],
                'port': row[4],
                'confidence': row[5],
                'detection_method': row[6],
                'context': row[7]
            })
        
        export_data = {
            'summary': summary,
            'detailed_endpoints': detailed_endpoints,
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        # Ensure output directory exists
        Path(output_file).parent.mkdir(exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        conn.close()
        return export_data

if __name__ == "__main__":
    analyzer = NetworkAnalyzer()
    
    print("=== NETWORK ENDPOINT ANALYSIS ===")
    results = analyzer.analyze_high_value_targets()
    
    print(f"\n=== ANALYSIS COMPLETE ===")
    print(f"Analyzed {len(results)} executables")
    
    # Export results
    export_data = analyzer.export_network_findings()
    print(f"Exported network analysis to analysis_output/network_endpoints.json")
    
    # Print summary
    print("\n=== NETWORK SUMMARY ===")
    for item in export_data['summary']:
        if item['analysis_success'] and item['total_endpoints'] > 0:
            protocols = ', '.join(item['protocols_detected']) if item['protocols_detected'] else 'none'
            print(f"{item['executable']}: {item['total_endpoints']} endpoints "
                  f"(URLs: {item['urls']}, IPs: {item['ip_addresses']}, "
                  f"ports: {item['ports']}, hosts: {item['hostnames']}) "
                  f"[{protocols}]")