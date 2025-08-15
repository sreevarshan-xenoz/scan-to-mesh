#!/usr/bin/env python3
"""
IPC Endpoint Detection System
Detects named pipes, shared memory, sockets, and other IPC mechanisms
"""
import sqlite3
import re
import subprocess
import json
import os
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional

class IPCDetector:
    def __init__(self, db_path="analysis_results.db"):
        self.db_path = db_path
        self.init_database()
        
        # IPC pattern definitions - using simple string matching to avoid regex issues
        self.ipc_keywords = {
            'named_pipes': [
                'pipe\\',
                'CreateNamedPipe',
                'ConnectNamedPipe',
                'DisconnectNamedPipe',
                'PeekNamedPipe',
                'SetNamedPipeHandleState',
                'WaitNamedPipe'
            ],
            'shared_memory': [
                'CreateFileMapping',
                'OpenFileMapping',
                'MapViewOfFile',
                'UnmapViewOfFile',
                'Global\\',
                'Local\\',
                'shared_memory',
                'SharedMemory'
            ],
            'sockets': [
                'WSAStartup',
                'socket(',
                'bind(',
                'listen(',
                'accept(',
                'connect(',
                'send(',
                'recv(',
                'localhost:',
                '127.0.0.1:',
                'gethostbyname'
            ],
            'message_queues': [
                'CreateMailslot',
                'PostMessage',
                'SendMessage',
                'GetMessage',
                'PeekMessage',
                'message_queue',
                'MessageQueue'
            ],
            'events': [
                'CreateEvent',
                'OpenEvent',
                'SetEvent',
                'ResetEvent',
                'PulseEvent',
                'WaitForSingleObject',
                'WaitForMultipleObjects'
            ]
        }
    
    def init_database(self):
        """Initialize database tables for IPC data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create IPC endpoints table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ipc_endpoints (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                executable_id INTEGER,
                endpoint_type TEXT NOT NULL, -- 'named_pipe', 'shared_memory', 'socket', 'message_queue', 'event'
                endpoint_name TEXT NOT NULL,
                direction TEXT, -- 'producer', 'consumer', 'bidirectional'
                confidence_score REAL, -- 0.0 to 1.0
                detection_method TEXT, -- 'string_analysis', 'process_monitor', 'manual'
                context TEXT, -- surrounding code context
                analysis_timestamp TEXT,
                FOREIGN KEY (executable_id) REFERENCES executables (id)
            )
        ''')
        
        # Create IPC analysis summary table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ipc_analysis (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                executable_id INTEGER,
                total_endpoints INTEGER,
                named_pipes INTEGER,
                shared_memory INTEGER,
                sockets INTEGER,
                message_queues INTEGER,
                events INTEGER,
                analysis_method TEXT,
                analysis_success BOOLEAN,
                error_message TEXT,
                analysis_timestamp TEXT,
                FOREIGN KEY (executable_id) REFERENCES executables (id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def extract_strings_floss(self, binary_path: str) -> List[str]:
        """Extract strings using FLOSS (FLARE Obfuscated String Solver)"""
        try:
            # Try FLOSS first (better for obfuscated binaries)
            result = subprocess.run([
                'floss', binary_path, '--no-static-strings', '--minimum-length', '4'
            ], capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                return result.stdout.split('\n')
            else:
                # Fallback to regular strings command
                return self.extract_strings_fallback(binary_path)
                
        except (subprocess.TimeoutExpired, FileNotFoundError):
            # FLOSS not available or timeout, use fallback
            return self.extract_strings_fallback(binary_path)
    
    def extract_strings_fallback(self, binary_path: str) -> List[str]:
        """Fallback string extraction using built-in strings command"""
        try:
            # Try Windows strings.exe if available
            result = subprocess.run([
                'strings', '-n', '4', binary_path
            ], capture_output=True, text=True, timeout=120)
            
            if result.returncode == 0:
                return result.stdout.split('\n')
            else:
                # Manual string extraction as last resort
                return self.extract_strings_manual(binary_path)
                
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return self.extract_strings_manual(binary_path)
    
    def extract_strings_manual(self, binary_path: str) -> List[str]:
        """Manual string extraction from binary file"""
        strings = []
        try:
            with open(binary_path, 'rb') as f:
                data = f.read()
                
            # Extract ASCII strings (minimum length 4)
            current_string = ""
            for byte in data:
                if 32 <= byte <= 126:  # Printable ASCII
                    current_string += chr(byte)
                else:
                    if len(current_string) >= 4:
                        strings.append(current_string)
                    current_string = ""
            
            # Don't forget the last string
            if len(current_string) >= 4:
                strings.append(current_string)
                
        except Exception as e:
            print(f"Manual string extraction failed for {binary_path}: {e}")
            
        return strings
    
    def analyze_strings_for_ipc(self, strings: List[str]) -> Dict[str, List[Dict]]:
        """Analyze strings for IPC patterns using keyword matching"""
        ipc_findings = {
            'named_pipes': [],
            'shared_memory': [],
            'sockets': [],
            'message_queues': [],
            'events': []
        }
        
        for string in strings:
            string_lower = string.lower()
            
            for ipc_type, keywords in self.ipc_keywords.items():
                for keyword in keywords:
                    if keyword.lower() in string_lower:
                        # Extract potential endpoint name from context
                        endpoint_name = self.extract_endpoint_name(string, keyword)
                        confidence = self.calculate_confidence_keyword(keyword, string)
                        
                        finding = {
                            'endpoint_name': endpoint_name,
                            'context': string.strip()[:200],  # Limit context length
                            'confidence_score': confidence,
                            'keyword_used': keyword
                        }
                        
                        ipc_findings[ipc_type].append(finding)
                        break  # Only match once per string per type
        
        return ipc_findings
    
    def extract_endpoint_name(self, context: str, keyword: str) -> str:
        """Extract endpoint name from context around keyword"""
        # Simple extraction - look for quoted strings or identifiers near keyword
        keyword_pos = context.lower().find(keyword.lower())
        if keyword_pos == -1:
            return keyword
        
        # Look for quoted strings after keyword
        after_keyword = context[keyword_pos + len(keyword):]
        
        # Find quoted strings
        for quote in ['"', "'"]:
            start = after_keyword.find(quote)
            if start != -1:
                end = after_keyword.find(quote, start + 1)
                if end != -1:
                    potential_name = after_keyword[start + 1:end]
                    if len(potential_name) > 2 and len(potential_name) < 100:
                        return potential_name
        
        # Look for identifiers (alphanumeric + underscore)
        import re
        match = re.search(r'[a-zA-Z_][a-zA-Z0-9_]{2,50}', after_keyword)
        if match:
            return match.group(0)
        
        return keyword
    
    def calculate_confidence_keyword(self, keyword: str, context: str) -> float:
        """Calculate confidence score for IPC detection based on keyword"""
        confidence = 0.5  # Base confidence
        
        # Higher confidence for specific API calls
        if any(api in keyword for api in ['CreateNamedPipe', 'CreateFileMapping', 'WSAStartup', 'CreateEvent']):
            confidence += 0.3
        
        # Higher confidence for function calls (contains parentheses)
        if '(' in keyword:
            confidence += 0.2
        
        # Lower confidence for very generic keywords
        if keyword.lower() in ['pipe', 'socket', 'event', 'message']:
            confidence -= 0.2
        
        # Context analysis
        context_lower = context.lower()
        if any(keyword in context_lower for keyword in ['ipc', 'communication', 'server', 'client', 'connect']):
            confidence += 0.1
        
        # Check for function-like context
        if any(char in context for char in ['(', ')', '{', '}', ';']):
            confidence += 0.1
        
        return min(1.0, max(0.1, confidence))
    
    def detect_ipc_in_executable(self, executable_path: str) -> Dict:
        """Detect IPC endpoints in a single executable"""
        print(f"Analyzing IPC endpoints in: {executable_path}")
        
        # Extract strings
        strings = self.extract_strings_floss(executable_path)
        print(f"Extracted {len(strings)} strings")
        
        # Analyze for IPC patterns
        ipc_findings = self.analyze_strings_for_ipc(strings)
        
        # Calculate summary statistics
        summary = {
            'total_endpoints': sum(len(findings) for findings in ipc_findings.values()),
            'named_pipes': len(ipc_findings['named_pipes']),
            'shared_memory': len(ipc_findings['shared_memory']),
            'sockets': len(ipc_findings['sockets']),
            'message_queues': len(ipc_findings['message_queues']),
            'events': len(ipc_findings['events']),
            'findings': ipc_findings
        }
        
        return summary
    
    def store_ipc_analysis(self, executable_id: int, analysis_result: Dict, analysis_method: str = "string_analysis"):
        """Store IPC analysis results in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        timestamp = datetime.now().isoformat()
        
        # Store summary
        cursor.execute('''
            INSERT INTO ipc_analysis (
                executable_id, total_endpoints, named_pipes, shared_memory,
                sockets, message_queues, events, analysis_method,
                analysis_success, analysis_timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            executable_id,
            analysis_result['total_endpoints'],
            analysis_result['named_pipes'],
            analysis_result['shared_memory'],
            analysis_result['sockets'],
            analysis_result['message_queues'],
            analysis_result['events'],
            analysis_method,
            True,
            timestamp
        ))
        
        # Store individual endpoints
        for ipc_type, findings in analysis_result['findings'].items():
            for finding in findings:
                cursor.execute('''
                    INSERT INTO ipc_endpoints (
                        executable_id, endpoint_type, endpoint_name,
                        confidence_score, detection_method, context, analysis_timestamp
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    executable_id,
                    ipc_type,
                    finding['endpoint_name'],
                    finding['confidence_score'],
                    analysis_method,
                    finding['context'][:500],  # Limit context length
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
        """Analyze IPC endpoints in high-value target executables"""
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
            
            # Analyze IPC endpoints
            try:
                analysis_result = self.detect_ipc_in_executable(executable_path)
                
                # Store results
                self.store_ipc_analysis(executable_id, analysis_result)
                
                results[target] = analysis_result
                print(f"Found {analysis_result['total_endpoints']} IPC endpoints in {target}")
                
            except Exception as e:
                print(f"Error analyzing {target}: {e}")
                # Store error in database
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO ipc_analysis (
                        executable_id, analysis_method, analysis_success,
                        error_message, analysis_timestamp
                    ) VALUES (?, ?, ?, ?, ?)
                ''', (executable_id, "string_analysis", False, str(e), datetime.now().isoformat()))
                conn.commit()
                conn.close()
        
        return results
    
    def export_ipc_findings(self, output_file: str = "analysis_output/ipc_endpoints.json"):
        """Export IPC findings to JSON"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get all IPC analysis results
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
            ORDER BY ia.total_endpoints DESC
        ''')
        
        summary = []
        for row in cursor.fetchall():
            summary.append({
                'executable': row[0],
                'total_endpoints': row[1],
                'named_pipes': row[2],
                'shared_memory': row[3],
                'sockets': row[4],
                'message_queues': row[5],
                'events': row[6],
                'analysis_success': bool(row[7])
            })
        
        # Get detailed endpoints
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
        
        detailed_endpoints = {}
        for row in cursor.fetchall():
            executable = row[0]
            if executable not in detailed_endpoints:
                detailed_endpoints[executable] = []
            
            detailed_endpoints[executable].append({
                'type': row[1],
                'name': row[2],
                'confidence': row[3],
                'context': row[4]
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
    detector = IPCDetector()
    
    print("=== IPC ENDPOINT DETECTION ===")
    results = detector.analyze_high_value_targets()
    
    print(f"\n=== ANALYSIS COMPLETE ===")
    print(f"Analyzed {len(results)} executables")
    
    # Export results
    export_data = detector.export_ipc_findings()
    print(f"Exported IPC analysis to analysis_output/ipc_endpoints.json")
    
    # Print summary
    print("\n=== IPC SUMMARY ===")
    for item in export_data['summary']:
        if item['analysis_success'] and item['total_endpoints'] > 0:
            print(f"{item['executable']}: {item['total_endpoints']} endpoints "
                  f"(pipes: {item['named_pipes']}, sockets: {item['sockets']}, "
                  f"shared_mem: {item['shared_memory']})")