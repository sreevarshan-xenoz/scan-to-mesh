#!/usr/bin/env python3
"""
QML and Qt Resource Extractor for IntraoralScan Analysis
Implements task 4.2: Build QML and Qt resource extractor
"""
import os
import json
import sqlite3
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import re

class QMLQtExtractor:
    def __init__(self, db_path="analysis_results.db"):
        self.db_path = db_path
        self.init_database()
        self.confidence_scores = {}
        
    def init_database(self):
        """Initialize database tables for QML and Qt resource analysis"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create QML files table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS qml_files (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                executable_source TEXT,
                file_path TEXT,
                file_name TEXT,
                file_size INTEGER,
                extraction_method TEXT,
                ui_component_type TEXT,
                signal_slot_connections INTEGER,
                service_connections TEXT,
                confidence_score REAL,
                analysis_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create Qt resources table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS qt_resources (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                executable_source TEXT,
                resource_type TEXT,
                resource_path TEXT,
                resource_name TEXT,
                resource_size INTEGER,
                extraction_method TEXT,
                confidence_score REAL,
                analysis_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create UI structure table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ui_structure (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                executable_source TEXT,
                component_name TEXT,
                component_type TEXT,
                parent_component TEXT,
                properties TEXT,
                signals TEXT,
                slots TEXT,
                imports TEXT,
                confidence_score REAL,
                analysis_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def find_qt_executables(self) -> List[str]:
        """Find executables that likely contain Qt resources"""
        qt_executables = []
        bin_path = Path("IntraoralScan/Bin")
        
        if not bin_path.exists():
            return qt_executables
            
        for exe_file in bin_path.glob("*.exe"):
            # Check if executable likely contains Qt resources
            if self._has_qt_resources(exe_file):
                qt_executables.append(str(exe_file))
                
        return qt_executables
    
    def _has_qt_resources(self, exe_path: Path) -> bool:
        """Check if executable contains Qt resources using string analysis"""
        try:
            # Use strings command to check for Qt resource indicators
            result = subprocess.run(
                ['strings', str(exe_path)], 
                capture_output=True, 
                text=True, 
                timeout=30
            )
            
            if result.returncode != 0:
                return False
                
            content = result.stdout.lower()
            
            # Look for Qt resource indicators
            qt_indicators = [
                'qrc:/',
                '.qml',
                'qmldir',
                'qt_resource_data',
                'qt_resource_name',
                'qt_resource_struct'
            ]
            
            return any(indicator in content for indicator in qt_indicators)
            
        except (subprocess.TimeoutExpired, FileNotFoundError):
            # Fallback: check file size and name patterns
            return (exe_path.stat().st_size > 1024*1024 and  # > 1MB
                   any(pattern in exe_path.name.lower() for pattern in 
                       ['dental', 'scan', 'ui', 'app', 'logic']))
    
    def extract_with_rcc(self, exe_path: str) -> Tuple[List[Dict], float]:
        """Extract Qt resources using rcc tool"""
        extracted_resources = []
        confidence = 0.0
        
        try:
            # Try to find rcc tool
            rcc_paths = [
                'rcc',
                '/usr/bin/rcc',
                '/usr/local/bin/rcc',
                'C:\\Qt\\5.15.2\\msvc2019_64\\bin\\rcc.exe',
                'C:\\Qt\\Tools\\QtCreator\\bin\\rcc.exe'
            ]
            
            rcc_tool = None
            for rcc_path in rcc_paths:
                if shutil.which(rcc_path) or os.path.exists(rcc_path):
                    rcc_tool = rcc_path
                    break
            
            if not rcc_tool:
                return extracted_resources, 0.1  # Low confidence without rcc
            
            # Create temporary directory for extraction
            with tempfile.TemporaryDirectory() as temp_dir:
                # Try to extract resources (this may not work directly on executables)
                try:
                    result = subprocess.run([
                        rcc_tool, '--list', exe_path
                    ], capture_output=True, text=True, timeout=60)
                    
                    if result.returncode == 0:
                        # Parse resource list
                        for line in result.stdout.strip().split('\n'):
                            if line.strip():
                                resource_info = {
                                    'type': 'qt_resource',
                                    'path': line.strip(),
                                    'name': Path(line.strip()).name,
                                    'size': 0,
                                    'extraction_method': 'rcc_list'
                                }
                                extracted_resources.append(resource_info)
                        
                        confidence = 0.9  # High confidence with successful rcc extraction
                    
                except subprocess.TimeoutExpired:
                    confidence = 0.2
                    
        except Exception as e:
            confidence = 0.1
            
        return extracted_resources, confidence
    
    def extract_with_resource_hacker(self, exe_path: str) -> Tuple[List[Dict], float]:
        """Fallback: Extract resources using Resource Hacker approach"""
        extracted_resources = []
        confidence = 0.0
        
        try:
            # Use strings to find resource-like patterns
            result = subprocess.run(
                ['strings', '-n', '8', exe_path], 
                capture_output=True, 
                text=True, 
                timeout=60
            )
            
            if result.returncode != 0:
                return extracted_resources, 0.1
            
            content = result.stdout
            
            # Look for QML file patterns
            qml_patterns = [
                r'qrc:/[^"\s]+\.qml',
                r'/[^"\s]+\.qml',
                r'[A-Za-z][A-Za-z0-9_]+\.qml'
            ]
            
            qml_files = set()
            for pattern in qml_patterns:
                matches = re.findall(pattern, content)
                qml_files.update(matches)
            
            # Look for other Qt resource patterns
            resource_patterns = [
                r'qrc:/[^"\s]+\.(png|jpg|jpeg|svg|ico)',
                r'qrc:/[^"\s]+\.(js|css)',
                r'qrc:/[^"\s]+\.qmldir'
            ]
            
            other_resources = set()
            for pattern in resource_patterns:
                matches = re.findall(pattern, content)
                other_resources.update([match[0] if isinstance(match, tuple) else match 
                                      for match in matches])
            
            # Add QML files
            for qml_file in qml_files:
                resource_info = {
                    'type': 'qml_file',
                    'path': qml_file,
                    'name': Path(qml_file).name if '/' in qml_file else qml_file,
                    'size': 0,
                    'extraction_method': 'string_analysis'
                }
                extracted_resources.append(resource_info)
            
            # Add other resources
            for resource in other_resources:
                resource_info = {
                    'type': 'qt_resource',
                    'path': resource,
                    'name': Path(resource).name if '/' in resource else resource,
                    'size': 0,
                    'extraction_method': 'string_analysis'
                }
                extracted_resources.append(resource_info)
            
            # Calculate confidence based on findings
            if len(extracted_resources) > 0:
                confidence = min(0.7, 0.3 + (len(extracted_resources) * 0.05))
            else:
                confidence = 0.2
                
        except Exception as e:
            confidence = 0.1
            
        return extracted_resources, confidence
    
    def analyze_qml_content(self, qml_content: str) -> Dict:
        """Analyze QML file content for UI structure and connections"""
        analysis = {
            'imports': [],
            'components': [],
            'signals': [],
            'slots': [],
            'properties': [],
            'service_connections': []
        }
        
        lines = qml_content.split('\n')
        
        for line in lines:
            line = line.strip()
            
            # Find imports
            if line.startswith('import '):
                import_match = re.match(r'import\s+([^\s]+)', line)
                if import_match:
                    analysis['imports'].append(import_match.group(1))
            
            # Find component definitions
            component_match = re.match(r'(\w+)\s*{', line)
            if component_match and component_match.group(1)[0].isupper():
                analysis['components'].append(component_match.group(1))
            
            # Find signals
            if 'signal ' in line:
                signal_match = re.search(r'signal\s+(\w+)', line)
                if signal_match:
                    analysis['signals'].append(signal_match.group(1))
            
            # Find slots/functions
            if 'function ' in line:
                function_match = re.search(r'function\s+(\w+)', line)
                if function_match:
                    analysis['slots'].append(function_match.group(1))
            
            # Find properties
            if 'property ' in line:
                prop_match = re.search(r'property\s+\w+\s+(\w+)', line)
                if prop_match:
                    analysis['properties'].append(prop_match.group(1))
            
            # Look for service connections (Qt.createQmlObject, Connections, etc.)
            if any(keyword in line.lower() for keyword in 
                   ['connections', 'createqmlobject', 'service', 'backend']):
                analysis['service_connections'].append(line.strip())
        
        return analysis
    
    def extract_qml_from_directory(self) -> List[Dict]:
        """Extract QML files from known directories"""
        qml_files = []
        
        # Check common QML directories
        qml_dirs = [
            "IntraoralScan/Bin/QtQuick",
            "IntraoralScan/Bin/QtQuick.2",
            "IntraoralScan/Bin/QuickControls",
            "IntraoralScan/Bin/DentalQuickControls"
        ]
        
        for qml_dir in qml_dirs:
            qml_path = Path(qml_dir)
            if qml_path.exists():
                for qml_file in qml_path.rglob("*.qml"):
                    try:
                        with open(qml_file, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                        
                        analysis = self.analyze_qml_content(content)
                        
                        qml_info = {
                            'source': 'directory_scan',
                            'path': str(qml_file),
                            'name': qml_file.name,
                            'size': qml_file.stat().st_size,
                            'analysis': analysis,
                            'extraction_method': 'direct_file_access',
                            'confidence': 1.0
                        }
                        qml_files.append(qml_info)
                        
                    except Exception as e:
                        # Still record the file even if we can't read it
                        qml_info = {
                            'source': 'directory_scan',
                            'path': str(qml_file),
                            'name': qml_file.name,
                            'size': qml_file.stat().st_size,
                            'analysis': {},
                            'extraction_method': 'direct_file_access',
                            'confidence': 0.5,
                            'error': str(e)
                        }
                        qml_files.append(qml_info)
        
        return qml_files
    
    def store_results(self, executable: str, resources: List[Dict], qml_files: List[Dict], confidence: float):
        """Store extraction results in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Store Qt resources
        for resource in resources:
            cursor.execute('''
                INSERT INTO qt_resources 
                (executable_source, resource_type, resource_path, resource_name, 
                 resource_size, extraction_method, confidence_score)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                executable,
                resource.get('type', 'unknown'),
                resource.get('path', ''),
                resource.get('name', ''),
                resource.get('size', 0),
                resource.get('extraction_method', ''),
                confidence
            ))
        
        # Store QML files
        for qml_file in qml_files:
            analysis = qml_file.get('analysis', {})
            
            cursor.execute('''
                INSERT INTO qml_files 
                (executable_source, file_path, file_name, file_size, 
                 extraction_method, ui_component_type, signal_slot_connections,
                 service_connections, confidence_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                qml_file.get('source', executable),
                qml_file.get('path', ''),
                qml_file.get('name', ''),
                qml_file.get('size', 0),
                qml_file.get('extraction_method', ''),
                ','.join(analysis.get('components', [])),
                len(analysis.get('signals', [])) + len(analysis.get('slots', [])),
                json.dumps(analysis.get('service_connections', [])),
                qml_file.get('confidence', confidence)
            ))
            
            # Store detailed UI structure
            for component in analysis.get('components', []):
                cursor.execute('''
                    INSERT INTO ui_structure 
                    (executable_source, component_name, component_type, 
                     properties, signals, slots, imports, confidence_score)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    qml_file.get('source', executable),
                    component,
                    'QML_Component',
                    json.dumps(analysis.get('properties', [])),
                    json.dumps(analysis.get('signals', [])),
                    json.dumps(analysis.get('slots', [])),
                    json.dumps(analysis.get('imports', [])),
                    qml_file.get('confidence', confidence)
                ))
        
        conn.commit()
        conn.close()
    
    def run_extraction(self) -> Dict:
        """Run complete QML and Qt resource extraction"""
        results = {
            'executables_analyzed': 0,
            'resources_extracted': 0,
            'qml_files_found': 0,
            'average_confidence': 0.0,
            'extraction_methods': {},
            'ui_components_identified': 0
        }
        
        # Find Qt executables
        qt_executables = self.find_qt_executables()
        results['executables_analyzed'] = len(qt_executables)
        
        total_confidence = 0.0
        total_resources = 0
        
        # Extract from executables
        for exe_path in qt_executables:
            print(f"Analyzing {exe_path}...")
            
            # Try rcc extraction first
            resources, confidence = self.extract_with_rcc(exe_path)
            
            if confidence < 0.5:
                # Fallback to Resource Hacker approach
                resources, confidence = self.extract_with_resource_hacker(exe_path)
            
            # Store results
            self.store_results(exe_path, resources, [], confidence)
            
            total_resources += len(resources)
            total_confidence += confidence
            
            method = resources[0]['extraction_method'] if resources else 'none'
            results['extraction_methods'][method] = results['extraction_methods'].get(method, 0) + 1
        
        # Extract QML files from directories
        qml_files = self.extract_qml_from_directory()
        results['qml_files_found'] = len(qml_files)
        
        # Store QML files
        if qml_files:
            self.store_results('directory_scan', [], qml_files, 1.0)
            total_confidence += len(qml_files) * 1.0
            
            # Count UI components
            for qml_file in qml_files:
                analysis = qml_file.get('analysis', {})
                results['ui_components_identified'] += len(analysis.get('components', []))
        
        results['resources_extracted'] = total_resources
        
        # Calculate average confidence
        total_items = len(qt_executables) + len(qml_files)
        if total_items > 0:
            results['average_confidence'] = total_confidence / total_items
        
        return results
    
    def generate_report(self) -> Dict:
        """Generate QML and Qt resource analysis report"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        report = {
            'summary': {},
            'qml_files': [],
            'qt_resources': [],
            'ui_structure': [],
            'extraction_methods': {},
            'confidence_analysis': {}
        }
        
        # Get summary statistics
        cursor.execute('SELECT COUNT(*) FROM qml_files')
        qml_count = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM qt_resources')
        resource_count = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM ui_structure')
        ui_component_count = cursor.fetchone()[0]
        
        cursor.execute('SELECT AVG(confidence_score) FROM qml_files')
        avg_qml_confidence = cursor.fetchone()[0] or 0.0
        
        cursor.execute('SELECT AVG(confidence_score) FROM qt_resources')
        avg_resource_confidence = cursor.fetchone()[0] or 0.0
        
        report['summary'] = {
            'qml_files_found': qml_count,
            'qt_resources_found': resource_count,
            'ui_components_identified': ui_component_count,
            'average_qml_confidence': round(avg_qml_confidence, 2),
            'average_resource_confidence': round(avg_resource_confidence, 2)
        }
        
        # Get detailed QML files
        cursor.execute('''
            SELECT file_name, file_path, file_size, ui_component_type, 
                   signal_slot_connections, extraction_method, confidence_score
            FROM qml_files ORDER BY confidence_score DESC, file_size DESC
        ''')
        
        for row in cursor.fetchall():
            report['qml_files'].append({
                'name': row[0],
                'path': row[1],
                'size_kb': round(row[2] / 1024, 1) if row[2] else 0,
                'components': row[3].split(',') if row[3] else [],
                'signal_slot_count': row[4],
                'extraction_method': row[5],
                'confidence': row[6]
            })
        
        # Get Qt resources
        cursor.execute('''
            SELECT resource_name, resource_path, resource_type, resource_size,
                   extraction_method, confidence_score
            FROM qt_resources ORDER BY confidence_score DESC
        ''')
        
        for row in cursor.fetchall():
            report['qt_resources'].append({
                'name': row[0],
                'path': row[1],
                'type': row[2],
                'size_kb': round(row[3] / 1024, 1) if row[3] else 0,
                'extraction_method': row[4],
                'confidence': row[5]
            })
        
        # Get extraction method statistics
        cursor.execute('''
            SELECT extraction_method, COUNT(*) as count
            FROM (
                SELECT extraction_method FROM qml_files
                UNION ALL
                SELECT extraction_method FROM qt_resources
            ) GROUP BY extraction_method
        ''')
        
        for row in cursor.fetchall():
            report['extraction_methods'][row[0]] = row[1]
        
        conn.close()
        return report

if __name__ == "__main__":
    extractor = QMLQtExtractor()
    
    print("=== QML AND QT RESOURCE EXTRACTION ===")
    print("Starting extraction process...")
    
    # Run extraction
    results = extractor.run_extraction()
    
    print(f"\nExtraction Results:")
    print(f"- Executables analyzed: {results['executables_analyzed']}")
    print(f"- Qt resources extracted: {results['resources_extracted']}")
    print(f"- QML files found: {results['qml_files_found']}")
    print(f"- UI components identified: {results['ui_components_identified']}")
    print(f"- Average confidence: {results['average_confidence']:.2f}")
    
    print(f"\nExtraction methods used:")
    for method, count in results['extraction_methods'].items():
        print(f"- {method}: {count} items")
    
    # Generate detailed report
    report = extractor.generate_report()
    
    # Save report
    os.makedirs('analysis_output', exist_ok=True)
    with open('analysis_output/qml_qt_analysis.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n=== DETAILED ANALYSIS REPORT ===")
    print(f"QML Files Found: {report['summary']['qml_files_found']}")
    print(f"Qt Resources Found: {report['summary']['qt_resources_found']}")
    print(f"UI Components: {report['summary']['ui_components_identified']}")
    
    if report['qml_files']:
        print(f"\nTop QML Files:")
        for qml_file in report['qml_files'][:5]:
            print(f"- {qml_file['name']} ({qml_file['size_kb']} KB, "
                  f"{len(qml_file['components'])} components, "
                  f"confidence: {qml_file['confidence']:.2f})")
    
    print(f"\nDetailed report saved to: analysis_output/qml_qt_analysis.json")
    print("=== QML/QT EXTRACTION COMPLETE ===")