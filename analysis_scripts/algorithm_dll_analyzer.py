#!/usr/bin/env python3
"""
Algorithm DLL Export Analyzer
Analyzes Sn3D*.dll and algorithm*.dll files for exported functions and signatures
"""

import os
import sys
import json
import sqlite3
import subprocess
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AlgorithmDLLAnalyzer:
    def __init__(self, base_path: str, db_path: str = "analysis_results.db"):
        self.base_path = Path(base_path)
        self.db_path = db_path
        self.init_database()
        
        # Algorithm DLL patterns to analyze
        self.dll_patterns = [
            "Sn3D*.dll",
            "algorithm*.dll"
        ]
        
        # Processing stage mappings based on naming patterns
        self.stage_mappings = {
            'calibration': ['Sn3DCalibrationJR.dll'],
            'acquisition': ['Sn3DRealtimeScan.dll', 'Sn3DPhotometricStereo.dll'],
            'registration': ['Sn3DRegistration.dll', 'Sn3DTextureBasedTrack.dll', 'Sn3DScanSlam.dll'],
            'fusion': ['Sn3DSpeckleFusion.dll', 'Sn3DSpeckle.dll', 'Sn3DPhaseBuild.dll'],
            'ai_analysis': ['Sn3DDentalAI.dll', 'Sn3DDentalRealTimeSemSeg.dll', 'Sn3DDentalBoxDet.dll', 
                           'Sn3DDentalOralCls.dll', 'Sn3DInfraredCariesDet.dll', 'Sn3DLandmarksDetForCli.dll'],
            'mesh_processing': ['Sn3DMagic.dll', 'Sn3DDraco.dll', 'Sn3DCork.dll', 'Sn3DTooling.dll'],
            'dental_specific': ['Sn3DDental.dll', 'Sn3DDentalDesktop.dll', 'Sn3DDentalOral.dll', 
                               'Sn3DCrownGen.dll', 'Sn3DOrthoEx.dll'],
            'visualization': ['Sn3DFace.dll', 'Sn3DFaceUnity.dll', 'Sn3DAIOutUnity.dll'],
            'algorithms': ['algorithm1.dll', 'algorithm2.dll', 'algorithmHlj.dll', 'algorithmLy.dll',
                          'algorithmLzy.dll', 'algorithmMigrate.dll', 'algorithmSaj.dll', 
                          'algorithmZbt.dll', 'algorithmZys.dll', 'algorithmlyc.dll']
        }
    
    def init_database(self):
        """Initialize SQLite database for storing analysis results"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create algorithm_dlls table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS algorithm_dlls (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                dll_name TEXT UNIQUE,
                file_path TEXT,
                file_size INTEGER,
                processing_stage TEXT,
                export_count INTEGER,
                analysis_timestamp TEXT,
                confidence_score REAL
            )
        ''')
        
        # Create dll_exports table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS dll_exports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                dll_name TEXT,
                function_name TEXT,
                ordinal INTEGER,
                signature_hint TEXT,
                parameter_hints TEXT,
                confidence_score REAL,
                FOREIGN KEY (dll_name) REFERENCES algorithm_dlls (dll_name)
            )
        ''')
        
        # Create dll_strings table for string analysis
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS dll_strings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                dll_name TEXT,
                string_value TEXT,
                string_type TEXT,
                relevance_score REAL,
                FOREIGN KEY (dll_name) REFERENCES algorithm_dlls (dll_name)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def find_algorithm_dlls(self) -> List[Path]:
        """Find all algorithm DLLs matching our patterns"""
        dll_files = []
        
        # Search in main Bin directory
        bin_path = self.base_path / "IntraoralScan" / "Bin"
        if bin_path.exists():
            for pattern in self.dll_patterns:
                dll_files.extend(bin_path.glob(pattern))
        
        # Also search in subdirectories for duplicates/variants
        for root, dirs, files in os.walk(bin_path):
            for file in files:
                if (file.startswith("Sn3D") and file.endswith(".dll")) or \
                   (file.startswith("algorithm") and file.endswith(".dll")):
                    dll_files.append(Path(root) / file)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_dlls = []
        for dll in dll_files:
            if dll.name not in seen:
                seen.add(dll.name)
                unique_dlls.append(dll)
        
        logger.info(f"Found {len(unique_dlls)} unique algorithm DLLs")
        return unique_dlls
    
    def get_processing_stage(self, dll_name: str) -> str:
        """Determine processing stage based on DLL name"""
        for stage, dlls in self.stage_mappings.items():
            if dll_name in dlls:
                return stage
        return "unknown"
    
    def analyze_dll_exports(self, dll_path: Path) -> Tuple[List[Dict], float]:
        """Analyze DLL exports using dumpbin or objdump"""
        exports = []
        confidence = 0.0
        
        try:
            # Try using dumpbin (Visual Studio tools)
            result = subprocess.run([
                'dumpbin', '/exports', str(dll_path)
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                exports, confidence = self._parse_dumpbin_output(result.stdout)
            else:
                # Fallback to objdump (MinGW/MSYS2)
                result = subprocess.run([
                    'objdump', '-p', str(dll_path)
                ], capture_output=True, text=True, timeout=30)
                
                if result.returncode == 0:
                    exports, confidence = self._parse_objdump_output(result.stdout)
                else:
                    logger.warning(f"Could not analyze exports for {dll_path.name}")
                    confidence = 0.1
        
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            logger.warning(f"Export analysis failed for {dll_path.name}: {e}")
            confidence = 0.1
        
        return exports, confidence
    
    def _parse_dumpbin_output(self, output: str) -> Tuple[List[Dict], float]:
        """Parse dumpbin /exports output"""
        exports = []
        confidence = 0.8
        
        # Look for export table section
        in_exports = False
        for line in output.split('\n'):
            line = line.strip()
            
            if 'ordinal hint RVA      name' in line:
                in_exports = True
                continue
            
            if in_exports and line:
                # Parse export line: ordinal hint RVA name
                parts = line.split()
                if len(parts) >= 4:
                    try:
                        ordinal = int(parts[0])
                        name = parts[-1]
                        
                        export_info = {
                            'function_name': name,
                            'ordinal': ordinal,
                            'signature_hint': self._infer_signature_from_name(name),
                            'parameter_hints': self._extract_parameter_hints(name)
                        }
                        exports.append(export_info)
                    except ValueError:
                        continue
        
        return exports, confidence
    
    def _parse_objdump_output(self, output: str) -> Tuple[List[Dict], float]:
        """Parse objdump -p output for exports"""
        exports = []
        confidence = 0.7
        
        # Look for export table
        in_exports = False
        for line in output.split('\n'):
            line = line.strip()
            
            if '[Ordinal/Name Pointer] Table' in line:
                in_exports = True
                continue
            
            if in_exports and line.startswith('['):
                # Parse export line
                match = re.search(r'\[(\d+)\]\s+(.+)', line)
                if match:
                    ordinal = int(match.group(1))
                    name = match.group(2).strip()
                    
                    export_info = {
                        'function_name': name,
                        'ordinal': ordinal,
                        'signature_hint': self._infer_signature_from_name(name),
                        'parameter_hints': self._extract_parameter_hints(name)
                    }
                    exports.append(export_info)
        
        return exports, confidence
    
    def _infer_signature_from_name(self, function_name: str) -> str:
        """Infer function signature hints from naming patterns"""
        name_lower = function_name.lower()
        
        # Common patterns in algorithm DLLs
        if 'init' in name_lower or 'initialize' in name_lower:
            return "initialization_function"
        elif 'process' in name_lower or 'compute' in name_lower:
            return "processing_function"
        elif 'get' in name_lower and ('result' in name_lower or 'output' in name_lower):
            return "result_getter"
        elif 'set' in name_lower and ('param' in name_lower or 'config' in name_lower):
            return "parameter_setter"
        elif 'create' in name_lower or 'new' in name_lower:
            return "constructor_function"
        elif 'destroy' in name_lower or 'delete' in name_lower or 'free' in name_lower:
            return "destructor_function"
        elif 'register' in name_lower or 'align' in name_lower:
            return "registration_function"
        elif 'fuse' in name_lower or 'merge' in name_lower:
            return "fusion_function"
        elif 'segment' in name_lower or 'classify' in name_lower:
            return "ai_analysis_function"
        else:
            return "unknown_function"
    
    def _extract_parameter_hints(self, function_name: str) -> str:
        """Extract parameter hints from function names"""
        hints = []
        name_lower = function_name.lower()
        
        # Look for common parameter patterns
        if 'point' in name_lower and 'cloud' in name_lower:
            hints.append("point_cloud_data")
        if 'mesh' in name_lower:
            hints.append("mesh_data")
        if 'image' in name_lower or 'frame' in name_lower:
            hints.append("image_data")
        if 'matrix' in name_lower or 'transform' in name_lower:
            hints.append("transformation_matrix")
        if 'param' in name_lower or 'config' in name_lower:
            hints.append("configuration_parameters")
        if 'result' in name_lower or 'output' in name_lower:
            hints.append("output_buffer")
        
        return ", ".join(hints) if hints else "unknown_parameters"
    
    def analyze_dll_strings(self, dll_path: Path) -> Tuple[List[Dict], float]:
        """Extract and analyze strings from DLL for additional insights"""
        strings_data = []
        confidence = 0.0
        
        try:
            # Use strings command to extract readable strings
            result = subprocess.run([
                'strings', '-n', '4', str(dll_path)
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                strings_data, confidence = self._analyze_extracted_strings(result.stdout)
            else:
                logger.warning(f"String extraction failed for {dll_path.name}")
                confidence = 0.1
        
        except (subprocess.TimeoutExpired, FileNotFoundError):
            # Fallback: try to read strings manually (basic implementation)
            try:
                with open(dll_path, 'rb') as f:
                    content = f.read()
                    strings_data, confidence = self._extract_strings_manually(content)
            except Exception as e:
                logger.warning(f"Manual string extraction failed for {dll_path.name}: {e}")
                confidence = 0.1
        
        return strings_data, confidence
    
    def _analyze_extracted_strings(self, strings_output: str) -> Tuple[List[Dict], float]:
        """Analyze extracted strings for relevance"""
        strings_data = []
        confidence = 0.6
        
        # Patterns that indicate algorithm-relevant strings
        relevant_patterns = [
            r'(?i)(point.*cloud|mesh|vertex|triangle)',
            r'(?i)(register|align|icp|slam)',
            r'(?i)(fuse|fusion|tsdf|voxel)',
            r'(?i)(segment|classify|detect|neural)',
            r'(?i)(calibrat|intrinsic|extrinsic)',
            r'(?i)(cuda|gpu|opencl)',
            r'(?i)(opencv|pcl|eigen)',
            r'(?i)(dental|tooth|oral|scan)',
            r'(?i)(error|warning|debug|log)',
            r'(?i)(param|config|setting)'
        ]
        
        for line in strings_output.split('\n'):
            line = line.strip()
            if len(line) < 4 or len(line) > 200:  # Filter reasonable string lengths
                continue
            
            relevance_score = 0.0
            string_type = "generic"
            
            # Check against relevant patterns
            for pattern in relevant_patterns:
                if re.search(pattern, line):
                    relevance_score += 0.2
                    if 'point.*cloud|mesh' in pattern:
                        string_type = "geometry"
                    elif 'register|align' in pattern:
                        string_type = "registration"
                    elif 'fuse|fusion' in pattern:
                        string_type = "fusion"
                    elif 'segment|classify' in pattern:
                        string_type = "ai_analysis"
                    elif 'error|warning' in pattern:
                        string_type = "logging"
                    elif 'param|config' in pattern:
                        string_type = "configuration"
            
            # Only store strings with some relevance
            if relevance_score > 0.1:
                strings_data.append({
                    'string_value': line,
                    'string_type': string_type,
                    'relevance_score': min(relevance_score, 1.0)
                })
        
        return strings_data, confidence
    
    def _extract_strings_manually(self, content: bytes) -> Tuple[List[Dict], float]:
        """Manual string extraction as fallback"""
        strings_data = []
        confidence = 0.3
        
        # Simple ASCII string extraction
        current_string = ""
        for byte in content:
            if 32 <= byte <= 126:  # Printable ASCII
                current_string += chr(byte)
            else:
                if len(current_string) >= 4:
                    strings_data.append({
                        'string_value': current_string,
                        'string_type': "generic",
                        'relevance_score': 0.1
                    })
                current_string = ""
        
        # Add final string if exists
        if len(current_string) >= 4:
            strings_data.append({
                'string_value': current_string,
                'string_type': "generic", 
                'relevance_score': 0.1
            })
        
        return strings_data[:100], confidence  # Limit to first 100 strings
    
    def store_analysis_results(self, dll_path: Path, exports: List[Dict], 
                             strings_data: List[Dict], confidence: float):
        """Store analysis results in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        dll_name = dll_path.name
        processing_stage = self.get_processing_stage(dll_name)
        
        try:
            # Store DLL info
            cursor.execute('''
                INSERT OR REPLACE INTO algorithm_dlls 
                (dll_name, file_path, file_size, processing_stage, export_count, 
                 analysis_timestamp, confidence_score)
                VALUES (?, ?, ?, ?, ?, datetime('now'), ?)
            ''', (dll_name, str(dll_path), dll_path.stat().st_size, 
                  processing_stage, len(exports), confidence))
            
            # Clear existing exports and strings for this DLL
            cursor.execute('DELETE FROM dll_exports WHERE dll_name = ?', (dll_name,))
            cursor.execute('DELETE FROM dll_strings WHERE dll_name = ?', (dll_name,))
            
            # Store exports
            for export in exports:
                cursor.execute('''
                    INSERT INTO dll_exports 
                    (dll_name, function_name, ordinal, signature_hint, parameter_hints, confidence_score)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (dll_name, export['function_name'], export.get('ordinal', 0),
                      export.get('signature_hint', ''), export.get('parameter_hints', ''), 
                      confidence))
            
            # Store relevant strings
            for string_info in strings_data:
                cursor.execute('''
                    INSERT INTO dll_strings 
                    (dll_name, string_value, string_type, relevance_score)
                    VALUES (?, ?, ?, ?)
                ''', (dll_name, string_info['string_value'], 
                      string_info['string_type'], string_info['relevance_score']))
            
            conn.commit()
            logger.info(f"Stored analysis results for {dll_name}")
            
        except sqlite3.Error as e:
            logger.error(f"Database error storing results for {dll_name}: {e}")
        finally:
            conn.close()
    
    def generate_analysis_report(self) -> Dict:
        """Generate comprehensive analysis report"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        report = {
            'analysis_summary': {},
            'processing_stages': {},
            'high_confidence_findings': [],
            'dll_details': []
        }
        
        try:
            # Get summary statistics
            cursor.execute('''
                SELECT COUNT(*) as total_dlls, 
                       AVG(confidence_score) as avg_confidence,
                       SUM(export_count) as total_exports
                FROM algorithm_dlls
            ''')
            summary = cursor.fetchone()
            report['analysis_summary'] = {
                'total_dlls_analyzed': summary[0],
                'average_confidence': round(summary[1], 2) if summary[1] else 0,
                'total_exports_found': summary[2] if summary[2] else 0
            }
            
            # Get processing stage breakdown
            cursor.execute('''
                SELECT processing_stage, COUNT(*) as dll_count, AVG(confidence_score) as avg_confidence
                FROM algorithm_dlls 
                GROUP BY processing_stage
                ORDER BY dll_count DESC
            ''')
            for row in cursor.fetchall():
                report['processing_stages'][row[0]] = {
                    'dll_count': row[1],
                    'average_confidence': round(row[2], 2)
                }
            
            # Get high confidence findings
            cursor.execute('''
                SELECT dll_name, processing_stage, export_count, confidence_score
                FROM algorithm_dlls 
                WHERE confidence_score > 0.7
                ORDER BY confidence_score DESC
            ''')
            for row in cursor.fetchall():
                report['high_confidence_findings'].append({
                    'dll_name': row[0],
                    'processing_stage': row[1],
                    'export_count': row[2],
                    'confidence_score': row[3]
                })
            
            # Get detailed DLL information
            cursor.execute('''
                SELECT d.dll_name, d.processing_stage, d.export_count, d.confidence_score,
                       GROUP_CONCAT(e.function_name, '; ') as sample_exports
                FROM algorithm_dlls d
                LEFT JOIN (
                    SELECT dll_name, function_name, 
                           ROW_NUMBER() OVER (PARTITION BY dll_name ORDER BY ordinal) as rn
                    FROM dll_exports
                ) e ON d.dll_name = e.dll_name AND e.rn <= 5
                GROUP BY d.dll_name, d.processing_stage, d.export_count, d.confidence_score
                ORDER BY d.confidence_score DESC
            ''')
            for row in cursor.fetchall():
                report['dll_details'].append({
                    'dll_name': row[0],
                    'processing_stage': row[1],
                    'export_count': row[2],
                    'confidence_score': row[3],
                    'sample_exports': row[4] if row[4] else "No exports found"
                })
        
        except sqlite3.Error as e:
            logger.error(f"Database error generating report: {e}")
        finally:
            conn.close()
        
        return report
    
    def run_analysis(self) -> Dict:
        """Run complete algorithm DLL analysis"""
        logger.info("Starting algorithm DLL analysis...")
        
        # Find all algorithm DLLs
        dll_files = self.find_algorithm_dlls()
        
        if not dll_files:
            logger.warning("No algorithm DLLs found!")
            return {"error": "No algorithm DLLs found"}
        
        # Analyze each DLL
        for dll_path in dll_files:
            logger.info(f"Analyzing {dll_path.name}...")
            
            # Analyze exports
            exports, export_confidence = self.analyze_dll_exports(dll_path)
            
            # Analyze strings
            strings_data, string_confidence = self.analyze_dll_strings(dll_path)
            
            # Combined confidence score
            overall_confidence = (export_confidence + string_confidence) / 2
            
            # Store results
            self.store_analysis_results(dll_path, exports, strings_data, overall_confidence)
        
        # Generate final report
        report = self.generate_analysis_report()
        logger.info("Algorithm DLL analysis completed!")
        
        return report

def main():
    if len(sys.argv) != 2:
        print("Usage: python algorithm_dll_analyzer.py <base_path>")
        sys.exit(1)
    
    base_path = sys.argv[1]
    analyzer = AlgorithmDLLAnalyzer(base_path)
    
    # Run analysis
    report = analyzer.run_analysis()
    
    # Save report
    output_file = "analysis_output/algorithm_dll_analysis.json"
    os.makedirs("analysis_output", exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"Analysis complete! Report saved to {output_file}")
    print(f"Total DLLs analyzed: {report.get('analysis_summary', {}).get('total_dlls_analyzed', 0)}")
    print(f"Average confidence: {report.get('analysis_summary', {}).get('average_confidence', 0)}")

if __name__ == "__main__":
    main()