#!/usr/bin/env python3
"""
Algorithm DLL Export Analyzer for IntraoralScan Reverse Engineering
Focuses on Sn3D*.dll and algorithm*.dll files for core functionality analysis
"""

import os
import sys
import sqlite3
import json
import subprocess
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import hashlib

class AlgorithmDLLAnalyzer:
    def __init__(self, base_path: str = "IntraoralScan/Bin", db_path: str = "analysis_results.db"):
        self.base_path = Path(base_path)
        self.db_path = db_path
        self.init_database()
        
        # Priority order for analysis based on domain knowledge
        self.priority_dlls = [
            # Core 3D processing pipeline
            "Sn3DRealtimeScan.dll",
            "Sn3DRegistration.dll", 
            "Sn3DSpeckleFusion.dll",
            "Sn3DTextureBasedTrack.dll",
            "Sn3DScanSlam.dll",
            
            # AI/ML processing
            "Sn3DDentalAI.dll",
            "Sn3DDentalRealTimeSemSeg.dll",
            "Sn3DDentalOral.dll",
            "Sn3DDentalBoxDet.dll",
            
            # Calibration and geometry
            "Sn3DCalibrationJR.dll",
            "Sn3DPhotometricStereo.dll",
            "Sn3DGeometricTrackFusion.dll",
            
            # Algorithm variants (likely different implementations)
            "algorithmLzy.dll",
            "algorithmSaj.dll", 
            "algorithm1.dll",
            "algorithm2.dll"
        ]
        
        # Processing stage mapping based on naming patterns
        self.stage_mapping = {
            "acquisition": ["Sn3DRealtimeScan", "Sn3DPhotometricStereo", "Sn3DLineCode"],
            "calibration": ["Sn3DCalibrationJR", "Sn3DColorCorrect"],
            "registration": ["Sn3DRegistration", "Sn3DTextureBasedTrack", "Sn3DScanSlam"],
            "fusion": ["Sn3DSpeckleFusion", "Sn3DSpeckle", "Sn3DGeometricTrackFusion"],
            "ai_analysis": ["Sn3DDentalAI", "Sn3DDentalRealTimeSemSeg", "Sn3DDentalOral", "Sn3DDentalBoxDet"],
            "mesh_processing": ["Sn3DMagic", "Sn3DDraco", "Sn3DTooling"],
            "clinical_analysis": ["Sn3DOralExamCCP", "Sn3DOralExamResidualCrown", "Sn3DInfraredCariesDet"]
        }

    def init_database(self):
        """Initialize SQLite database for storing analysis results"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Drop existing table if it exists to ensure correct schema
        cursor.execute('DROP TABLE IF EXISTS dll_exports')
        cursor.execute('DROP TABLE IF EXISTS function_signatures')
        
        # Create table for DLL export analysis
        cursor.execute('''
            CREATE TABLE dll_exports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                dll_name TEXT NOT NULL,
                dll_path TEXT NOT NULL,
                file_size INTEGER,
                md5_hash TEXT,
                export_count INTEGER,
                exports TEXT,
                processing_stage TEXT,
                confidence_score REAL,
                analysis_method TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create table for function signatures
        cursor.execute('''
            CREATE TABLE function_signatures (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                dll_name TEXT NOT NULL,
                function_name TEXT NOT NULL,
                ordinal INTEGER,
                signature_hint TEXT,
                parameter_hints TEXT,
                return_type_hint TEXT,
                confidence_score REAL,
                analysis_method TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()

    def get_dll_files(self) -> List[Path]:
        """Get all Sn3D*.dll and algorithm*.dll files"""
        dll_files = []
        
        # Find Sn3D*.dll files
        sn3d_files = list(self.base_path.glob("Sn3D*.dll"))
        algorithm_files = list(self.base_path.glob("algorithm*.dll"))
        
        all_files = sn3d_files + algorithm_files
        
        # Sort by priority
        prioritized = []
        remaining = []
        
        for dll_file in all_files:
            if dll_file.name in self.priority_dlls:
                prioritized.append(dll_file)
            else:
                remaining.append(dll_file)
        
        # Sort prioritized by their order in priority_dlls
        prioritized.sort(key=lambda x: self.priority_dlls.index(x.name) if x.name in self.priority_dlls else 999)
        
        return prioritized + remaining

    def analyze_dll_exports_linux(self, dll_path: Path) -> Dict:
        """Analyze DLL exports using Linux tools (objdump, strings)"""
        result = {
            "dll_name": dll_path.name,
            "dll_path": str(dll_path),
            "file_size": dll_path.stat().st_size,
            "md5_hash": self.get_file_hash(dll_path),
            "exports": [],
            "export_count": 0,
            "analysis_method": "linux_objdump",
            "confidence_score": 0.0
        }
        
        try:
            # Try objdump for PE export table
            cmd = ["objdump", "-p", str(dll_path)]
            process = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if process.returncode == 0:
                exports = self.parse_objdump_exports(process.stdout)
                result["exports"] = exports
                result["export_count"] = len(exports)
                result["confidence_score"] = 0.8 if exports else 0.3
            else:
                # Fallback to strings analysis
                result.update(self.analyze_dll_strings(dll_path))
                result["analysis_method"] = "linux_strings_fallback"
                
        except Exception as e:
            print(f"Error analyzing {dll_path.name}: {e}")
            # Final fallback to strings
            result.update(self.analyze_dll_strings(dll_path))
            result["analysis_method"] = "linux_strings_error_fallback"
            
        return result

    def parse_objdump_exports(self, objdump_output: str) -> List[Dict]:
        """Parse objdump output to extract export functions"""
        exports = []
        in_export_section = False
        
        for line in objdump_output.split('\n'):
            line = line.strip()
            
            if "Export Table:" in line or "[Ordinal/Name Pointer] Table" in line:
                in_export_section = True
                continue
                
            if in_export_section and line:
                # Look for export entries (various formats)
                # Format: [ordinal] name
                match = re.match(r'\s*\[?\s*(\d+)\s*\]?\s+(.+)', line)
                if match:
                    ordinal = int(match.group(1))
                    name = match.group(2).strip()
                    
                    exports.append({
                        "name": name,
                        "ordinal": ordinal,
                        "signature_hint": self.infer_function_signature(name)
                    })
                elif line.startswith('[') or 'Table' in line:
                    # End of export section
                    break
                    
        return exports

    def analyze_dll_strings(self, dll_path: Path) -> Dict:
        """Fallback analysis using strings command"""
        result = {
            "exports": [],
            "export_count": 0,
            "confidence_score": 0.2
        }
        
        try:
            # Extract strings and look for function-like patterns
            cmd = ["strings", "-n", "4", str(dll_path)]
            process = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if process.returncode == 0:
                function_patterns = self.extract_function_patterns(process.stdout)
                result["exports"] = function_patterns
                result["export_count"] = len(function_patterns)
                result["confidence_score"] = 0.4 if function_patterns else 0.1
                
        except Exception as e:
            print(f"String analysis failed for {dll_path.name}: {e}")
            
        return result

    def extract_function_patterns(self, strings_output: str) -> List[Dict]:
        """Extract likely function names from strings output"""
        functions = []
        
        # Common function name patterns in dental/3D processing
        function_patterns = [
            r'\b[A-Z][a-zA-Z]*(?:Init|Create|Process|Update|Compute|Calculate|Generate|Build|Reconstruct|Register|Align|Fuse|Segment|Detect|Extract|Transform|Convert|Export|Import|Load|Save)\w*\b',
            r'\b(?:init|create|process|update|compute|calculate|generate|build|reconstruct|register|align|fuse|segment|detect|extract|transform|convert|export|import|load|save)[A-Z]\w*\b',
            r'\b[a-zA-Z]*(?:3D|Mesh|Point|Cloud|Texture|Image|Camera|Scan|Dental|Tooth|Jaw|Oral)\w*\b'
        ]
        
        seen_functions = set()
        
        for line in strings_output.split('\n'):
            line = line.strip()
            if len(line) < 4 or len(line) > 100:
                continue
                
            for pattern in function_patterns:
                matches = re.findall(pattern, line)
                for match in matches:
                    if match not in seen_functions and self.is_likely_function_name(match):
                        seen_functions.add(match)
                        functions.append({
                            "name": match,
                            "ordinal": None,
                            "signature_hint": self.infer_function_signature(match),
                            "source_string": line[:100]  # Context
                        })
        
        return functions[:50]  # Limit to top 50 candidates

    def is_likely_function_name(self, name: str) -> bool:
        """Heuristic to determine if a string is likely a function name"""
        if len(name) < 4 or len(name) > 50:
            return False
            
        # Must contain letters
        if not re.search(r'[a-zA-Z]', name):
            return False
            
        # Should not be all caps (likely constants)
        if name.isupper() and len(name) > 8:
            return False
            
        # Should not contain spaces or special chars (except underscore)
        if re.search(r'[^a-zA-Z0-9_]', name):
            return False
            
        # Dental/3D processing keywords boost confidence
        dental_keywords = ['dental', 'tooth', 'jaw', 'oral', 'scan', '3d', 'mesh', 'point', 'cloud', 'texture', 'camera', 'calibrat', 'register', 'fus', 'segment']
        if any(keyword in name.lower() for keyword in dental_keywords):
            return True
            
        # Common function prefixes/suffixes
        function_indicators = ['init', 'create', 'process', 'update', 'compute', 'generate', 'build', 'get', 'set', 'load', 'save', 'export', 'import']
        if any(indicator in name.lower() for indicator in function_indicators):
            return True
            
        return False

    def infer_function_signature(self, function_name: str) -> str:
        """Infer likely function signature based on naming patterns"""
        name_lower = function_name.lower()
        
        # Return type inference
        if any(keyword in name_lower for keyword in ['get', 'compute', 'calculate', 'generate', 'create']):
            if 'count' in name_lower or 'size' in name_lower:
                return_type = "int"
            elif 'point' in name_lower or 'mesh' in name_lower:
                return_type = "void*"
            elif 'matrix' in name_lower or 'transform' in name_lower:
                return_type = "float*"
            else:
                return_type = "bool"
        elif any(keyword in name_lower for keyword in ['init', 'set', 'update', 'process']):
            return_type = "bool"
        else:
            return_type = "void"
            
        # Parameter inference
        if 'init' in name_lower:
            params = "void"
        elif any(keyword in name_lower for keyword in ['process', 'compute', 'transform']):
            params = "void* input, void* output"
        elif 'set' in name_lower:
            params = "void* data"
        elif 'get' in name_lower:
            params = "void* buffer"
        else:
            params = "..."
            
        return f"{return_type} {function_name}({params})"

    def get_processing_stage(self, dll_name: str) -> Tuple[str, float]:
        """Determine processing stage based on DLL name"""
        dll_base = dll_name.replace('.dll', '')
        
        for stage, patterns in self.stage_mapping.items():
            for pattern in patterns:
                if pattern in dll_base:
                    return stage, 0.9
                    
        # Fallback pattern matching
        if 'realtime' in dll_name.lower() or 'scan' in dll_name.lower():
            return "acquisition", 0.6
        elif 'register' in dll_name.lower() or 'track' in dll_name.lower():
            return "registration", 0.6
        elif 'fusion' in dll_name.lower() or 'speckle' in dll_name.lower():
            return "fusion", 0.6
        elif 'ai' in dll_name.lower() or 'dental' in dll_name.lower():
            return "ai_analysis", 0.6
        elif 'algorithm' in dll_name.lower():
            return "unknown_algorithm", 0.4
        else:
            return "unknown", 0.1

    def get_file_hash(self, file_path: Path) -> str:
        """Calculate MD5 hash of file"""
        hash_md5 = hashlib.md5()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception:
            return "unknown"

    def save_analysis_results(self, analysis_results: List[Dict]):
        """Save analysis results to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for result in analysis_results:
            # Insert DLL analysis
            stage, stage_confidence = self.get_processing_stage(result["dll_name"])
            
            cursor.execute('''
                INSERT OR REPLACE INTO dll_exports 
                (dll_name, dll_path, file_size, md5_hash, export_count, exports, 
                 processing_stage, confidence_score, analysis_method)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                result["dll_name"],
                result["dll_path"], 
                result["file_size"],
                result["md5_hash"],
                result["export_count"],
                json.dumps(result["exports"]),
                stage,
                result["confidence_score"] * stage_confidence,
                result["analysis_method"]
            ))
            
            # Insert function signatures
            for export in result["exports"]:
                cursor.execute('''
                    INSERT OR REPLACE INTO function_signatures
                    (dll_name, function_name, ordinal, signature_hint, confidence_score, analysis_method)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    result["dll_name"],
                    export["name"],
                    export.get("ordinal"),
                    export.get("signature_hint", ""),
                    result["confidence_score"],
                    result["analysis_method"]
                ))
        
        conn.commit()
        conn.close()

    def generate_analysis_report(self) -> Dict:
        """Generate comprehensive analysis report"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get summary statistics
        cursor.execute('SELECT COUNT(*) FROM dll_exports')
        total_dlls = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM function_signatures')
        total_functions = cursor.fetchone()[0]
        
        cursor.execute('SELECT processing_stage, COUNT(*) FROM dll_exports GROUP BY processing_stage')
        stage_distribution = dict(cursor.fetchall())
        
        cursor.execute('SELECT dll_name, export_count, confidence_score FROM dll_exports ORDER BY confidence_score DESC')
        dll_analysis = cursor.fetchall()
        
        # Get high-confidence functions
        cursor.execute('''
            SELECT dll_name, function_name, signature_hint, confidence_score 
            FROM function_signatures 
            WHERE confidence_score > 0.5 
            ORDER BY confidence_score DESC 
            LIMIT 50
        ''')
        high_confidence_functions = cursor.fetchall()
        
        conn.close()
        
        return {
            "summary": {
                "total_dlls_analyzed": total_dlls,
                "total_functions_found": total_functions,
                "stage_distribution": stage_distribution
            },
            "dll_analysis": [
                {"name": name, "export_count": count, "confidence": conf}
                for name, count, conf in dll_analysis
            ],
            "high_confidence_functions": [
                {"dll": dll, "function": func, "signature": sig, "confidence": conf}
                for dll, func, sig, conf in high_confidence_functions
            ]
        }

    def run_analysis(self) -> Dict:
        """Run complete algorithm DLL analysis"""
        print("Starting Algorithm DLL Export Analysis...")
        
        dll_files = self.get_dll_files()
        print(f"Found {len(dll_files)} algorithm DLL files to analyze")
        
        analysis_results = []
        
        for i, dll_file in enumerate(dll_files, 1):
            print(f"[{i}/{len(dll_files)}] Analyzing {dll_file.name}...")
            
            try:
                result = self.analyze_dll_exports_linux(dll_file)
                analysis_results.append(result)
                
                print(f"  - Found {result['export_count']} exports (confidence: {result['confidence_score']:.2f})")
                
            except Exception as e:
                print(f"  - Error: {e}")
                continue
        
        # Save results
        print("Saving analysis results to database...")
        self.save_analysis_results(analysis_results)
        
        # Generate report
        print("Generating analysis report...")
        report = self.generate_analysis_report()
        
        return report

def main():
    if len(sys.argv) > 1:
        base_path = sys.argv[1]
    else:
        base_path = "IntraoralScan/Bin"
    
    analyzer = AlgorithmDLLAnalyzer(base_path)
    report = analyzer.run_analysis()
    
    # Print summary
    print("\n" + "="*60)
    print("ALGORITHM DLL ANALYSIS SUMMARY")
    print("="*60)
    print(f"Total DLLs analyzed: {report['summary']['total_dlls_analyzed']}")
    print(f"Total functions found: {report['summary']['total_functions_found']}")
    print(f"Processing stage distribution: {report['summary']['stage_distribution']}")
    
    print(f"\nTop DLLs by confidence:")
    for dll in report['dll_analysis'][:10]:
        print(f"  {dll['name']}: {dll['export_count']} exports (confidence: {dll['confidence']:.2f})")
    
    print(f"\nHigh-confidence functions (top 10):")
    for func in report['high_confidence_functions'][:10]:
        print(f"  {func['dll']}::{func['function']} - {func['signature']}")
    
    # Save report to JSON
    with open("analysis_output/algorithm_dll_analysis_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"\nDetailed report saved to: analysis_output/algorithm_dll_analysis_report.json")
    print("Database updated with analysis results.")

if __name__ == "__main__":
    main()