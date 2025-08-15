#!/usr/bin/env python3
"""
Focused Algorithm DLL Analyzer
Extracts meaningful function signatures and algorithm insights from high-value DLLs
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

class FocusedAlgorithmAnalyzer:
    def __init__(self, base_path: str, db_path: str = "analysis_results.db"):
        self.base_path = Path(base_path)
        self.db_path = db_path
        
        # High-value DLLs with their expected functionality
        self.high_value_dlls = {
            'Sn3DSpeckleFusion.dll': 'TSDF volume fusion and mesh generation',
            'Sn3DRegistration.dll': 'Point cloud registration and ICP alignment',
            'Sn3DRealtimeScan.dll': 'Real-time scanning and data acquisition',
            'Sn3DDentalAI.dll': 'AI-based dental analysis and segmentation',
            'Sn3DMagic.dll': 'Mesh processing and geometric operations',
            'Sn3DCalibrationJR.dll': 'Camera calibration and stereo setup',
            'algorithmLzy.dll': 'Custom algorithm implementation',
            'algorithm1.dll': 'Primary algorithm processing',
            'algorithm2.dll': 'Secondary algorithm processing'
        }
    
    def extract_meaningful_functions(self, dll_path: Path) -> Tuple[List[Dict], float]:
        """Extract meaningful function signatures from DLL strings"""
        functions = []
        confidence = 0.0
        
        try:
            # Extract strings with focus on function signatures
            result = subprocess.run([
                'strings', '-n', '8', str(dll_path)
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                functions, confidence = self._parse_function_signatures(result.stdout, dll_path.name)
                logger.info(f"Extracted {len(functions)} meaningful functions from {dll_path.name}")
            
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            logger.warning(f"Function extraction failed for {dll_path.name}: {e}")
        
        return functions, confidence
    
    def _parse_function_signatures(self, strings_output: str, dll_name: str) -> Tuple[List[Dict], float]:
        """Parse meaningful function signatures from strings"""
        functions = []
        confidence = 0.7
        
        # Function signature patterns
        function_patterns = [
            # C++ class methods
            r'Sn3D\w+::\w+::\w+',
            r'Sn3DAlgorithm::\w+::\w+',
            
            # Algorithm-specific functions
            r'\w*[Rr]egister\w*',
            r'\w*[Ff]usion\w*',
            r'\w*[Cc]alibrat\w*',
            r'\w*[Ss]egment\w*',
            r'\w*[Cc]lassify\w*',
            r'\w*[Dd]etect\w*',
            r'\w*[Pp]rocess\w*',
            r'\w*[Cc]ompute\w*',
            
            # CUDA kernels
            r'\w+_kernel\w*',
            r'__global__\s+\w+',
            
            # OpenCV/PCL functions
            r'cv::\w+',
            r'pcl::\w+',
            
            # Dental-specific functions
            r'\w*[Tt]ooth\w*',
            r'\w*[Dd]ental\w*',
            r'\w*[Oo]ral\w*',
            r'\w*[Mm]esh\w*',
            r'\w*[Pp]oint[Cc]loud\w*'
        ]
        
        lines = strings_output.split('\n')
        processed_lines = 0
        
        for line in lines:
            line = line.strip()
            processed_lines += 1
            
            # Limit processing for performance
            if processed_lines > 10000:
                break
                
            if len(line) < 8 or len(line) > 200:
                continue
            
            # Check against function patterns
            for pattern in function_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    function_info = {
                        'function_signature': line,
                        'function_type': self._classify_function(line),
                        'algorithm_stage': self._map_to_algorithm_stage(line, dll_name),
                        'parameters': self._extract_function_parameters(line),
                        'confidence': self._calculate_function_confidence(line, dll_name)
                    }
                    
                    # Only add high-confidence functions
                    if function_info['confidence'] > 0.5:
                        functions.append(function_info)
                    break
        
        # Remove duplicates and sort by confidence
        unique_functions = []
        seen_signatures = set()
        
        for func in sorted(functions, key=lambda x: x['confidence'], reverse=True):
            if func['function_signature'] not in seen_signatures:
                unique_functions.append(func)
                seen_signatures.add(func['function_signature'])
                
                # Limit to top functions
                if len(unique_functions) >= 50:
                    break
        
        return unique_functions, confidence
    
    def _classify_function(self, signature: str) -> str:
        """Classify function by type"""
        sig_lower = signature.lower()
        
        if any(term in sig_lower for term in ['init', 'initialize', 'create', 'new']):
            return 'initialization'
        elif any(term in sig_lower for term in ['process', 'compute', 'execute', 'run']):
            return 'processing'
        elif any(term in sig_lower for term in ['register', 'align', 'match']):
            return 'registration'
        elif any(term in sig_lower for term in ['fuse', 'fusion', 'merge', 'combine']):
            return 'fusion'
        elif any(term in sig_lower for term in ['segment', 'classify', 'detect', 'recognize']):
            return 'analysis'
        elif any(term in sig_lower for term in ['calibrat', 'correct', 'adjust']):
            return 'calibration'
        elif any(term in sig_lower for term in ['mesh', 'triangle', 'vertex', 'geometry']):
            return 'geometry'
        elif any(term in sig_lower for term in ['kernel', 'cuda', 'gpu']):
            return 'gpu_compute'
        elif any(term in sig_lower for term in ['get', 'set', 'param', 'config']):
            return 'parameter_access'
        else:
            return 'utility'
    
    def _map_to_algorithm_stage(self, signature: str, dll_name: str) -> str:
        """Map function to algorithm processing stage"""
        sig_lower = signature.lower()
        dll_lower = dll_name.lower()
        
        # DLL-based mapping
        if 'calibration' in dll_lower:
            return 'calibration'
        elif 'registration' in dll_lower:
            return 'registration'
        elif 'fusion' in dll_lower or 'speckle' in dll_lower:
            return 'fusion'
        elif 'realtime' in dll_lower or 'scan' in dll_lower:
            return 'acquisition'
        elif 'ai' in dll_lower or 'dental' in dll_lower:
            return 'ai_analysis'
        elif 'magic' in dll_lower:
            return 'mesh_processing'
        
        # Function-based mapping
        if any(term in sig_lower for term in ['calibrat', 'intrinsic', 'extrinsic']):
            return 'calibration'
        elif any(term in sig_lower for term in ['register', 'align', 'icp', 'slam']):
            return 'registration'
        elif any(term in sig_lower for term in ['fuse', 'fusion', 'tsdf', 'voxel']):
            return 'fusion'
        elif any(term in sig_lower for term in ['segment', 'classify', 'detect', 'neural']):
            return 'ai_analysis'
        elif any(term in sig_lower for term in ['mesh', 'triangle', 'vertex']):
            return 'mesh_processing'
        elif any(term in sig_lower for term in ['acquire', 'capture', 'stream']):
            return 'acquisition'
        else:
            return 'general'
    
    def _extract_function_parameters(self, signature: str) -> str:
        """Extract parameter information from function signature"""
        params = []
        sig_lower = signature.lower()
        
        if 'pointcloud' in sig_lower or 'point_cloud' in sig_lower:
            params.append('point_cloud')
        if 'mesh' in sig_lower:
            params.append('mesh_data')
        if 'image' in sig_lower or 'frame' in sig_lower:
            params.append('image_data')
        if 'matrix' in sig_lower or 'transform' in sig_lower:
            params.append('transformation_matrix')
        if 'camera' in sig_lower:
            params.append('camera_parameters')
        if 'config' in sig_lower or 'param' in sig_lower:
            params.append('configuration')
        if 'result' in sig_lower or 'output' in sig_lower:
            params.append('output_data')
        if 'cuda' in sig_lower or 'gpu' in sig_lower:
            params.append('gpu_memory')
        
        return ', '.join(params) if params else 'unknown'
    
    def _calculate_function_confidence(self, signature: str, dll_name: str) -> float:
        """Calculate confidence score for function relevance"""
        confidence = 0.3  # Base confidence
        
        sig_lower = signature.lower()
        dll_lower = dll_name.lower()
        
        # Boost confidence for algorithm-relevant terms
        algorithm_terms = ['register', 'fusion', 'calibrat', 'segment', 'detect', 
                          'process', 'compute', 'mesh', 'point', 'cloud', 'dental']
        for term in algorithm_terms:
            if term in sig_lower:
                confidence += 0.1
        
        # Boost for Sn3D namespace functions
        if 'sn3d' in sig_lower:
            confidence += 0.2
        
        # Boost for CUDA/GPU functions
        if any(term in sig_lower for term in ['cuda', 'gpu', 'kernel']):
            confidence += 0.15
        
        # Boost for class method signatures
        if '::' in signature:
            confidence += 0.1
        
        # Boost for DLL-specific relevance
        if any(term in dll_lower for term in ['fusion', 'registration', 'ai', 'calibration']):
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def store_algorithm_analysis(self, dll_name: str, functions: List[Dict], confidence: float):
        """Store algorithm analysis results in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Create algorithm_functions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS algorithm_functions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    dll_name TEXT,
                    function_signature TEXT,
                    function_type TEXT,
                    algorithm_stage TEXT,
                    parameters TEXT,
                    confidence REAL,
                    FOREIGN KEY (dll_name) REFERENCES algorithm_dlls (dll_name)
                )
            ''')
            
            # Clear existing functions for this DLL
            cursor.execute('DELETE FROM algorithm_functions WHERE dll_name = ?', (dll_name,))
            
            # Insert new functions
            for func in functions:
                cursor.execute('''
                    INSERT INTO algorithm_functions 
                    (dll_name, function_signature, function_type, algorithm_stage, parameters, confidence)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (dll_name, func['function_signature'], func['function_type'],
                      func['algorithm_stage'], func['parameters'], func['confidence']))
            
            # Update algorithm_dlls table with function count
            cursor.execute('''
                UPDATE algorithm_dlls 
                SET export_count = ?, confidence_score = ?
                WHERE dll_name = ?
            ''', (len(functions), confidence, dll_name))
            
            conn.commit()
            logger.info(f"Stored {len(functions)} algorithm functions for {dll_name}")
            
        except sqlite3.Error as e:
            logger.error(f"Database error storing algorithm functions for {dll_name}: {e}")
        finally:
            conn.close()
    
    def generate_algorithm_mapping_report(self) -> Dict:
        """Generate comprehensive algorithm mapping report"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        report = {
            'dll_analysis': {},
            'algorithm_stages': {},
            'function_types': {},
            'high_confidence_functions': [],
            'processing_pipeline': {}
        }
        
        try:
            # Get DLL analysis summary
            cursor.execute('''
                SELECT dll_name, COUNT(*) as function_count, AVG(confidence) as avg_confidence
                FROM algorithm_functions 
                GROUP BY dll_name
                ORDER BY avg_confidence DESC
            ''')
            
            for row in cursor.fetchall():
                dll_name, func_count, avg_conf = row
                expected_functionality = self.high_value_dlls.get(dll_name, 'Unknown functionality')
                
                report['dll_analysis'][dll_name] = {
                    'function_count': func_count,
                    'average_confidence': round(avg_conf, 2),
                    'expected_functionality': expected_functionality
                }
            
            # Get algorithm stage breakdown
            cursor.execute('''
                SELECT algorithm_stage, COUNT(*) as function_count, 
                       GROUP_CONCAT(DISTINCT dll_name) as dlls
                FROM algorithm_functions 
                GROUP BY algorithm_stage
                ORDER BY function_count DESC
            ''')
            
            for row in cursor.fetchall():
                stage, count, dlls = row
                report['algorithm_stages'][stage] = {
                    'function_count': count,
                    'involved_dlls': dlls.split(',') if dlls else []
                }
            
            # Get function type breakdown
            cursor.execute('''
                SELECT function_type, COUNT(*) as count
                FROM algorithm_functions 
                GROUP BY function_type
                ORDER BY count DESC
            ''')
            
            for row in cursor.fetchall():
                func_type, count = row
                report['function_types'][func_type] = count
            
            # Get high confidence functions
            cursor.execute('''
                SELECT dll_name, function_signature, function_type, algorithm_stage, confidence
                FROM algorithm_functions 
                WHERE confidence > 0.8
                ORDER BY confidence DESC
                LIMIT 20
            ''')
            
            for row in cursor.fetchall():
                report['high_confidence_functions'].append({
                    'dll_name': row[0],
                    'function_signature': row[1],
                    'function_type': row[2],
                    'algorithm_stage': row[3],
                    'confidence': row[4]
                })
            
            # Build processing pipeline map
            pipeline_stages = ['calibration', 'acquisition', 'registration', 'fusion', 'ai_analysis', 'mesh_processing']
            
            for stage in pipeline_stages:
                cursor.execute('''
                    SELECT dll_name, COUNT(*) as function_count
                    FROM algorithm_functions 
                    WHERE algorithm_stage = ?
                    GROUP BY dll_name
                    ORDER BY function_count DESC
                ''', (stage,))
                
                stage_dlls = {}
                for row in cursor.fetchall():
                    stage_dlls[row[0]] = row[1]
                
                if stage_dlls:
                    report['processing_pipeline'][stage] = stage_dlls
        
        except sqlite3.Error as e:
            logger.error(f"Database error generating algorithm mapping report: {e}")
        finally:
            conn.close()
        
        return report
    
    def run_focused_analysis(self) -> Dict:
        """Run focused analysis on high-value algorithm DLLs"""
        logger.info("Starting focused algorithm DLL analysis...")
        
        bin_path = self.base_path / "IntraoralScan" / "Bin"
        results = {
            'analyzed_dlls': [],
            'total_functions_extracted': 0,
            'analysis_summary': {}
        }
        
        for dll_name, expected_func in self.high_value_dlls.items():
            dll_path = bin_path / dll_name
            
            if not dll_path.exists():
                logger.warning(f"DLL not found: {dll_path}")
                continue
            
            logger.info(f"Analyzing {dll_name} - {expected_func}")
            
            # Extract meaningful functions
            functions, confidence = self.extract_meaningful_functions(dll_path)
            
            # Store results
            self.store_algorithm_analysis(dll_name, functions, confidence)
            
            # Add to results
            results['analyzed_dlls'].append({
                'dll_name': dll_name,
                'expected_functionality': expected_func,
                'functions_extracted': len(functions),
                'confidence_score': confidence,
                'top_functions': [f['function_signature'] for f in functions[:3]]
            })
            
            results['total_functions_extracted'] += len(functions)
        
        # Generate comprehensive report
        algorithm_report = self.generate_algorithm_mapping_report()
        results['algorithm_mapping'] = algorithm_report
        
        results['analysis_summary'] = {
            'dlls_analyzed': len(results['analyzed_dlls']),
            'average_functions_per_dll': results['total_functions_extracted'] / len(results['analyzed_dlls']) if results['analyzed_dlls'] else 0,
            'processing_stages_identified': len(algorithm_report.get('algorithm_stages', {})),
            'high_confidence_functions': len(algorithm_report.get('high_confidence_functions', []))
        }
        
        logger.info("Focused algorithm analysis completed!")
        return results

def main():
    if len(sys.argv) != 2:
        print("Usage: python focused_algorithm_analyzer.py <base_path>")
        sys.exit(1)
    
    base_path = sys.argv[1]
    analyzer = FocusedAlgorithmAnalyzer(base_path)
    
    # Run focused analysis
    results = analyzer.run_focused_analysis()
    
    # Save results
    output_file = "analysis_output/focused_algorithm_analysis.json"
    os.makedirs("analysis_output", exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Focused algorithm analysis complete! Report saved to {output_file}")
    print(f"Total functions extracted: {results['total_functions_extracted']}")
    print(f"Processing stages identified: {results['analysis_summary']['processing_stages_identified']}")

if __name__ == "__main__":
    main()