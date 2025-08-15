#!/usr/bin/env python3
"""
Linux-compatible DLL Export Analyzer
Uses objdump and nm to analyze Windows DLL exports on Linux
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

class LinuxDLLExportAnalyzer:
    def __init__(self, base_path: str, db_path: str = "analysis_results.db"):
        self.base_path = Path(base_path)
        self.db_path = db_path
        
    def analyze_dll_with_objdump(self, dll_path: Path) -> Tuple[List[Dict], float]:
        """Analyze DLL using objdump for export information"""
        exports = []
        confidence = 0.0
        
        try:
            # Try objdump -p for PE headers
            result = subprocess.run([
                'objdump', '-p', str(dll_path)
            ], capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                exports, confidence = self._parse_objdump_exports(result.stdout)
                logger.info(f"objdump analysis for {dll_path.name}: {len(exports)} exports found")
            else:
                logger.warning(f"objdump failed for {dll_path.name}: {result.stderr}")
        
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            logger.warning(f"objdump analysis failed for {dll_path.name}: {e}")
        
        return exports, confidence
    
    def _parse_objdump_exports(self, output: str) -> Tuple[List[Dict], float]:
        """Parse objdump output for export information"""
        exports = []
        confidence = 0.7
        
        # Look for Export Table section
        in_export_section = False
        export_section_found = False
        
        lines = output.split('\n')
        for i, line in enumerate(lines):
            line = line.strip()
            
            # Look for export table indicators
            if 'Export Table' in line or 'Export Address Table' in line:
                in_export_section = True
                export_section_found = True
                continue
            
            # Look for DLL name and characteristics
            if in_export_section and ('DLL Name' in line or 'Name:' in line):
                continue
            
            # Parse export entries
            if in_export_section and line:
                # Look for ordinal/name patterns
                ordinal_match = re.search(r'^\s*(\d+)\s+([0-9a-fA-F]+)\s+(.+)', line)
                if ordinal_match:
                    ordinal = int(ordinal_match.group(1))
                    address = ordinal_match.group(2)
                    name = ordinal_match.group(3).strip()
                    
                    export_info = {
                        'function_name': name,
                        'ordinal': ordinal,
                        'address': address,
                        'signature_hint': self._infer_signature_from_name(name),
                        'parameter_hints': self._extract_parameter_hints(name)
                    }
                    exports.append(export_info)
                
                # Alternative pattern for name-only exports
                elif re.match(r'^\s*[a-zA-Z_][a-zA-Z0-9_]*', line):
                    name = line.strip()
                    if len(name) > 2 and not name.startswith('['):
                        export_info = {
                            'function_name': name,
                            'ordinal': 0,
                            'address': '',
                            'signature_hint': self._infer_signature_from_name(name),
                            'parameter_hints': self._extract_parameter_hints(name)
                        }
                        exports.append(export_info)
            
            # Stop if we hit another section
            if in_export_section and line.startswith('The ') and 'section' in line.lower():
                break
        
        if not export_section_found:
            # Try alternative parsing for different objdump formats
            exports, confidence = self._parse_alternative_objdump_format(output)
        
        return exports, confidence
    
    def _parse_alternative_objdump_format(self, output: str) -> Tuple[List[Dict], float]:
        """Alternative parsing for different objdump output formats"""
        exports = []
        confidence = 0.5
        
        # Look for function names in various sections
        function_patterns = [
            r'^\s*[0-9a-fA-F]+\s+[gl]\s+F\s+\S+\s+[0-9a-fA-F]+\s+(.+)',  # Symbol table format
            r'^\s*([a-zA-Z_][a-zA-Z0-9_@]+)\s*$',  # Simple name format
        ]
        
        for line in output.split('\n'):
            line = line.strip()
            for pattern in function_patterns:
                match = re.search(pattern, line)
                if match:
                    name = match.group(1).strip()
                    if len(name) > 2 and not any(skip in name for skip in ['section', 'Disassembly', 'file format']):
                        export_info = {
                            'function_name': name,
                            'ordinal': 0,
                            'address': '',
                            'signature_hint': self._infer_signature_from_name(name),
                            'parameter_hints': self._extract_parameter_hints(name)
                        }
                        exports.append(export_info)
                        break
        
        return exports, confidence
    
    def analyze_dll_with_nm(self, dll_path: Path) -> Tuple[List[Dict], float]:
        """Analyze DLL using nm for symbol information"""
        exports = []
        confidence = 0.0
        
        try:
            # Try nm for symbol analysis
            result = subprocess.run([
                'nm', '-D', str(dll_path)
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                exports, confidence = self._parse_nm_output(result.stdout)
                logger.info(f"nm analysis for {dll_path.name}: {len(exports)} symbols found")
            else:
                # Try without -D flag
                result = subprocess.run([
                    'nm', str(dll_path)
                ], capture_output=True, text=True, timeout=30)
                
                if result.returncode == 0:
                    exports, confidence = self._parse_nm_output(result.stdout)
                    logger.info(f"nm analysis for {dll_path.name}: {len(exports)} symbols found")
        
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            logger.warning(f"nm analysis failed for {dll_path.name}: {e}")
        
        return exports, confidence
    
    def _parse_nm_output(self, output: str) -> Tuple[List[Dict], float]:
        """Parse nm output for symbol information"""
        exports = []
        confidence = 0.6
        
        for line in output.split('\n'):
            line = line.strip()
            if not line:
                continue
            
            # Parse nm output format: address type name
            parts = line.split()
            if len(parts) >= 3:
                address = parts[0]
                symbol_type = parts[1]
                name = ' '.join(parts[2:])
                
                # Focus on exported functions (T, D, B types typically)
                if symbol_type in ['T', 'D', 'B', 'R'] and len(name) > 2:
                    export_info = {
                        'function_name': name,
                        'ordinal': 0,
                        'address': address,
                        'symbol_type': symbol_type,
                        'signature_hint': self._infer_signature_from_name(name),
                        'parameter_hints': self._extract_parameter_hints(name)
                    }
                    exports.append(export_info)
        
        return exports, confidence
    
    def analyze_cuda_kernels(self, dll_path: Path) -> Tuple[List[Dict], float]:
        """Extract CUDA kernel information from strings"""
        kernels = []
        confidence = 0.0
        
        try:
            # Extract strings and look for CUDA kernel signatures
            result = subprocess.run([
                'strings', '-n', '10', str(dll_path)
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                kernels, confidence = self._parse_cuda_kernels(result.stdout)
                logger.info(f"CUDA kernel analysis for {dll_path.name}: {len(kernels)} kernels found")
        
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            logger.warning(f"CUDA kernel analysis failed for {dll_path.name}: {e}")
        
        return kernels, confidence
    
    def _parse_cuda_kernels(self, strings_output: str) -> Tuple[List[Dict], float]:
        """Parse CUDA kernel signatures from strings"""
        kernels = []
        confidence = 0.8
        
        # Limit processing to avoid timeout
        lines = strings_output.split('\n')[:5000]  # Process first 5000 lines only
        
        # CUDA kernel patterns - simplified for performance
        cuda_indicators = ['kernel', 'cuda', '__global__', '.entry', 'ptx', 'Sn3D']
        
        for line in lines:
            line = line.strip()
            if len(line) < 10 or len(line) > 500:  # Skip very short or very long lines
                continue
            
            # Quick check for CUDA-specific strings
            line_lower = line.lower()
            if any(indicator in line_lower for indicator in cuda_indicators):
                # Extract kernel name using simpler logic
                kernel_name = self._extract_kernel_name(line)
                if kernel_name:
                    kernel_info = {
                        'kernel_name': kernel_name,
                        'full_signature': line[:200],  # Truncate long signatures
                        'kernel_type': self._classify_cuda_kernel(line),
                        'parameters': self._extract_cuda_parameters(line)
                    }
                    kernels.append(kernel_info)
                    
                    # Limit number of kernels to avoid excessive processing
                    if len(kernels) >= 50:
                        break
        
        return kernels, confidence
    
    def _extract_kernel_name(self, line: str) -> Optional[str]:
        """Extract kernel name using simple heuristics"""
        # Look for function-like patterns
        if 'Sn3D' in line:
            # Extract Sn3D function names
            match = re.search(r'(Sn3D\w+)', line)
            if match:
                return match.group(1)
        
        # Look for kernel in the name
        if 'kernel' in line.lower():
            # Try to extract word before 'kernel'
            match = re.search(r'(\w+).*kernel', line, re.IGNORECASE)
            if match:
                return match.group(1)
        
        # Look for entry points
        if '.entry' in line:
            match = re.search(r'\.entry\s+(\w+)', line)
            if match:
                return match.group(1)
        
        return None
    
    def _classify_cuda_kernel(self, signature: str) -> str:
        """Classify CUDA kernel by function"""
        sig_lower = signature.lower()
        
        if any(term in sig_lower for term in ['rectify', 'undistort']):
            return 'image_rectification'
        elif any(term in sig_lower for term in ['pyramid', 'scale']):
            return 'image_pyramid'
        elif any(term in sig_lower for term in ['fusion', 'tsdf', 'voxel']):
            return 'volume_fusion'
        elif any(term in sig_lower for term in ['register', 'align', 'icp']):
            return 'registration'
        elif any(term in sig_lower for term in ['segment', 'classify']):
            return 'segmentation'
        elif any(term in sig_lower for term in ['mesh', 'triangle', 'vertex']):
            return 'mesh_processing'
        else:
            return 'general_compute'
    
    def _extract_cuda_parameters(self, signature: str) -> str:
        """Extract parameter hints from CUDA kernel signature"""
        hints = []
        
        if 'PtrStepSz' in signature:
            hints.append('gpu_memory_pointers')
        if 'float' in signature or 'If' in signature:
            hints.append('float_data')
        if 'int' in signature or 'Ih' in signature:
            hints.append('integer_data')
        if 'param' in signature:
            hints.append('kernel_parameters')
        
        return ', '.join(hints) if hints else 'unknown_parameters'
    
    def _infer_signature_from_name(self, function_name: str) -> str:
        """Infer function signature hints from naming patterns"""
        name_lower = function_name.lower()
        
        # C++ mangled names
        if function_name.startswith('_Z'):
            return "cpp_mangled_function"
        
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
        elif 'kernel' in name_lower or 'cuda' in name_lower:
            return "gpu_kernel_function"
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
        if 'cuda' in name_lower or 'gpu' in name_lower:
            hints.append("gpu_memory")
        
        return ", ".join(hints) if hints else "unknown_parameters"
    
    def update_database_with_exports(self, dll_name: str, exports: List[Dict], 
                                   cuda_kernels: List[Dict], confidence: float):
        """Update database with export analysis results"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Update export count in algorithm_dlls table
            cursor.execute('''
                UPDATE algorithm_dlls 
                SET export_count = ?, confidence_score = ?
                WHERE dll_name = ?
            ''', (len(exports), confidence, dll_name))
            
            # Clear existing exports for this DLL
            cursor.execute('DELETE FROM dll_exports WHERE dll_name = ?', (dll_name,))
            
            # Insert new exports
            for export in exports:
                cursor.execute('''
                    INSERT INTO dll_exports 
                    (dll_name, function_name, ordinal, signature_hint, parameter_hints, confidence_score)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (dll_name, export['function_name'], export.get('ordinal', 0),
                      export.get('signature_hint', ''), export.get('parameter_hints', ''), 
                      confidence))
            
            # Create CUDA kernels table if it doesn't exist
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS cuda_kernels (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    dll_name TEXT,
                    kernel_name TEXT,
                    full_signature TEXT,
                    kernel_type TEXT,
                    parameters TEXT,
                    FOREIGN KEY (dll_name) REFERENCES algorithm_dlls (dll_name)
                )
            ''')
            
            # Clear existing CUDA kernels for this DLL
            cursor.execute('DELETE FROM cuda_kernels WHERE dll_name = ?', (dll_name,))
            
            # Insert CUDA kernels
            for kernel in cuda_kernels:
                cursor.execute('''
                    INSERT INTO cuda_kernels 
                    (dll_name, kernel_name, full_signature, kernel_type, parameters)
                    VALUES (?, ?, ?, ?, ?)
                ''', (dll_name, kernel['kernel_name'], kernel['full_signature'],
                      kernel['kernel_type'], kernel['parameters']))
            
            conn.commit()
            logger.info(f"Updated database for {dll_name}: {len(exports)} exports, {len(cuda_kernels)} CUDA kernels")
            
        except sqlite3.Error as e:
            logger.error(f"Database error updating {dll_name}: {e}")
        finally:
            conn.close()
    
    def analyze_high_value_dlls(self) -> Dict:
        """Analyze high-value algorithm DLLs with comprehensive export analysis"""
        
        # High-value DLLs to focus on
        high_value_dlls = [
            'Sn3DSpeckleFusion.dll',
            'Sn3DRegistration.dll', 
            'Sn3DRealtimeScan.dll',
            'Sn3DDentalAI.dll',
            'Sn3DMagic.dll',
            'Sn3DCalibrationJR.dll',
            'algorithmLzy.dll',
            'algorithm1.dll',
            'algorithm2.dll'
        ]
        
        results = {
            'analyzed_dlls': [],
            'total_exports_found': 0,
            'total_cuda_kernels_found': 0,
            'analysis_summary': {}
        }
        
        bin_path = self.base_path / "IntraoralScan" / "Bin"
        
        for dll_name in high_value_dlls:
            dll_path = bin_path / dll_name
            if not dll_path.exists():
                logger.warning(f"DLL not found: {dll_path}")
                continue
            
            logger.info(f"Analyzing high-value DLL: {dll_name}")
            
            # Analyze with objdump
            objdump_exports, objdump_confidence = self.analyze_dll_with_objdump(dll_path)
            
            # Analyze with nm
            nm_exports, nm_confidence = self.analyze_dll_with_nm(dll_path)
            
            # Analyze CUDA kernels
            cuda_kernels, cuda_confidence = self.analyze_cuda_kernels(dll_path)
            
            # Combine results
            all_exports = objdump_exports + nm_exports
            
            # Remove duplicates based on function name
            unique_exports = []
            seen_names = set()
            for export in all_exports:
                if export['function_name'] not in seen_names:
                    unique_exports.append(export)
                    seen_names.add(export['function_name'])
            
            # Calculate combined confidence
            combined_confidence = max(objdump_confidence, nm_confidence, cuda_confidence)
            
            # Update database
            self.update_database_with_exports(dll_name, unique_exports, cuda_kernels, combined_confidence)
            
            # Add to results
            dll_result = {
                'dll_name': dll_name,
                'exports_found': len(unique_exports),
                'cuda_kernels_found': len(cuda_kernels),
                'confidence_score': combined_confidence,
                'sample_exports': [e['function_name'] for e in unique_exports[:5]],
                'sample_kernels': [k['kernel_name'] for k in cuda_kernels[:3]]
            }
            results['analyzed_dlls'].append(dll_result)
            results['total_exports_found'] += len(unique_exports)
            results['total_cuda_kernels_found'] += len(cuda_kernels)
        
        # Generate summary
        results['analysis_summary'] = {
            'dlls_analyzed': len(results['analyzed_dlls']),
            'average_exports_per_dll': results['total_exports_found'] / len(results['analyzed_dlls']) if results['analyzed_dlls'] else 0,
            'average_kernels_per_dll': results['total_cuda_kernels_found'] / len(results['analyzed_dlls']) if results['analyzed_dlls'] else 0
        }
        
        return results

def main():
    if len(sys.argv) != 2:
        print("Usage: python linux_dll_export_analyzer.py <base_path>")
        sys.exit(1)
    
    base_path = sys.argv[1]
    analyzer = LinuxDLLExportAnalyzer(base_path)
    
    # Run analysis on high-value DLLs
    results = analyzer.analyze_high_value_dlls()
    
    # Save results
    output_file = "analysis_output/linux_dll_export_analysis.json"
    os.makedirs("analysis_output", exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Linux DLL export analysis complete! Report saved to {output_file}")
    print(f"Total exports found: {results['total_exports_found']}")
    print(f"Total CUDA kernels found: {results['total_cuda_kernels_found']}")

if __name__ == "__main__":
    main()