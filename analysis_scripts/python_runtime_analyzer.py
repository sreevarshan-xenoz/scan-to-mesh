#!/usr/bin/env python3
"""
Python Runtime Analyzer for IntraoralScan Analysis
Implements task 4.3: Create Python runtime analyzer with robust fallbacks
"""
import os
import json
import sqlite3
import subprocess
import tempfile
import shutil
import struct
import marshal
import dis
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import importlib.util
import ast

class PythonRuntimeAnalyzer:
    def __init__(self, db_path="analysis_results.db"):
        self.db_path = db_path
        self.init_database()
        self.python_version = self.detect_python_version()
        
    def init_database(self):
        """Initialize database tables for Python analysis"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create Python modules table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS python_modules (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                module_name TEXT,
                module_path TEXT,
                module_type TEXT,
                file_size INTEGER,
                python_version TEXT,
                decompilation_method TEXT,
                decompilation_success BOOLEAN,
                imports TEXT,
                functions TEXT,
                classes TEXT,
                service_mappings TEXT,
                confidence_score REAL,
                analysis_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create PYD analysis table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS pyd_modules (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                module_name TEXT,
                module_path TEXT,
                file_size INTEGER,
                exported_functions TEXT,
                imported_libraries TEXT,
                analysis_method TEXT,
                c_extensions TEXT,
                confidence_score REAL,
                analysis_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create Python service mappings table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS python_service_mappings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                module_name TEXT,
                service_process TEXT,
                import_relationship TEXT,
                function_calls TEXT,
                data_flow TEXT,
                confidence_score REAL,
                analysis_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def detect_python_version(self) -> str:
        """Detect Python version used by the application"""
        version_indicators = [
            ("IntraoralScan/Bin/python38.dll", "3.8"),
            ("IntraoralScan/Bin/python39.dll", "3.9"),
            ("IntraoralScan/Bin/python37.dll", "3.7"),
            ("IntraoralScan/Bin/python3.dll", "3.x")
        ]
        
        for dll_path, version in version_indicators:
            if os.path.exists(dll_path):
                return version
        
        return "unknown"
    
    def find_python_files(self) -> Dict[str, List[str]]:
        """Find all Python-related files"""
        python_files = {
            'pyc_files': [],
            'pyd_files': [],
            'py_files': [],
            'python_dirs': []
        }
        
        # Find .pyc files
        bin_path = Path("IntraoralScan/Bin")
        if bin_path.exists():
            # Look in python directory
            python_dir = bin_path / "python"
            if python_dir.exists():
                python_files['python_dirs'].append(str(python_dir))
                for pyc_file in python_dir.rglob("*.pyc"):
                    python_files['pyc_files'].append(str(pyc_file))
                for py_file in python_dir.rglob("*.py"):
                    python_files['py_files'].append(str(py_file))
            
            # Find .pyd files in Bin directory
            for pyd_file in bin_path.glob("*.pyd"):
                python_files['pyd_files'].append(str(pyd_file))
        
        return python_files
    
    def decompile_with_uncompyle6(self, pyc_path: str) -> Tuple[Optional[str], float]:
        """Decompile .pyc file using uncompyle6"""
        try:
            # Check if uncompyle6 is available
            result = subprocess.run(['uncompyle6', '--version'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode != 0:
                return None, 0.0
            
            # Attempt decompilation
            result = subprocess.run(['uncompyle6', pyc_path], 
                                  capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0 and result.stdout:
                return result.stdout, 0.9
            else:
                return None, 0.1
                
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return None, 0.0
    
    def decompile_with_pycdc(self, pyc_path: str) -> Tuple[Optional[str], float]:
        """Fallback: Decompile .pyc file using pycdc"""
        try:
            # Check if pycdc is available
            result = subprocess.run(['pycdc', pyc_path], 
                                  capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0 and result.stdout:
                return result.stdout, 0.7
            else:
                return None, 0.1
                
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return None, 0.0
    
    def analyze_bytecode(self, pyc_path: str) -> Tuple[Optional[Dict], float]:
        """Fallback: Analyze bytecode without full decompilation"""
        try:
            with open(pyc_path, 'rb') as f:
                # Skip magic number and timestamp (first 8-16 bytes depending on version)
                magic = f.read(4)
                if self.python_version.startswith('3.8'):
                    f.read(12)  # Skip timestamp and size for Python 3.8+
                else:
                    f.read(8)   # Skip timestamp for older versions
                
                # Read and unmarshal code object
                code_data = f.read()
                try:
                    code_obj = marshal.loads(code_data)
                    
                    # Extract information from code object
                    analysis = {
                        'names': list(code_obj.co_names) if hasattr(code_obj, 'co_names') else [],
                        'varnames': list(code_obj.co_varnames) if hasattr(code_obj, 'co_varnames') else [],
                        'filename': code_obj.co_filename if hasattr(code_obj, 'co_filename') else '',
                        'constants': [str(const) for const in code_obj.co_consts 
                                    if isinstance(const, (str, int, float))] if hasattr(code_obj, 'co_consts') else []
                    }
                    
                    return analysis, 0.5
                    
                except (ValueError, EOFError):
                    return None, 0.1
                    
        except Exception as e:
            return None, 0.0
    
    def analyze_py_file(self, py_path: str) -> Tuple[Optional[Dict], float]:
        """Analyze .py file directly"""
        try:
            with open(py_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Parse AST
            try:
                tree = ast.parse(content)
                
                analysis = {
                    'imports': [],
                    'functions': [],
                    'classes': [],
                    'constants': []
                }
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            analysis['imports'].append(alias.name)
                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            analysis['imports'].append(node.module)
                    elif isinstance(node, ast.FunctionDef):
                        analysis['functions'].append(node.name)
                    elif isinstance(node, ast.ClassDef):
                        analysis['classes'].append(node.name)
                    elif isinstance(node, ast.Assign):
                        for target in node.targets:
                            if isinstance(target, ast.Name):
                                analysis['constants'].append(target.id)
                
                return analysis, 1.0
                
            except SyntaxError:
                # Fallback to simple text analysis
                return self.simple_text_analysis(content), 0.6
                
        except Exception as e:
            return None, 0.0
    
    def simple_text_analysis(self, content: str) -> Dict:
        """Simple text-based analysis of Python code"""
        import re
        
        analysis = {
            'imports': [],
            'functions': [],
            'classes': [],
            'constants': []
        }
        
        lines = content.split('\n')
        
        for line in lines:
            line = line.strip()
            
            # Find imports
            import_match = re.match(r'(?:from\s+(\S+)\s+)?import\s+([^#]+)', line)
            if import_match:
                if import_match.group(1):
                    analysis['imports'].append(import_match.group(1))
                imports = [imp.strip() for imp in import_match.group(2).split(',')]
                analysis['imports'].extend(imports)
            
            # Find function definitions
            func_match = re.match(r'def\s+(\w+)', line)
            if func_match:
                analysis['functions'].append(func_match.group(1))
            
            # Find class definitions
            class_match = re.match(r'class\s+(\w+)', line)
            if class_match:
                analysis['classes'].append(class_match.group(1))
        
        return analysis
    
    def analyze_pyd_with_ghidra(self, pyd_path: str) -> Tuple[Optional[Dict], float]:
        """Analyze .pyd file using Ghidra (if available)"""
        # This is a placeholder for Ghidra integration
        # In practice, this would require Ghidra headless analyzer
        return None, 0.0
    
    def analyze_pyd_with_strings(self, pyd_path: str) -> Tuple[Optional[Dict], float]:
        """Analyze .pyd file using strings command"""
        try:
            result = subprocess.run(['strings', '-n', '4', pyd_path], 
                                  capture_output=True, text=True, timeout=30)
            
            if result.returncode != 0:
                return None, 0.1
            
            content = result.stdout
            
            analysis = {
                'exported_functions': [],
                'imported_libraries': [],
                'python_api_calls': [],
                'error_messages': []
            }
            
            lines = content.split('\n')
            
            for line in lines:
                line = line.strip()
                
                # Look for Python API function patterns
                if line.startswith('Py') and len(line) > 3:
                    analysis['python_api_calls'].append(line)
                
                # Look for DLL imports
                if line.endswith('.dll') or line.endswith('.so'):
                    analysis['imported_libraries'].append(line)
                
                # Look for function-like patterns
                if ('(' in line and ')' in line and 
                    not line.startswith('/') and 
                    len(line) < 100):
                    analysis['exported_functions'].append(line)
                
                # Look for error messages
                if any(keyword in line.lower() for keyword in 
                       ['error', 'exception', 'failed', 'invalid']):
                    analysis['error_messages'].append(line)
            
            # Clean up duplicates and filter
            for key in analysis:
                analysis[key] = list(set(analysis[key]))[:20]  # Limit to 20 items
            
            confidence = 0.6 if any(analysis.values()) else 0.2
            return analysis, confidence
            
        except Exception as e:
            return None, 0.0
    
    def map_python_to_services(self, python_analysis: Dict) -> Dict:
        """Map Python modules to services based on imports and function calls"""
        service_mappings = {}
        
        for module_name, analysis in python_analysis.items():
            mappings = []
            
            # Check imports for service indicators
            imports = analysis.get('imports', [])
            for imp in imports:
                if any(keyword in imp.lower() for keyword in 
                       ['service', 'client', 'server', 'network', 'socket']):
                    mappings.append({
                        'type': 'service_import',
                        'target': imp,
                        'confidence': 0.8
                    })
                
                # Check for Qt/QML integration
                if any(keyword in imp.lower() for keyword in 
                       ['qt', 'qml', 'pyqt', 'pyside']):
                    mappings.append({
                        'type': 'ui_integration',
                        'target': imp,
                        'confidence': 0.9
                    })
            
            # Check functions for service patterns
            functions = analysis.get('functions', [])
            for func in functions:
                if any(keyword in func.lower() for keyword in 
                       ['connect', 'send', 'receive', 'handle', 'process']):
                    mappings.append({
                        'type': 'service_function',
                        'target': func,
                        'confidence': 0.7
                    })
            
            if mappings:
                service_mappings[module_name] = mappings
        
        return service_mappings
    
    def store_results(self, python_files: Dict, analysis_results: Dict, service_mappings: Dict):
        """Store Python analysis results in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Store Python module analysis
        for module_path, analysis in analysis_results.items():
            if analysis['type'] in ['pyc', 'py']:
                cursor.execute('''
                    INSERT INTO python_modules 
                    (module_name, module_path, module_type, file_size, python_version,
                     decompilation_method, decompilation_success, imports, functions, 
                     classes, service_mappings, confidence_score)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    Path(module_path).stem,
                    module_path,
                    analysis['type'],
                    analysis.get('file_size', 0),
                    self.python_version,
                    analysis.get('method', ''),
                    analysis.get('success', False),
                    json.dumps(analysis.get('imports', [])),
                    json.dumps(analysis.get('functions', [])),
                    json.dumps(analysis.get('classes', [])),
                    json.dumps(service_mappings.get(Path(module_path).stem, [])),
                    analysis.get('confidence', 0.0)
                ))
            
            elif analysis['type'] == 'pyd':
                cursor.execute('''
                    INSERT INTO pyd_modules 
                    (module_name, module_path, file_size, exported_functions,
                     imported_libraries, analysis_method, c_extensions, confidence_score)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    Path(module_path).stem,
                    module_path,
                    analysis.get('file_size', 0),
                    json.dumps(analysis.get('exported_functions', [])),
                    json.dumps(analysis.get('imported_libraries', [])),
                    analysis.get('method', ''),
                    json.dumps(analysis.get('python_api_calls', [])),
                    analysis.get('confidence', 0.0)
                ))
        
        # Store service mappings
        for module_name, mappings in service_mappings.items():
            for mapping in mappings:
                cursor.execute('''
                    INSERT INTO python_service_mappings 
                    (module_name, service_process, import_relationship, 
                     function_calls, data_flow, confidence_score)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    module_name,
                    mapping.get('target', ''),
                    mapping.get('type', ''),
                    json.dumps([]),
                    json.dumps([]),
                    mapping.get('confidence', 0.0)
                ))
        
        conn.commit()
        conn.close()
    
    def run_analysis(self) -> Dict:
        """Run complete Python runtime analysis"""
        results = {
            'python_version': self.python_version,
            'files_found': {},
            'analysis_results': {},
            'decompilation_success_rate': 0.0,
            'service_mappings_found': 0,
            'average_confidence': 0.0
        }
        
        # Find Python files
        python_files = self.find_python_files()
        results['files_found'] = {
            key: len(files) for key, files in python_files.items()
        }
        
        analysis_results = {}
        total_confidence = 0.0
        successful_decompilations = 0
        total_attempts = 0
        
        # Analyze .pyc files
        for pyc_path in python_files['pyc_files']:
            print(f"Analyzing {pyc_path}...")
            total_attempts += 1
            
            # Try uncompyle6 first
            decompiled_code, confidence = self.decompile_with_uncompyle6(pyc_path)
            method = 'uncompyle6'
            
            if confidence < 0.5:
                # Try pycdc fallback
                decompiled_code, confidence = self.decompile_with_pycdc(pyc_path)
                method = 'pycdc'
            
            if confidence < 0.5:
                # Try bytecode analysis fallback
                bytecode_analysis, confidence = self.analyze_bytecode(pyc_path)
                method = 'bytecode_analysis'
                
                if bytecode_analysis:
                    analysis_results[pyc_path] = {
                        'type': 'pyc',
                        'method': method,
                        'success': True,
                        'confidence': confidence,
                        'file_size': os.path.getsize(pyc_path),
                        **bytecode_analysis
                    }
                    successful_decompilations += 1
                else:
                    analysis_results[pyc_path] = {
                        'type': 'pyc',
                        'method': method,
                        'success': False,
                        'confidence': 0.1,
                        'file_size': os.path.getsize(pyc_path)
                    }
            else:
                # Successful decompilation - analyze the code
                if decompiled_code:
                    code_analysis = self.simple_text_analysis(decompiled_code)
                    analysis_results[pyc_path] = {
                        'type': 'pyc',
                        'method': method,
                        'success': True,
                        'confidence': confidence,
                        'file_size': os.path.getsize(pyc_path),
                        **code_analysis
                    }
                    successful_decompilations += 1
                else:
                    analysis_results[pyc_path] = {
                        'type': 'pyc',
                        'method': method,
                        'success': False,
                        'confidence': confidence,
                        'file_size': os.path.getsize(pyc_path)
                    }
            
            total_confidence += confidence
        
        # Analyze .py files
        for py_path in python_files['py_files']:
            print(f"Analyzing {py_path}...")
            py_analysis, confidence = self.analyze_py_file(py_path)
            
            if py_analysis:
                analysis_results[py_path] = {
                    'type': 'py',
                    'method': 'ast_analysis',
                    'success': True,
                    'confidence': confidence,
                    'file_size': os.path.getsize(py_path),
                    **py_analysis
                }
            else:
                analysis_results[py_path] = {
                    'type': 'py',
                    'method': 'ast_analysis',
                    'success': False,
                    'confidence': 0.1,
                    'file_size': os.path.getsize(py_path)
                }
            
            total_confidence += confidence
            total_attempts += 1
        
        # Analyze .pyd files
        for pyd_path in python_files['pyd_files']:
            print(f"Analyzing {pyd_path}...")
            
            # Try Ghidra analysis first (placeholder)
            pyd_analysis, confidence = self.analyze_pyd_with_ghidra(pyd_path)
            method = 'ghidra'
            
            if confidence < 0.5:
                # Fallback to strings analysis
                pyd_analysis, confidence = self.analyze_pyd_with_strings(pyd_path)
                method = 'strings_analysis'
            
            if pyd_analysis:
                analysis_results[pyd_path] = {
                    'type': 'pyd',
                    'method': method,
                    'success': True,
                    'confidence': confidence,
                    'file_size': os.path.getsize(pyd_path),
                    **pyd_analysis
                }
            else:
                analysis_results[pyd_path] = {
                    'type': 'pyd',
                    'method': method,
                    'success': False,
                    'confidence': 0.1,
                    'file_size': os.path.getsize(pyd_path)
                }
            
            total_confidence += confidence
            total_attempts += 1
        
        # Calculate success rate
        if total_attempts > 0:
            results['decompilation_success_rate'] = successful_decompilations / total_attempts
            results['average_confidence'] = total_confidence / total_attempts
        
        # Map Python modules to services
        service_mappings = self.map_python_to_services(analysis_results)
        results['service_mappings_found'] = len(service_mappings)
        
        # Store results
        self.store_results(python_files, analysis_results, service_mappings)
        
        results['analysis_results'] = analysis_results
        
        return results
    
    def generate_report(self) -> Dict:
        """Generate Python runtime analysis report"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        report = {
            'summary': {},
            'python_modules': [],
            'pyd_modules': [],
            'service_mappings': [],
            'decompilation_methods': {},
            'confidence_analysis': {}
        }
        
        # Get summary statistics
        cursor.execute('SELECT COUNT(*) FROM python_modules')
        python_module_count = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM pyd_modules')
        pyd_module_count = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM python_service_mappings')
        service_mapping_count = cursor.fetchone()[0]
        
        cursor.execute('SELECT AVG(confidence_score) FROM python_modules')
        avg_python_confidence = cursor.fetchone()[0] or 0.0
        
        cursor.execute('SELECT AVG(confidence_score) FROM pyd_modules')
        avg_pyd_confidence = cursor.fetchone()[0] or 0.0
        
        cursor.execute('SELECT COUNT(*) FROM python_modules WHERE decompilation_success = 1')
        successful_decompilations = cursor.fetchone()[0]
        
        report['summary'] = {
            'python_version': self.python_version,
            'python_modules_found': python_module_count,
            'pyd_modules_found': pyd_module_count,
            'service_mappings_found': service_mapping_count,
            'successful_decompilations': successful_decompilations,
            'decompilation_success_rate': successful_decompilations / max(python_module_count, 1),
            'average_python_confidence': round(avg_python_confidence, 2),
            'average_pyd_confidence': round(avg_pyd_confidence, 2)
        }
        
        # Get detailed Python modules
        cursor.execute('''
            SELECT module_name, module_path, file_size, decompilation_method,
                   decompilation_success, imports, functions, classes, confidence_score
            FROM python_modules ORDER BY confidence_score DESC, file_size DESC
        ''')
        
        for row in cursor.fetchall():
            imports = json.loads(row[5]) if row[5] else []
            functions = json.loads(row[6]) if row[6] else []
            classes = json.loads(row[7]) if row[7] else []
            
            report['python_modules'].append({
                'name': row[0],
                'path': row[1],
                'size_kb': round(row[2] / 1024, 1) if row[2] else 0,
                'decompilation_method': row[3],
                'decompilation_success': bool(row[4]),
                'imports_count': len(imports),
                'functions_count': len(functions),
                'classes_count': len(classes),
                'confidence': row[8]
            })
        
        # Get PYD modules
        cursor.execute('''
            SELECT module_name, module_path, file_size, exported_functions,
                   imported_libraries, analysis_method, confidence_score
            FROM pyd_modules ORDER BY confidence_score DESC, file_size DESC
        ''')
        
        for row in cursor.fetchall():
            exported_functions = json.loads(row[3]) if row[3] else []
            imported_libraries = json.loads(row[4]) if row[4] else []
            
            report['pyd_modules'].append({
                'name': row[0],
                'path': row[1],
                'size_kb': round(row[2] / 1024, 1) if row[2] else 0,
                'exported_functions_count': len(exported_functions),
                'imported_libraries_count': len(imported_libraries),
                'analysis_method': row[5],
                'confidence': row[6]
            })
        
        # Get decompilation method statistics
        cursor.execute('''
            SELECT decompilation_method, COUNT(*) as count,
                   AVG(confidence_score) as avg_confidence
            FROM python_modules 
            GROUP BY decompilation_method
        ''')
        
        for row in cursor.fetchall():
            report['decompilation_methods'][row[0]] = {
                'count': row[1],
                'average_confidence': round(row[2], 2)
            }
        
        conn.close()
        return report

if __name__ == "__main__":
    analyzer = PythonRuntimeAnalyzer()
    
    print("=== PYTHON RUNTIME ANALYSIS ===")
    print(f"Detected Python version: {analyzer.python_version}")
    print("Starting analysis...")
    
    # Run analysis
    results = analyzer.run_analysis()
    
    print(f"\nAnalysis Results:")
    print(f"- Python version: {results['python_version']}")
    print(f"- .pyc files found: {results['files_found'].get('pyc_files', 0)}")
    print(f"- .py files found: {results['files_found'].get('py_files', 0)}")
    print(f"- .pyd files found: {results['files_found'].get('pyd_files', 0)}")
    print(f"- Decompilation success rate: {results['decompilation_success_rate']:.2%}")
    print(f"- Service mappings found: {results['service_mappings_found']}")
    print(f"- Average confidence: {results['average_confidence']:.2f}")
    
    # Generate detailed report
    report = analyzer.generate_report()
    
    # Save report
    os.makedirs('analysis_output', exist_ok=True)
    with open('analysis_output/python_runtime_analysis.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n=== DETAILED ANALYSIS REPORT ===")
    print(f"Python Modules: {report['summary']['python_modules_found']}")
    print(f"PYD Modules: {report['summary']['pyd_modules_found']}")
    print(f"Service Mappings: {report['summary']['service_mappings_found']}")
    print(f"Successful Decompilations: {report['summary']['successful_decompilations']}")
    
    if report['python_modules']:
        print(f"\nTop Python Modules:")
        for module in report['python_modules'][:5]:
            print(f"- {module['name']} ({module['size_kb']} KB, "
                  f"{module['imports_count']} imports, "
                  f"{module['functions_count']} functions, "
                  f"confidence: {module['confidence']:.2f})")
    
    if report['pyd_modules']:
        print(f"\nPYD Modules:")
        for module in report['pyd_modules']:
            print(f"- {module['name']} ({module['size_kb']} KB, "
                  f"{module['exported_functions_count']} exports, "
                  f"confidence: {module['confidence']:.2f})")
    
    print(f"\nDecompilation Methods:")
    for method, stats in report['decompilation_methods'].items():
        print(f"- {method}: {stats['count']} modules "
              f"(avg confidence: {stats['average_confidence']:.2f})")
    
    print(f"\nDetailed report saved to: analysis_output/python_runtime_analysis.json")
    print("=== PYTHON RUNTIME ANALYSIS COMPLETE ===")