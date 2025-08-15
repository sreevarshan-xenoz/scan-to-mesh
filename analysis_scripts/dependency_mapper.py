#!/usr/bin/env python3
"""
Dependency mapping script for high-value IntraoralScan executables
Implements Dependencies tool integration with PE-sieve fallback
"""
import os
import json
import sqlite3
import subprocess
import sys
from pathlib import Path
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DependencyMapper:
    def __init__(self, db_path="analysis_results.db"):
        self.db_path = db_path
        self.init_database()
        
    def init_database(self):
        """Initialize SQLite database with required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables for storing analysis results
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS executables (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                path TEXT NOT NULL UNIQUE,
                size_bytes INTEGER,
                md5_hash TEXT,
                classification TEXT,
                confidence_score REAL,
                analysis_timestamp TEXT,
                file_type TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS dependencies (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                executable_id INTEGER,
                dependency_name TEXT NOT NULL,
                dependency_path TEXT,
                dependency_type TEXT, -- 'dll', 'exe', 'sys'
                is_system_library BOOLEAN,
                is_found BOOLEAN,
                analysis_method TEXT, -- 'dependencies', 'pe-sieve', 'manual'
                analysis_timestamp TEXT,
                FOREIGN KEY (executable_id) REFERENCES executables (id)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS dependency_analysis (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                executable_id INTEGER,
                total_dependencies INTEGER,
                system_dependencies INTEGER,
                missing_dependencies INTEGER,
                analysis_method TEXT,
                analysis_success BOOLEAN,
                error_message TEXT,
                analysis_timestamp TEXT,
                FOREIGN KEY (executable_id) REFERENCES executables (id)
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info(f"Database initialized at {self.db_path}")
    
    def get_high_value_targets(self):
        """Get top 5 high-value executables from previous analysis"""
        try:
            with open('analysis_output/high_value_analysis.json', 'r') as f:
                data = json.load(f)
            
            # Get top 5 executables by size and importance
            high_value_exes = data.get('high_value_executables', [])
            
            # Prioritize key executables
            priority_order = [
                'IntraoralScan.exe',
                'DentalAlgoService.exe', 
                'DentalScanAppLogic.exe',
                'DentalLauncher.exe',
                'DentalNetwork.exe'
            ]
            
            # Sort by priority, then by size
            sorted_exes = []
            for priority_exe in priority_order:
                for exe in high_value_exes:
                    if exe['name'] == priority_exe:
                        sorted_exes.append(exe)
                        break
            
            # Add remaining executables sorted by size
            remaining = [exe for exe in high_value_exes if exe not in sorted_exes]
            remaining.sort(key=lambda x: x.get('size_mb', 0), reverse=True)
            sorted_exes.extend(remaining)
            
            return sorted_exes[:5]  # Top 5
            
        except FileNotFoundError:
            logger.error("High value analysis file not found. Run high_value_analyzer.py first.")
            return []
        except Exception as e:
            logger.error(f"Error loading high value targets: {e}")
            return []
    
    def store_executable_info(self, exe_info):
        """Store executable information in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO executables 
            (name, path, size_bytes, md5_hash, classification, confidence_score, analysis_timestamp, file_type)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            exe_info['name'],
            exe_info['path'],
            exe_info.get('size_mb', 0) * 1024 * 1024,  # Convert MB to bytes
            exe_info.get('md5', ''),
            exe_info.get('classification', 'unknown'),
            exe_info.get('confidence', 0.0),
            datetime.now().isoformat(),
            exe_info.get('file_type', 'unknown')
        ))
        
        executable_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return executable_id
    
    def analyze_with_dependencies_tool(self, exe_path):
        """Analyze executable using Dependencies tool (if available)"""
        try:
            # Try to run Dependencies tool in command line mode
            # Note: This assumes Dependencies.exe is in PATH or we have a specific path
            dependencies_cmd = ['Dependencies.exe', '-json', exe_path]
            
            result = subprocess.run(dependencies_cmd, 
                                  capture_output=True, 
                                  text=True, 
                                  timeout=60)
            
            if result.returncode == 0:
                # Parse JSON output from Dependencies tool
                try:
                    deps_data = json.loads(result.stdout)
                    return self.parse_dependencies_output(deps_data), 'dependencies'
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse Dependencies tool output for {exe_path}")
                    return None, None
            else:
                logger.warning(f"Dependencies tool failed for {exe_path}: {result.stderr}")
                return None, None
                
        except FileNotFoundError:
            logger.info("Dependencies tool not found in PATH")
            return None, None
        except subprocess.TimeoutExpired:
            logger.warning(f"Dependencies tool timed out for {exe_path}")
            return None, None
        except Exception as e:
            logger.error(f"Error running Dependencies tool: {e}")
            return None, None
    
    def analyze_with_pe_sieve(self, exe_path):
        """Fallback analysis using PE-sieve or similar tools"""
        try:
            # Try pe-sieve for basic PE analysis
            pe_sieve_cmd = ['pe-sieve.exe', '/pid', '0', '/modules_filter', '3', '/json']
            
            result = subprocess.run(pe_sieve_cmd, 
                                  capture_output=True, 
                                  text=True, 
                                  timeout=30)
            
            if result.returncode == 0:
                # Parse pe-sieve output
                return self.parse_pe_sieve_output(result.stdout), 'pe-sieve'
            else:
                return None, None
                
        except FileNotFoundError:
            logger.info("PE-sieve not found, using manual analysis")
            return self.manual_dependency_analysis(exe_path), 'manual'
        except Exception as e:
            logger.error(f"Error with PE-sieve: {e}")
            return self.manual_dependency_analysis(exe_path), 'manual'
    
    def manual_dependency_analysis(self, exe_path):
        """Manual dependency analysis using available system tools"""
        dependencies = []
        
        try:
            # Use objdump or similar if available on Linux/WSL
            if os.name == 'posix':
                result = subprocess.run(['objdump', '-p', exe_path], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    # Parse objdump output for DLL names
                    for line in result.stdout.split('\n'):
                        if 'DLL Name:' in line:
                            dll_name = line.split('DLL Name:')[1].strip()
                            dependencies.append({
                                'name': dll_name,
                                'path': '',
                                'type': 'dll',
                                'is_system': self.is_system_library(dll_name),
                                'found': False  # Can't verify without full path
                            })
            
            # Use strings command to find potential DLL references
            try:
                result = subprocess.run(['strings', exe_path], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    # Look for .dll references in strings
                    for line in result.stdout.split('\n'):
                        if '.dll' in line.lower() and len(line) < 100:
                            # Extract potential DLL names
                            if line.lower().endswith('.dll'):
                                dll_name = line.strip()
                                if dll_name not in [d['name'] for d in dependencies]:
                                    dependencies.append({
                                        'name': dll_name,
                                        'path': '',
                                        'type': 'dll',
                                        'is_system': self.is_system_library(dll_name),
                                        'found': False
                                    })
            except:
                pass
                
        except Exception as e:
            logger.warning(f"Manual analysis failed for {exe_path}: {e}")
        
        return dependencies
    
    def parse_dependencies_output(self, deps_data):
        """Parse Dependencies tool JSON output"""
        dependencies = []
        
        # This would need to be adapted based on actual Dependencies tool output format
        # Placeholder implementation
        if isinstance(deps_data, dict) and 'modules' in deps_data:
            for module in deps_data['modules']:
                dependencies.append({
                    'name': module.get('name', ''),
                    'path': module.get('path', ''),
                    'type': 'dll',
                    'is_system': self.is_system_library(module.get('name', '')),
                    'found': module.get('found', True)
                })
        
        return dependencies
    
    def parse_pe_sieve_output(self, output):
        """Parse PE-sieve output"""
        dependencies = []
        # Placeholder - would need actual PE-sieve output format
        return dependencies
    
    def is_system_library(self, dll_name):
        """Determine if a DLL is a system library"""
        system_dlls = {
            'kernel32.dll', 'user32.dll', 'gdi32.dll', 'winspool.drv',
            'comdlg32.dll', 'advapi32.dll', 'shell32.dll', 'ole32.dll',
            'oleaut32.dll', 'uuid.dll', 'odbc32.dll', 'odbccp32.dll',
            'msvcrt.dll', 'msvcp140.dll', 'vcruntime140.dll',
            'api-ms-win-', 'ntdll.dll', 'ws2_32.dll'
        }
        
        dll_lower = dll_name.lower()
        return any(sys_dll in dll_lower for sys_dll in system_dlls)
    
    def store_dependencies(self, executable_id, dependencies, analysis_method):
        """Store dependency information in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for dep in dependencies:
            cursor.execute('''
                INSERT INTO dependencies 
                (executable_id, dependency_name, dependency_path, dependency_type, 
                 is_system_library, is_found, analysis_method, analysis_timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                executable_id,
                dep['name'],
                dep.get('path', ''),
                dep.get('type', 'dll'),
                dep.get('is_system', False),
                dep.get('found', False),
                analysis_method,
                datetime.now().isoformat()
            ))
        
        # Store analysis summary
        total_deps = len(dependencies)
        system_deps = sum(1 for d in dependencies if d.get('is_system', False))
        missing_deps = sum(1 for d in dependencies if not d.get('found', True))
        
        cursor.execute('''
            INSERT INTO dependency_analysis 
            (executable_id, total_dependencies, system_dependencies, missing_dependencies,
             analysis_method, analysis_success, analysis_timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            executable_id,
            total_deps,
            system_deps,
            missing_deps,
            analysis_method,
            True,
            datetime.now().isoformat()
        ))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Stored {total_deps} dependencies ({system_deps} system, {missing_deps} missing)")
    
    def analyze_executable_dependencies(self, exe_info):
        """Analyze dependencies for a single executable"""
        logger.info(f"Analyzing dependencies for {exe_info['name']}")
        
        # Store executable info
        executable_id = self.store_executable_info(exe_info)
        
        exe_path = exe_info['path']
        if not os.path.exists(exe_path):
            logger.error(f"Executable not found: {exe_path}")
            return False
        
        # Try Dependencies tool first
        dependencies, method = self.analyze_with_dependencies_tool(exe_path)
        
        # Fallback to PE-sieve or manual analysis
        if dependencies is None:
            logger.info(f"Falling back to alternative analysis for {exe_info['name']}")
            dependencies, method = self.analyze_with_pe_sieve(exe_path)
        
        if dependencies is None:
            logger.error(f"All analysis methods failed for {exe_info['name']}")
            return False
        
        # Store results
        self.store_dependencies(executable_id, dependencies, method)
        return True
    
    def generate_dependency_overview(self):
        """Generate quick dependency overview from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get overview statistics
        cursor.execute('''
            SELECT 
                e.name,
                e.classification,
                da.total_dependencies,
                da.system_dependencies,
                da.missing_dependencies,
                da.analysis_method,
                da.analysis_timestamp
            FROM executables e
            JOIN dependency_analysis da ON e.id = da.executable_id
            ORDER BY da.analysis_timestamp DESC
        ''')
        
        overview = []
        for row in cursor.fetchall():
            overview.append({
                'executable': row[0],
                'classification': row[1],
                'total_dependencies': row[2],
                'system_dependencies': row[3],
                'missing_dependencies': row[4],
                'analysis_method': row[5],
                'timestamp': row[6]
            })
        
        conn.close()
        return overview
    
    def run_analysis(self):
        """Run dependency analysis on high-value targets"""
        logger.info("Starting dependency mapping analysis")
        
        # Get high-value targets
        targets = self.get_high_value_targets()
        if not targets:
            logger.error("No high-value targets found")
            return False
        
        logger.info(f"Found {len(targets)} high-value targets for analysis")
        
        success_count = 0
        for exe_info in targets:
            try:
                if self.analyze_executable_dependencies(exe_info):
                    success_count += 1
                else:
                    logger.warning(f"Failed to analyze {exe_info['name']}")
            except Exception as e:
                logger.error(f"Error analyzing {exe_info['name']}: {e}")
        
        logger.info(f"Successfully analyzed {success_count}/{len(targets)} executables")
        
        # Generate overview
        overview = self.generate_dependency_overview()
        
        # Save overview to JSON for easy access
        with open('analysis_output/dependency_overview.json', 'w') as f:
            json.dump(overview, f, indent=2)
        
        logger.info("Dependency analysis complete. Results stored in database and overview file.")
        return success_count > 0

if __name__ == "__main__":
    mapper = DependencyMapper()
    success = mapper.run_analysis()
    
    if success:
        print("=== DEPENDENCY MAPPING COMPLETE ===")
        overview = mapper.generate_dependency_overview()
        
        print(f"\nAnalyzed {len(overview)} executables:")
        for item in overview:
            print(f"  {item['executable']}: {item['total_dependencies']} deps "
                  f"({item['system_dependencies']} system, {item['missing_dependencies']} missing) "
                  f"[{item['analysis_method']}]")
        
        print(f"\nResults stored in: {mapper.db_path}")
        print("Overview saved to: analysis_output/dependency_overview.json")
    else:
        print("Dependency analysis failed")
        sys.exit(1)