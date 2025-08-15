#!/usr/bin/env python3
"""
Hardware Interface Analyzer for IntraoralScan Reverse Engineering
Analyzes driver DLLs for camera and scanner communication protocols
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

class HardwareInterfaceAnalyzer:
    def __init__(self, base_path: str = "IntraoralScan", db_path: str = "analysis_results.db"):
        self.base_path = Path(base_path)
        self.db_path = db_path
        self.init_database()
        
        # Hardware-related DLL patterns
        self.hardware_patterns = [
            # Camera and imaging
            "camera", "cam", "image", "capture", "video", "sensor",
            # USB and communication
            "usb", "hid", "communication", "comm", "serial", "port",
            # Device drivers
            "driver", "device", "hardware", "hw",
            # Scanning specific
            "scan", "scanner", "3d", "stereo", "calibrat",
            # Vendor specific
            "vimba", "ch375", "fx3", "vulkan"
        ]
        
        # Device capability keywords
        self.capability_keywords = [
            "resolution", "fps", "framerate", "exposure", "gain", "brightness",
            "calibration", "intrinsic", "extrinsic", "distortion",
            "depth", "stereo", "baseline", "focal", "principal",
            "usb3", "usb2", "bandwidth", "buffer", "stream",
            "trigger", "sync", "timestamp", "frame"
        ]

    def init_database(self):
        """Initialize SQLite database for storing hardware analysis results"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create table for hardware interface analysis
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS hardware_interfaces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                dll_name TEXT NOT NULL,
                dll_path TEXT NOT NULL,
                file_size INTEGER,
                md5_hash TEXT,
                interface_type TEXT,
                device_capabilities TEXT,
                communication_protocols TEXT,
                configuration_hints TEXT,
                confidence_score REAL,
                analysis_method TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create table for device configurations
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS device_configurations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_file TEXT NOT NULL,
                device_type TEXT,
                vendor_id TEXT,
                product_id TEXT,
                device_name TEXT,
                capabilities TEXT,
                configuration_data TEXT,
                confidence_score REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()

    def find_hardware_dlls(self) -> List[Path]:
        """Find all hardware-related DLL files"""
        hardware_dlls = []
        
        # Search in Bin directory
        bin_path = self.base_path / "Bin"
        if bin_path.exists():
            for dll_file in bin_path.glob("*.dll"):
                if self.is_hardware_related(dll_file.name):
                    hardware_dlls.append(dll_file)
        
        # Search in Driver directory
        driver_path = self.base_path / "Driver"
        if driver_path.exists():
            for dll_file in driver_path.rglob("*.dll"):
                hardware_dlls.append(dll_file)
            # Also look for .inf files
            for inf_file in driver_path.rglob("*.inf"):
                hardware_dlls.append(inf_file)
        
        return sorted(hardware_dlls)

    def is_hardware_related(self, filename: str) -> bool:
        """Check if a DLL is likely hardware-related"""
        filename_lower = filename.lower()
        
        for pattern in self.hardware_patterns:
            if pattern in filename_lower:
                return True
                
        return False

    def analyze_hardware_dll(self, dll_path: Path) -> Dict:
        """Analyze a hardware-related DLL or driver file"""
        result = {
            "dll_name": dll_path.name,
            "dll_path": str(dll_path),
            "file_size": dll_path.stat().st_size,
            "md5_hash": self.get_file_hash(dll_path),
            "interface_type": "unknown",
            "device_capabilities": [],
            "communication_protocols": [],
            "configuration_hints": [],
            "confidence_score": 0.0,
            "analysis_method": "string_analysis"
        }
        
        try:
            if dll_path.suffix.lower() == '.inf':
                result.update(self.analyze_inf_file(dll_path))
            else:
                result.update(self.analyze_dll_strings(dll_path))
                
            # Determine interface type
            result["interface_type"] = self.determine_interface_type(dll_path.name, result)
            
        except Exception as e:
            print(f"Error analyzing {dll_path.name}: {e}")
            result["confidence_score"] = 0.1
            
        return result

    def analyze_inf_file(self, inf_path: Path) -> Dict:
        """Analyze Windows driver .inf file"""
        result = {
            "device_capabilities": [],
            "communication_protocols": [],
            "configuration_hints": [],
            "confidence_score": 0.8,
            "analysis_method": "inf_file_parsing"
        }
        
        try:
            with open(inf_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                
            # Extract device IDs
            vendor_ids = re.findall(r'VID_([0-9A-Fa-f]{4})', content)
            product_ids = re.findall(r'PID_([0-9A-Fa-f]{4})', content)
            
            # Extract device descriptions
            descriptions = re.findall(r'"([^"]*(?:camera|scanner|3d|sensor)[^"]*)"', content, re.IGNORECASE)
            
            # Extract capability hints
            capabilities = []
            for keyword in self.capability_keywords:
                if keyword.lower() in content.lower():
                    capabilities.append(keyword)
            
            result["device_capabilities"] = capabilities
            result["configuration_hints"] = descriptions
            
            if vendor_ids or product_ids:
                result["communication_protocols"] = ["USB"]
                result["confidence_score"] = 0.9
                
        except Exception as e:
            print(f"Error parsing INF file {inf_path.name}: {e}")
            result["confidence_score"] = 0.2
            
        return result

    def analyze_dll_strings(self, dll_path: Path) -> Dict:
        """Analyze DLL using strings extraction"""
        result = {
            "device_capabilities": [],
            "communication_protocols": [],
            "configuration_hints": [],
            "confidence_score": 0.3
        }
        
        try:
            # Extract strings
            cmd = ["strings", "-n", "4", str(dll_path)]
            process = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if process.returncode == 0:
                strings_output = process.stdout
                
                # Analyze strings for hardware patterns
                capabilities = self.extract_device_capabilities(strings_output)
                protocols = self.extract_communication_protocols(strings_output)
                config_hints = self.extract_configuration_hints(strings_output)
                
                result["device_capabilities"] = capabilities
                result["communication_protocols"] = protocols
                result["configuration_hints"] = config_hints
                
                if capabilities or protocols or config_hints:
                    result["confidence_score"] = 0.6
                    
        except Exception as e:
            print(f"String analysis failed for {dll_path.name}: {e}")
            
        return result

    def extract_device_capabilities(self, strings_output: str) -> List[str]:
        """Extract device capability information from strings"""
        capabilities = []
        
        for line in strings_output.split('\n'):
            line = line.strip().lower()
            
            # Look for resolution patterns
            if re.search(r'\d{3,4}x\d{3,4}', line):
                capabilities.append(f"resolution: {line}")
                
            # Look for frame rate patterns
            if re.search(r'\d+\s*fps', line):
                capabilities.append(f"framerate: {line}")
                
            # Look for capability keywords
            for keyword in self.capability_keywords:
                if keyword in line and len(line) < 100:
                    capabilities.append(f"{keyword}: {line}")
                    
        return list(set(capabilities))[:20]  # Limit and deduplicate

    def extract_communication_protocols(self, strings_output: str) -> List[str]:
        """Extract communication protocol information"""
        protocols = []
        
        protocol_patterns = {
            "USB": [r'usb', r'vid_', r'pid_', r'endpoint'],
            "Serial": [r'com\d+', r'serial', r'uart', r'rs232'],
            "Ethernet": [r'tcp', r'udp', r'ip', r'ethernet'],
            "I2C": [r'i2c', r'iic'],
            "SPI": [r'spi'],
            "PCIe": [r'pci', r'pcie'],
            "Camera": [r'mipi', r'csi', r'camera'],
            "HID": [r'hid', r'human interface']
        }
        
        strings_lower = strings_output.lower()
        
        for protocol, patterns in protocol_patterns.items():
            for pattern in patterns:
                if re.search(pattern, strings_lower):
                    protocols.append(protocol)
                    break
                    
        return list(set(protocols))

    def extract_configuration_hints(self, strings_output: str) -> List[str]:
        """Extract configuration and device information hints"""
        hints = []
        
        for line in strings_output.split('\n'):
            line = line.strip()
            
            # Skip very short or very long lines
            if len(line) < 10 or len(line) > 200:
                continue
                
            # Look for device names and descriptions
            if any(keyword in line.lower() for keyword in ['camera', 'scanner', '3d', 'sensor', 'device']):
                if not any(char in line for char in ['\\', '/', '?', '*', '<', '>', '|']):
                    hints.append(line)
                    
            # Look for error messages and status strings
            if any(keyword in line.lower() for keyword in ['error', 'failed', 'success', 'ready', 'connected']):
                if len(line) < 100:
                    hints.append(line)
                    
        return list(set(hints))[:15]  # Limit and deduplicate

    def determine_interface_type(self, filename: str, analysis_result: Dict) -> str:
        """Determine the type of hardware interface"""
        filename_lower = filename.lower()
        protocols = analysis_result.get("communication_protocols", [])
        capabilities = analysis_result.get("device_capabilities", [])
        
        # Camera interfaces
        if any(keyword in filename_lower for keyword in ['camera', 'cam', 'image', 'capture']):
            return "camera_interface"
            
        # USB interfaces
        if "USB" in protocols or any(keyword in filename_lower for keyword in ['usb', 'hid']):
            return "usb_interface"
            
        # Scanner interfaces
        if any(keyword in filename_lower for keyword in ['scan', '3d', 'stereo']):
            return "scanner_interface"
            
        # Communication interfaces
        if any(keyword in filename_lower for keyword in ['comm', 'communication']):
            return "communication_interface"
            
        # Driver interfaces
        if "driver" in filename_lower or filename_lower.endswith('.inf'):
            return "device_driver"
            
        # Calibration interfaces
        if "calibrat" in filename_lower:
            return "calibration_interface"
            
        return "unknown_interface"

    def analyze_device_configurations(self) -> List[Dict]:
        """Analyze device configuration files"""
        configurations = []
        
        # Look for device configuration files
        config_paths = [
            self.base_path / "Bin" / "config",
            self.base_path / "Bin" / "device",
            self.base_path / "Driver"
        ]
        
        for config_path in config_paths:
            if config_path.exists():
                for config_file in config_path.rglob("*"):
                    if config_file.is_file() and config_file.suffix.lower() in ['.json', '.xml', '.ini', '.cfg', '.conf']:
                        config_data = self.analyze_config_file(config_file)
                        if config_data:
                            configurations.append(config_data)
                            
        return configurations

    def analyze_config_file(self, config_path: Path) -> Optional[Dict]:
        """Analyze a device configuration file"""
        try:
            with open(config_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                
            # Look for device-related configuration
            if not any(keyword in content.lower() for keyword in ['device', 'camera', 'scanner', 'usb', 'driver']):
                return None
                
            config_data = {
                "source_file": str(config_path),
                "device_type": "unknown",
                "vendor_id": None,
                "product_id": None,
                "device_name": None,
                "capabilities": [],
                "configuration_data": content[:1000],  # First 1000 chars
                "confidence_score": 0.5
            }
            
            # Extract device identifiers
            vendor_match = re.search(r'(?:vendor|vid)["\s:=]+([0-9a-fA-F]{4})', content)
            if vendor_match:
                config_data["vendor_id"] = vendor_match.group(1)
                
            product_match = re.search(r'(?:product|pid)["\s:=]+([0-9a-fA-F]{4})', content)
            if product_match:
                config_data["product_id"] = product_match.group(1)
                
            # Extract device name
            name_patterns = [
                r'"name"\s*:\s*"([^"]+)"',
                r'<name>([^<]+)</name>',
                r'name\s*=\s*([^\n\r]+)'
            ]
            
            for pattern in name_patterns:
                name_match = re.search(pattern, content, re.IGNORECASE)
                if name_match:
                    config_data["device_name"] = name_match.group(1).strip()
                    break
                    
            # Extract capabilities
            capabilities = []
            for keyword in self.capability_keywords:
                if keyword in content.lower():
                    capabilities.append(keyword)
                    
            config_data["capabilities"] = capabilities
            
            if config_data["vendor_id"] or config_data["product_id"] or config_data["device_name"]:
                config_data["confidence_score"] = 0.8
                
            return config_data
            
        except Exception as e:
            print(f"Error analyzing config file {config_path.name}: {e}")
            return None

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

    def save_analysis_results(self, hardware_results: List[Dict], config_results: List[Dict]):
        """Save analysis results to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Save hardware interface results
        for result in hardware_results:
            cursor.execute('''
                INSERT OR REPLACE INTO hardware_interfaces 
                (dll_name, dll_path, file_size, md5_hash, interface_type, 
                 device_capabilities, communication_protocols, configuration_hints, 
                 confidence_score, analysis_method)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                result["dll_name"],
                result["dll_path"],
                result["file_size"],
                result["md5_hash"],
                result["interface_type"],
                json.dumps(result["device_capabilities"]),
                json.dumps(result["communication_protocols"]),
                json.dumps(result["configuration_hints"]),
                result["confidence_score"],
                result["analysis_method"]
            ))
        
        # Save device configuration results
        for config in config_results:
            cursor.execute('''
                INSERT OR REPLACE INTO device_configurations
                (source_file, device_type, vendor_id, product_id, device_name, 
                 capabilities, configuration_data, confidence_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                config["source_file"],
                config["device_type"],
                config["vendor_id"],
                config["product_id"],
                config["device_name"],
                json.dumps(config["capabilities"]),
                config["configuration_data"],
                config["confidence_score"]
            ))
        
        conn.commit()
        conn.close()

    def generate_hardware_report(self) -> Dict:
        """Generate comprehensive hardware interface report"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get hardware interface summary
        cursor.execute('SELECT COUNT(*) FROM hardware_interfaces')
        total_interfaces = cursor.fetchone()[0]
        
        cursor.execute('SELECT interface_type, COUNT(*) FROM hardware_interfaces GROUP BY interface_type')
        interface_distribution = dict(cursor.fetchall())
        
        cursor.execute('SELECT COUNT(*) FROM device_configurations')
        total_configs = cursor.fetchone()[0]
        
        # Get high-confidence hardware interfaces
        cursor.execute('''
            SELECT dll_name, interface_type, communication_protocols, confidence_score 
            FROM hardware_interfaces 
            WHERE confidence_score > 0.5 
            ORDER BY confidence_score DESC
        ''')
        high_confidence_interfaces = cursor.fetchall()
        
        # Get device configurations
        cursor.execute('''
            SELECT device_name, vendor_id, product_id, capabilities, confidence_score
            FROM device_configurations 
            WHERE confidence_score > 0.5
            ORDER BY confidence_score DESC
        ''')
        device_configs = cursor.fetchall()
        
        conn.close()
        
        return {
            "summary": {
                "total_hardware_interfaces": total_interfaces,
                "interface_type_distribution": interface_distribution,
                "total_device_configurations": total_configs
            },
            "high_confidence_interfaces": [
                {
                    "dll_name": dll,
                    "interface_type": itype,
                    "protocols": json.loads(protocols) if protocols else [],
                    "confidence": conf
                }
                for dll, itype, protocols, conf in high_confidence_interfaces
            ],
            "device_configurations": [
                {
                    "device_name": name,
                    "vendor_id": vid,
                    "product_id": pid,
                    "capabilities": json.loads(caps) if caps else [],
                    "confidence": conf
                }
                for name, vid, pid, caps, conf in device_configs
            ]
        }

    def run_analysis(self) -> Dict:
        """Run complete hardware interface analysis"""
        print("Starting Hardware Interface Analysis...")
        
        # Find hardware-related files
        hardware_files = self.find_hardware_dlls()
        print(f"Found {len(hardware_files)} hardware-related files to analyze")
        
        # Analyze hardware DLLs
        hardware_results = []
        for i, hw_file in enumerate(hardware_files, 1):
            print(f"[{i}/{len(hardware_files)}] Analyzing {hw_file.name}...")
            
            try:
                result = self.analyze_hardware_dll(hw_file)
                hardware_results.append(result)
                
                print(f"  - Interface type: {result['interface_type']}")
                print(f"  - Protocols: {result['communication_protocols']}")
                print(f"  - Confidence: {result['confidence_score']:.2f}")
                
            except Exception as e:
                print(f"  - Error: {e}")
                continue
        
        # Analyze device configurations
        print("Analyzing device configuration files...")
        config_results = self.analyze_device_configurations()
        print(f"Found {len(config_results)} device configuration files")
        
        # Save results
        print("Saving analysis results to database...")
        self.save_analysis_results(hardware_results, config_results)
        
        # Generate report
        print("Generating hardware interface report...")
        report = self.generate_hardware_report()
        
        return report

def main():
    if len(sys.argv) > 1:
        base_path = sys.argv[1]
    else:
        base_path = "IntraoralScan"
    
    analyzer = HardwareInterfaceAnalyzer(base_path)
    report = analyzer.run_analysis()
    
    # Print summary
    print("\n" + "="*60)
    print("HARDWARE INTERFACE ANALYSIS SUMMARY")
    print("="*60)
    print(f"Total hardware interfaces: {report['summary']['total_hardware_interfaces']}")
    print(f"Interface type distribution: {report['summary']['interface_type_distribution']}")
    print(f"Device configurations found: {report['summary']['total_device_configurations']}")
    
    print(f"\nHigh-confidence hardware interfaces:")
    for interface in report['high_confidence_interfaces'][:10]:
        print(f"  {interface['dll_name']}: {interface['interface_type']} - {interface['protocols']}")
    
    print(f"\nDevice configurations:")
    for device in report['device_configurations'][:5]:
        print(f"  {device['device_name']}: VID={device['vendor_id']}, PID={device['product_id']}")
    
    # Save report to JSON
    with open("analysis_output/hardware_interface_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"\nDetailed report saved to: analysis_output/hardware_interface_report.json")
    print("Database updated with hardware interface analysis results.")

if __name__ == "__main__":
    main()