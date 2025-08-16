"""
Main Application v2 - Professional Intraoral Scanner
Enhanced with insights from IntraoralScan 3.5.4.6 reverse engineering analysis

Implements service-oriented architecture with:
- Scanning Service (DentalScanAppLogic.exe equivalent)
- AI Analysis Service (DentalAlgoService.exe equivalent)  
- Professional Qt6 UI (IntraoralScan.exe equivalent)
- Real-time performance monitoring
"""

import sys
import time
import threading
import multiprocessing as mp
from pathlib import Path
import argparse
import json

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from config.system_config import get_config, save_config
from services.scanning_service import ScanningService
from services.ai_analysis_service import AIAnalysisService
from ui.main_interface import MainInterface
from utils.service_manager import ServiceManager
from utils.performance_monitor import PerformanceMonitor

class IntraoralScannerV2:
    """
    Main application class implementing professional scanner architecture
    Based on analysis of IntraoralScan 3.5.4.6 multi-process design
    """
    
    def __init__(self):
        self.config = get_config()
        
        # Service manager for coordinating multiple processes
        self.service_manager = ServiceManager()
        
        # Performance monitoring
        self.performance_monitor = PerformanceMonitor()
        
        # Application state
        self.is_running = False
        self.services_started = False
        
        # Service processes
        self.scanning_process = None
        self.ai_analysis_process = None
        
        # Main UI (runs in main process)
        self.main_ui = None
        
    def initialize(self) -> bool:
        """Initialize the application and all services"""
        try:
            print("=== Intraoral Scanner v2.0 ===")
            print("Professional dental scanning with AI analysis")
            print("Based on IntraoralScan 3.5.4.6 reverse engineering\n")
            
            # Validate configuration
            config_issues = self.config.validate_config()
            if config_issues:
                print("Configuration issues found:")
                for issue in config_issues:
                    print(f"  - {issue}")
                print()
            
            # Initialize service manager
            if not self.service_manager.initialize():
                print("ERROR: Failed to initialize service manager")
                return False
            
            # Initialize performance monitoring
            self.performance_monitor.start_monitoring()
            
            print("✓ Application initialized successfully")
            return True
            
        except Exception as e:
            print(f"ERROR: Failed to initialize application: {e}")
            return False
    
    def start_services(self) -> bool:
        """Start all background services"""
        try:
            print("Starting services...")
            
            # Start scanning service (equivalent to DentalScanAppLogic.exe)
            print("  Starting scanning service...")
            self.scanning_process = mp.Process(
                target=self._run_scanning_service,
                args=(5555,)  # Port 5555
            )
            self.scanning_process.start()
            
            # Wait for scanning service to start
            time.sleep(2)
            
            # Start AI analysis service (equivalent to DentalAlgoService.exe)
            print("  Starting AI analysis service...")
            self.ai_analysis_process = mp.Process(
                target=self._run_ai_analysis_service,
                args=(5556,)  # Port 5556
            )
            self.ai_analysis_process.start()
            
            # Wait for AI service to start
            time.sleep(2)
            
            # Verify services are running
            if not self.service_manager.check_service_health("scanning", "tcp://localhost:5555"):
                print("ERROR: Scanning service failed to start")
                return False
            
            if not self.service_manager.check_service_health("ai_analysis", "tcp://localhost:5556"):
                print("ERROR: AI analysis service failed to start")
                return False
            
            self.services_started = True
            print("✓ All services started successfully")
            return True
            
        except Exception as e:
            print(f"ERROR: Failed to start services: {e}")
            return False
    
    def start_ui(self) -> bool:
        """Start the main user interface"""
        try:
            print("Starting main interface...")
            
            # Initialize Qt application and main interface
            self.main_ui = MainInterface()
            
            if not self.main_ui.initialize():
                print("ERROR: Failed to initialize main interface")
                return False
            
            print("✓ Main interface started")
            return True
            
        except Exception as e:
            print(f"ERROR: Failed to start UI: {e}")
            return False
    
    def run(self) -> int:
        """Main application run loop"""
        try:
            # Initialize application
            if not self.initialize():
                return 1
            
            # Start background services
            if not self.start_services():
                return 1
            
            # Start main UI
            if not self.start_ui():
                return 1
            
            # Run main event loop
            self.is_running = True
            print("\n=== Scanner Ready ===")
            print("Main interface is running...")
            print("Press Ctrl+C to shutdown\n")
            
            # Run Qt application
            exit_code = self.main_ui.run()
            
            return exit_code
            
        except KeyboardInterrupt:
            print("\nShutdown requested by user...")
            return 0
        except Exception as e:
            print(f"ERROR: Application error: {e}")
            return 1
        finally:
            self.shutdown()
    
    def shutdown(self):
        """Shutdown all services and cleanup"""
        print("Shutting down application...")
        
        self.is_running = False
        
        # Stop UI
        if self.main_ui:
            self.main_ui.shutdown()
        
        # Stop services
        if self.services_started:
            print("  Stopping services...")
            
            if self.scanning_process and self.scanning_process.is_alive():
                self.scanning_process.terminate()
                self.scanning_process.join(timeout=5)
                if self.scanning_process.is_alive():
                    self.scanning_process.kill()
            
            if self.ai_analysis_process and self.ai_analysis_process.is_alive():
                self.ai_analysis_process.terminate()
                self.ai_analysis_process.join(timeout=5)
                if self.ai_analysis_process.is_alive():
                    self.ai_analysis_process.kill()
        
        # Stop performance monitoring
        self.performance_monitor.stop_monitoring()
        
        # Cleanup service manager
        self.service_manager.cleanup()
        
        # Save configuration
        save_config()
        
        print("✓ Application shutdown complete")
    
    def _run_scanning_service(self, port: int):
        """Run scanning service in separate process"""
        try:
            service = ScanningService(service_port=port)
            if service.start_service():
                # Keep service running
                while True:
                    time.sleep(1)
        except KeyboardInterrupt:
            pass
        except Exception as e:
            print(f"Scanning service error: {e}")
        finally:
            if 'service' in locals():
                service.stop_service()
    
    def _run_ai_analysis_service(self, port: int):
        """Run AI analysis service in separate process"""
        try:
            service = AIAnalysisService(service_port=port)
            if service.start_service():
                # Keep service running
                while True:
                    time.sleep(1)
        except KeyboardInterrupt:
            pass
        except Exception as e:
            print(f"AI analysis service error: {e}")
        finally:
            if 'service' in locals():
                service.stop_service()

def create_default_config():
    """Create default configuration files"""
    config_dir = Path("config")
    config_dir.mkdir(exist_ok=True)
    
    # Create models directory
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # Create data directory
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Save default configuration
    config = get_config()
    config.save_config()
    
    print("✓ Default configuration created")

def check_dependencies():
    """Check if all required dependencies are available"""
    required_packages = [
        'opencv-python', 'open3d', 'numpy', 'scipy', 'zmq',
        'onnxruntime', 'PySide6', 'matplotlib'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'opencv-python':
                import cv2
            elif package == 'zmq':
                import zmq
            elif package == 'PySide6':
                import PySide6
            else:
                __import__(package.replace('-', '_'))
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} - MISSING")
            missing_packages.append(package)
    
    return missing_packages

def main():
    """Application entry point"""
    parser = argparse.ArgumentParser(description='Intraoral Scanner v2.0')
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--create-config', action='store_true', help='Create default configuration')
    parser.add_argument('--check-deps', action='store_true', help='Check dependencies')
    parser.add_argument('--service-only', type=str, choices=['scanning', 'ai'], 
                       help='Run only specific service')
    parser.add_argument('--port', type=int, help='Service port (for service-only mode)')
    
    args = parser.parse_args()
    
    # Create default configuration if requested
    if args.create_config:
        create_default_config()
        return 0
    
    # Check dependencies if requested
    if args.check_deps:
        missing = check_dependencies()
        if missing:
            print(f"\nMissing packages: {', '.join(missing)}")
            print("Install with: pip install " + " ".join(missing))
            return 1
        else:
            print("\n✓ All dependencies satisfied")
            return 0
    
    # Run specific service only
    if args.service_only:
        port = args.port or (5555 if args.service_only == 'scanning' else 5556)
        
        if args.service_only == 'scanning':
            service = ScanningService(service_port=port)
        else:
            service = AIAnalysisService(service_port=port)
        
        try:
            if service.start_service():
                print(f"{args.service_only} service running on port {port}")
                while True:
                    time.sleep(1)
        except KeyboardInterrupt:
            print(f"\nShutting down {args.service_only} service...")
        finally:
            service.stop_service()
        
        return 0
    
    # Run full application
    app = IntraoralScannerV2()
    return app.run()

if __name__ == "__main__":
    # Set multiprocessing start method for cross-platform compatibility
    if hasattr(mp, 'set_start_method'):
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass  # Already set
    
    exit(main())