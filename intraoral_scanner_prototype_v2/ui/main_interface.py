"""
Main Interface - Professional Qt6 UI based on IntraoralScan.exe analysis
Implements modern dental scanning interface with real-time 3D visualization
"""

import sys
import time
import threading
from typing import Optional, Dict, Any
import numpy as np

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QGridLayout, QPushButton, QLabel, QProgressBar, QTextEdit,
    QGroupBox, QSlider, QSpinBox, QComboBox, QCheckBox, QTabWidget,
    QSplitter, QFrame, QStatusBar, QMenuBar, QToolBar, QAction
)
from PySide6.QtCore import Qt, QTimer, Signal, QThread, pyqtSignal
from PySide6.QtGui import QFont, QPalette, QColor, QPixmap, QIcon

import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

from config.system_config import get_config
from utils.service_client import ServiceClient
from utils.performance_monitor import PerformanceMonitor

class ScanVisualizationWidget(QWidget):
    """3D visualization widget using Open3D"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.config = get_config()
        
        # Initialize Open3D visualization
        self.vis_widget = None
        self.scene = None
        self.current_mesh = None
        self.point_cloud = None
        
        self.setup_ui()
        self.initialize_3d_view()
    
    def setup_ui(self):
        """Setup the 3D visualization UI"""
        layout = QVBoxLayout(self)
        
        # Controls panel
        controls_frame = QFrame()
        controls_layout = QHBoxLayout(controls_frame)
        
        # View controls
        self.mesh_checkbox = QCheckBox("Show Mesh")
        self.mesh_checkbox.setChecked(True)
        self.mesh_checkbox.toggled.connect(self.toggle_mesh)
        
        self.points_checkbox = QCheckBox("Show Points")
        self.points_checkbox.setChecked(False)
        self.points_checkbox.toggled.connect(self.toggle_points)
        
        self.wireframe_checkbox = QCheckBox("Wireframe")
        self.wireframe_checkbox.setChecked(False)
        self.wireframe_checkbox.toggled.connect(self.toggle_wireframe)
        
        controls_layout.addWidget(self.mesh_checkbox)
        controls_layout.addWidget(self.points_checkbox)
        controls_layout.addWidget(self.wireframe_checkbox)
        controls_layout.addStretch()
        
        layout.addWidget(controls_frame)
        
        # 3D view placeholder (Open3D will be embedded here)
        self.view_frame = QFrame()
        self.view_frame.setMinimumSize(800, 600)
        self.view_frame.setStyleSheet("background-color: #2b2b2b; border: 1px solid #555;")
        layout.addWidget(self.view_frame)
    
    def initialize_3d_view(self):
        """Initialize Open3D 3D visualization"""
        try:
            # This would integrate Open3D visualization
            # For now, placeholder implementation
            pass
        except Exception as e:
            print(f"Error initializing 3D view: {e}")
    
    def update_mesh(self, mesh_data: Dict[str, Any]):
        """Update the displayed 3D mesh"""
        try:
            # Update mesh visualization
            pass
        except Exception as e:
            print(f"Error updating mesh: {e}")
    
    def update_point_cloud(self, points: np.ndarray, colors: np.ndarray):
        """Update the displayed point cloud"""
        try:
            # Update point cloud visualization
            pass
        except Exception as e:
            print(f"Error updating point cloud: {e}")
    
    def toggle_mesh(self, checked: bool):
        """Toggle mesh visibility"""
        # Implementation for mesh visibility toggle
        pass
    
    def toggle_points(self, checked: bool):
        """Toggle point cloud visibility"""
        # Implementation for point cloud visibility toggle
        pass
    
    def toggle_wireframe(self, checked: bool):
        """Toggle wireframe mode"""
        # Implementation for wireframe toggle
        pass

class ScanControlPanel(QWidget):
    """Scanning control panel with professional controls"""
    
    scan_started = Signal()
    scan_stopped = Signal()
    scan_paused = Signal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.config = get_config()
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the scan control UI"""
        layout = QVBoxLayout(self)
        
        # Scan controls group
        scan_group = QGroupBox("Scan Controls")
        scan_layout = QVBoxLayout(scan_group)
        
        # Main scan buttons
        button_layout = QHBoxLayout()
        
        self.start_button = QPushButton("Start Scan")
        self.start_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 10px;
                font-size: 14px;
                font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        self.start_button.clicked.connect(self.start_scan)
        
        self.stop_button = QPushButton("Stop Scan")
        self.stop_button.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                border: none;
                padding: 10px;
                font-size: 14px;
                font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #da190b;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        self.stop_button.clicked.connect(self.stop_scan)
        self.stop_button.setEnabled(False)
        
        self.pause_button = QPushButton("Pause")
        self.pause_button.setStyleSheet("""
            QPushButton {
                background-color: #ff9800;
                color: white;
                border: none;
                padding: 10px;
                font-size: 14px;
                font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #e68900;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        self.pause_button.clicked.connect(self.pause_scan)
        self.pause_button.setEnabled(False)
        
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.pause_button)
        button_layout.addWidget(self.stop_button)
        
        scan_layout.addLayout(button_layout)
        
        # Scan type selection
        type_layout = QHBoxLayout()
        type_layout.addWidget(QLabel("Scan Type:"))
        
        self.scan_type_combo = QComboBox()
        self.scan_type_combo.addItems(["Full Arch", "Quadrant", "Single Tooth", "Bite Registration"])
        type_layout.addWidget(self.scan_type_combo)
        
        scan_layout.addLayout(type_layout)
        
        # Quality settings
        quality_layout = QHBoxLayout()
        quality_layout.addWidget(QLabel("Quality:"))
        
        self.quality_combo = QComboBox()
        self.quality_combo.addItems(["Draft", "Standard", "High", "Ultra"])
        self.quality_combo.setCurrentText("High")
        quality_layout.addWidget(self.quality_combo)
        
        scan_layout.addLayout(quality_layout)
        
        layout.addWidget(scan_group)
        
        # Progress group
        progress_group = QGroupBox("Scan Progress")
        progress_layout = QVBoxLayout(progress_group)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        progress_layout.addWidget(self.progress_bar)
        
        self.status_label = QLabel("Ready to scan")
        self.status_label.setStyleSheet("color: #666; font-size: 12px;")
        progress_layout.addWidget(self.status_label)
        
        layout.addWidget(progress_group)
        
        # Statistics group
        stats_group = QGroupBox("Statistics")
        stats_layout = QGridLayout(stats_group)
        
        self.frames_label = QLabel("Frames: 0")
        self.fps_label = QLabel("FPS: 0.0")
        self.points_label = QLabel("Points: 0")
        self.vertices_label = QLabel("Vertices: 0")
        
        stats_layout.addWidget(self.frames_label, 0, 0)
        stats_layout.addWidget(self.fps_label, 0, 1)
        stats_layout.addWidget(self.points_label, 1, 0)
        stats_layout.addWidget(self.vertices_label, 1, 1)
        
        layout.addWidget(stats_group)
        
        layout.addStretch()
    
    def start_scan(self):
        """Start scanning"""
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.pause_button.setEnabled(True)
        self.status_label.setText("Scanning...")
        self.scan_started.emit()
    
    def stop_scan(self):
        """Stop scanning"""
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.pause_button.setEnabled(False)
        self.status_label.setText("Scan completed")
        self.progress_bar.setValue(100)
        self.scan_stopped.emit()
    
    def pause_scan(self):
        """Pause scanning"""
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(True)
        self.pause_button.setEnabled(False)
        self.status_label.setText("Scan paused")
        self.scan_paused.emit()
    
    def update_statistics(self, stats: Dict[str, Any]):
        """Update scan statistics"""
        self.frames_label.setText(f"Frames: {stats.get('frames', 0)}")
        self.fps_label.setText(f"FPS: {stats.get('fps', 0.0):.1f}")
        self.points_label.setText(f"Points: {stats.get('points', 0):,}")
        self.vertices_label.setText(f"Vertices: {stats.get('vertices', 0):,}")
        
        # Update progress based on scan completeness
        progress = min(100, stats.get('progress', 0))
        self.progress_bar.setValue(progress)

class AIAnalysisPanel(QWidget):
    """AI analysis results panel"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
    
    def setup_ui(self):
        """Setup AI analysis UI"""
        layout = QVBoxLayout(self)
        
        # Analysis controls
        controls_group = QGroupBox("AI Analysis")
        controls_layout = QVBoxLayout(controls_group)
        
        self.auto_analysis_checkbox = QCheckBox("Real-time Analysis")
        self.auto_analysis_checkbox.setChecked(True)
        controls_layout.addWidget(self.auto_analysis_checkbox)
        
        self.analyze_button = QPushButton("Analyze Current Scan")
        self.analyze_button.clicked.connect(self.analyze_scan)
        controls_layout.addWidget(self.analyze_button)
        
        layout.addWidget(controls_group)
        
        # Results display
        results_group = QGroupBox("Analysis Results")
        results_layout = QVBoxLayout(results_group)
        
        self.results_text = QTextEdit()
        self.results_text.setMaximumHeight(200)
        self.results_text.setReadOnly(True)
        results_layout.addWidget(self.results_text)
        
        layout.addWidget(results_group)
        
        # Tooth detection results
        teeth_group = QGroupBox("Detected Teeth")
        teeth_layout = QVBoxLayout(teeth_group)
        
        self.teeth_count_label = QLabel("Count: 0")
        self.teeth_confidence_label = QLabel("Confidence: 0%")
        
        teeth_layout.addWidget(self.teeth_count_label)
        teeth_layout.addWidget(self.teeth_confidence_label)
        
        layout.addWidget(teeth_group)
        
        layout.addStretch()
    
    def analyze_scan(self):
        """Trigger manual scan analysis"""
        self.results_text.append("Starting AI analysis...")
        # Implementation for manual analysis trigger
    
    def update_analysis_results(self, results: Dict[str, Any]):
        """Update AI analysis results"""
        if 'segmentation' in results:
            seg_results = results['segmentation']
            teeth_count = seg_results.get('teeth_count', 0)
            confidence = seg_results.get('confidence', 0.0) * 100
            
            self.teeth_count_label.setText(f"Count: {teeth_count}")
            self.teeth_confidence_label.setText(f"Confidence: {confidence:.1f}%")
        
        if 'clinical_analysis' in results:
            clinical = results['clinical_analysis']
            findings = clinical.get('findings', [])
            
            if findings:
                self.results_text.append("Clinical Findings:")
                for finding in findings:
                    self.results_text.append(f"  • {finding}")

class MainInterface(QMainWindow):
    """Main application interface"""
    
    def __init__(self):
        super().__init__()
        self.config = get_config()
        
        # Service clients for communication
        self.scanning_client = ServiceClient("tcp://localhost:5555")
        self.ai_client = ServiceClient("tcp://localhost:5556")
        
        # Performance monitoring
        self.performance_monitor = PerformanceMonitor()
        
        # UI update timer
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_ui)
        
        # Application state
        self.is_scanning = False
        self.scan_session_id = None
        
        self.setup_ui()
        self.setup_connections()
    
    def initialize(self) -> bool:
        """Initialize the main interface"""
        try:
            # Connect to services
            if not self.scanning_client.connect():
                print("Warning: Could not connect to scanning service")
            
            if not self.ai_client.connect():
                print("Warning: Could not connect to AI analysis service")
            
            # Start UI update timer
            self.update_timer.start(100)  # 10 FPS UI updates
            
            return True
            
        except Exception as e:
            print(f"Error initializing main interface: {e}")
            return False
    
    def setup_ui(self):
        """Setup the main user interface"""
        self.setWindowTitle("Intraoral Scanner v2.0 - Professional Edition")
        self.setGeometry(100, 100, 1600, 1000)
        
        # Apply professional dark theme
        self.apply_dark_theme()
        
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout(central_widget)
        
        # Create splitter for resizable panels
        splitter = QSplitter(Qt.Horizontal)
        
        # Left panel - Controls
        left_panel = QWidget()
        left_panel.setMaximumWidth(350)
        left_layout = QVBoxLayout(left_panel)
        
        # Scan control panel
        self.scan_control = ScanControlPanel()
        left_layout.addWidget(self.scan_control)
        
        # AI analysis panel
        self.ai_panel = AIAnalysisPanel()
        left_layout.addWidget(self.ai_panel)
        
        splitter.addWidget(left_panel)
        
        # Right panel - 3D Visualization
        self.visualization_widget = ScanVisualizationWidget()
        splitter.addWidget(self.visualization_widget)
        
        # Set splitter proportions
        splitter.setSizes([350, 1250])
        
        main_layout.addWidget(splitter)
        
        # Setup menu bar
        self.setup_menu_bar()
        
        # Setup status bar
        self.setup_status_bar()
        
        # Setup toolbar
        self.setup_toolbar()
    
    def apply_dark_theme(self):
        """Apply professional dark theme"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2b2b2b;
                color: #ffffff;
            }
            QWidget {
                background-color: #2b2b2b;
                color: #ffffff;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #555555;
                border-radius: 5px;
                margin-top: 1ex;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            QPushButton {
                background-color: #404040;
                border: 1px solid #555555;
                padding: 8px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #505050;
            }
            QPushButton:pressed {
                background-color: #353535;
            }
            QComboBox {
                background-color: #404040;
                border: 1px solid #555555;
                padding: 5px;
                border-radius: 3px;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 5px solid #ffffff;
            }
            QProgressBar {
                border: 1px solid #555555;
                border-radius: 3px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
                border-radius: 2px;
            }
            QTextEdit {
                background-color: #353535;
                border: 1px solid #555555;
                border-radius: 3px;
            }
            QLabel {
                color: #ffffff;
            }
            QCheckBox {
                spacing: 5px;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
            }
            QCheckBox::indicator:unchecked {
                border: 2px solid #555555;
                background-color: #2b2b2b;
                border-radius: 3px;
            }
            QCheckBox::indicator:checked {
                border: 2px solid #4CAF50;
                background-color: #4CAF50;
                border-radius: 3px;
            }
        """)
    
    def setup_menu_bar(self):
        """Setup application menu bar"""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu('File')
        
        new_scan_action = QAction('New Scan', self)
        new_scan_action.setShortcut('Ctrl+N')
        new_scan_action.triggered.connect(self.new_scan)
        file_menu.addAction(new_scan_action)
        
        open_scan_action = QAction('Open Scan...', self)
        open_scan_action.setShortcut('Ctrl+O')
        file_menu.addAction(open_scan_action)
        
        file_menu.addSeparator()
        
        export_action = QAction('Export...', self)
        export_action.setShortcut('Ctrl+E')
        file_menu.addAction(export_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction('Exit', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # View menu
        view_menu = menubar.addMenu('View')
        
        fullscreen_action = QAction('Fullscreen', self)
        fullscreen_action.setShortcut('F11')
        fullscreen_action.setCheckable(True)
        fullscreen_action.triggered.connect(self.toggle_fullscreen)
        view_menu.addAction(fullscreen_action)
        
        # Tools menu
        tools_menu = menubar.addMenu('Tools')
        
        calibrate_action = QAction('Camera Calibration...', self)
        tools_menu.addAction(calibrate_action)
        
        settings_action = QAction('Settings...', self)
        settings_action.triggered.connect(self.show_settings)
        tools_menu.addAction(settings_action)
        
        # Help menu
        help_menu = menubar.addMenu('Help')
        
        about_action = QAction('About...', self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
    
    def setup_status_bar(self):
        """Setup status bar"""
        self.status_bar = self.statusBar()
        
        # Service status indicators
        self.scanning_status = QLabel("Scanning: Disconnected")
        self.ai_status = QLabel("AI: Disconnected")
        self.performance_status = QLabel("FPS: 0.0")
        
        self.status_bar.addWidget(self.scanning_status)
        self.status_bar.addWidget(self.ai_status)
        self.status_bar.addPermanentWidget(self.performance_status)
    
    def setup_toolbar(self):
        """Setup application toolbar"""
        toolbar = self.addToolBar('Main')
        
        # Quick action buttons
        new_scan_action = QAction('New Scan', self)
        new_scan_action.triggered.connect(self.new_scan)
        toolbar.addAction(new_scan_action)
        
        toolbar.addSeparator()
        
        start_scan_action = QAction('Start', self)
        start_scan_action.triggered.connect(self.scan_control.start_scan)
        toolbar.addAction(start_scan_action)
        
        stop_scan_action = QAction('Stop', self)
        stop_scan_action.triggered.connect(self.scan_control.stop_scan)
        toolbar.addAction(stop_scan_action)
    
    def setup_connections(self):
        """Setup signal connections"""
        # Connect scan control signals
        self.scan_control.scan_started.connect(self.on_scan_started)
        self.scan_control.scan_stopped.connect(self.on_scan_stopped)
        self.scan_control.scan_paused.connect(self.on_scan_paused)
    
    def update_ui(self):
        """Update UI with latest data"""
        try:
            # Update service status
            self.update_service_status()
            
            # Update scan data if scanning
            if self.is_scanning:
                self.update_scan_data()
            
            # Update performance metrics
            self.update_performance_metrics()
            
        except Exception as e:
            print(f"UI update error: {e}")
    
    def update_service_status(self):
        """Update service connection status"""
        # Check scanning service
        if self.scanning_client.is_connected():
            self.scanning_status.setText("Scanning: Connected")
            self.scanning_status.setStyleSheet("color: #4CAF50;")
        else:
            self.scanning_status.setText("Scanning: Disconnected")
            self.scanning_status.setStyleSheet("color: #f44336;")
        
        # Check AI service
        if self.ai_client.is_connected():
            self.ai_status.setText("AI: Connected")
            self.ai_status.setStyleSheet("color: #4CAF50;")
        else:
            self.ai_status.setText("AI: Disconnected")
            self.ai_status.setStyleSheet("color: #f44336;")
    
    def update_scan_data(self):
        """Update scan data from services"""
        try:
            # Get scan status from scanning service
            response = self.scanning_client.send_request({
                'command': 'get_status'
            })
            
            if response and response.get('status') == 'success':
                stats = {
                    'frames': response.get('frame_count', 0),
                    'fps': 30.0,  # Would get from performance metrics
                    'points': 0,  # Would get from scan data
                    'vertices': 0  # Would get from mesh data
                }
                self.scan_control.update_statistics(stats)
            
        except Exception as e:
            print(f"Error updating scan data: {e}")
    
    def update_performance_metrics(self):
        """Update performance metrics display"""
        try:
            # Get performance metrics
            metrics = self.performance_monitor.get_metrics()
            fps = metrics.get('fps', 0.0)
            self.performance_status.setText(f"FPS: {fps:.1f}")
            
        except Exception as e:
            print(f"Error updating performance metrics: {e}")
    
    def on_scan_started(self):
        """Handle scan started event"""
        try:
            # Send start scan command to scanning service
            scan_params = {
                'scan_type': self.scan_control.scan_type_combo.currentText().lower().replace(' ', '_'),
                'quality': self.scan_control.quality_combo.currentText().lower()
            }
            
            response = self.scanning_client.send_request({
                'command': 'start_scan',
                'params': scan_params
            })
            
            if response and response.get('status') == 'success':
                self.is_scanning = True
                self.scan_session_id = response.get('scan_session_id')
                print(f"Scan started: {self.scan_session_id}")
            else:
                print("Failed to start scan")
                self.scan_control.stop_scan()  # Reset UI
                
        except Exception as e:
            print(f"Error starting scan: {e}")
            self.scan_control.stop_scan()
    
    def on_scan_stopped(self):
        """Handle scan stopped event"""
        try:
            # Send stop scan command to scanning service
            response = self.scanning_client.send_request({
                'command': 'stop_scan'
            })
            
            if response and response.get('status') == 'success':
                self.is_scanning = False
                stats = response.get('statistics', {})
                print(f"Scan completed: {stats}")
            
        except Exception as e:
            print(f"Error stopping scan: {e}")
    
    def on_scan_paused(self):
        """Handle scan paused event"""
        # Implementation for scan pause
        pass
    
    def new_scan(self):
        """Start a new scan session"""
        if self.is_scanning:
            self.scan_control.stop_scan()
        
        # Reset UI state
        self.scan_control.progress_bar.setValue(0)
        self.scan_control.status_label.setText("Ready to scan")
        self.ai_panel.results_text.clear()
    
    def toggle_fullscreen(self, checked: bool):
        """Toggle fullscreen mode"""
        if checked:
            self.showFullScreen()
        else:
            self.showNormal()
    
    def show_settings(self):
        """Show settings dialog"""
        # Implementation for settings dialog
        pass
    
    def show_about(self):
        """Show about dialog"""
        # Implementation for about dialog
        pass
    
    def run(self) -> int:
        """Run the Qt application"""
        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)
        
        self.show()
        return app.exec()
    
    def shutdown(self):
        """Shutdown the interface"""
        self.update_timer.stop()
        
        if self.is_scanning:
            self.on_scan_stopped()
        
        # Disconnect from services
        self.scanning_client.disconnect()
        self.ai_client.disconnect()
        
        self.close()

def main():
    """Run main interface as standalone application"""
    app = QApplication(sys.argv)
    
    interface = MainInterface()
    if interface.initialize():
        interface.show()
        return app.exec()
    else:
        print("Failed to initialize main interface")
        return 1

if __name__ == "__main__":
    exit(main())