"""
AI Analysis Service - Professional AI processing based on DentalAlgoService.exe analysis
Implements neural network inference for dental segmentation and clinical analysis
"""

import time
import threading
import queue
import numpy as np
from typing import Optional, Dict, List, Any, Tuple
import zmq
import json
import onnxruntime as ort
import cv2
from pathlib import Path

from config.system_config import get_config
from utils.performance_monitor import PerformanceMonitor
from utils.shared_memory import SharedMemoryManager

class AIAnalysisService:
    """
    Professional AI analysis service implementing:
    - Tooth segmentation (matching fine_turn_*_v5.onnx models)
    - Clinical analysis and pathology detection
    - Real-time inference with confidence scoring
    """
    
    def __init__(self, service_port: int = 5556):
        self.config = get_config()
        self.service_port = service_port
        
        # Service state
        self.is_running = False
        self.models_loaded = False
        
        # ONNX Runtime sessions
        self.segmentation_session = None
        self.detection_session = None
        self.clinical_session = None
        
        # Model configurations (from analysis)
        self.model_configs = {
            'segmentation_v5': {
                'input_size': (240, 176),  # From analysis
                'depth_size': (120, 88),   # From analysis
                'confidence_threshold': 0.7,
                'num_classes': 32  # Individual teeth + background
            },
            'detection_v3': {
                'input_size': (240, 176),
                'confidence_threshold': 0.8,
                'num_classes': 16  # Various dental features
            }
        }
        
        # Threading
        self.processing_thread = None
        self.communication_thread = None
        
        # Data queues
        self.analysis_queue = queue.Queue(maxsize=10)
        self.result_queue = queue.Queue(maxsize=5)
        
        # ZeroMQ communication
        self.context = zmq.Context()
        self.socket = None
        
        # Performance monitoring
        self.performance_monitor = PerformanceMonitor()
        self.shared_memory = SharedMemoryManager()
        
        # Analysis statistics
        self.inference_count = 0
        self.inference_times = []
        
    def start_service(self) -> bool:
        """Start the AI analysis service"""
        try:
            # Load AI models
            if not self._load_models():
                print("ERROR: Failed to load AI models")
                return False
            
            # Setup communication
            self.socket = self.context.socket(zmq.REP)
            self.socket.bind(f"tcp://*:{self.service_port}")
            
            # Start service threads
            self.is_running = True
            self.processing_thread = threading.Thread(target=self._processing_loop)
            self.communication_thread = threading.Thread(target=self._communication_loop)
            
            self.processing_thread.start()
            self.communication_thread.start()
            
            print(f"AI Analysis service started on port {self.service_port}")
            return True
            
        except Exception as e:
            print(f"Error starting AI analysis service: {e}")
            return False
    
    def stop_service(self):
        """Stop the AI analysis service"""
        self.is_running = False
        
        # Stop threads
        if self.processing_thread:
            self.processing_thread.join()
        if self.communication_thread:
            self.communication_thread.join()
        
        # Cleanup ONNX sessions
        if self.segmentation_session:
            del self.segmentation_session
        if self.detection_session:
            del self.detection_session
        if self.clinical_session:
            del self.clinical_session
        
        # Cleanup communication
        if self.socket:
            self.socket.close()
        self.context.term()
        
        print("AI Analysis service stopped")
    
    def _load_models(self) -> bool:
        """Load ONNX models for AI analysis"""
        try:
            models_dir = Path(self.config.ai.models_directory)
            
            # Configure ONNX Runtime
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if self.config.ai.use_gpu else ['CPUExecutionProvider']
            
            # Load segmentation model (V5 generation from analysis)
            segmentation_model_path = models_dir / self.config.ai.segmentation_model
            if segmentation_model_path.exists():
                self.segmentation_session = ort.InferenceSession(
                    str(segmentation_model_path), providers=providers)
                print(f"✓ Loaded segmentation model: {segmentation_model_path}")
            else:
                print(f"⚠ Segmentation model not found: {segmentation_model_path}")
                # Create dummy model for testing
                self._create_dummy_segmentation_model()
            
            # Load detection model
            detection_model_path = models_dir / self.config.ai.detection_model
            if detection_model_path.exists():
                self.detection_session = ort.InferenceSession(
                    str(detection_model_path), providers=providers)
                print(f"✓ Loaded detection model: {detection_model_path}")
            else:
                print(f"⚠ Detection model not found: {detection_model_path}")
                self._create_dummy_detection_model()
            
            self.models_loaded = True
            return True
            
        except Exception as e:
            print(f"Error loading AI models: {e}")
            return False
    
    def _create_dummy_segmentation_model(self):
        """Create dummy segmentation for testing when real models not available"""
        print("Creating dummy segmentation model for testing...")
        self.segmentation_session = None  # Will use classical CV methods
    
    def _create_dummy_detection_model(self):
        """Create dummy detection for testing when real models not available"""
        print("Creating dummy detection model for testing...")
        self.detection_session = None  # Will use classical CV methods
    
    def _processing_loop(self):
        """Main AI processing loop"""
        while self.is_running:
            try:
                # Check for analysis requests
                if not self.analysis_queue.empty():
                    analysis_request = self.analysis_queue.get()
                    
                    start_time = time.time()
                    
                    # Perform AI analysis
                    result = self._perform_analysis(analysis_request)
                    
                    # Update performance metrics
                    inference_time = time.time() - start_time
                    self.inference_times.append(inference_time)
                    self.inference_count += 1
                    
                    # Store result
                    result['inference_time'] = inference_time
                    result['inference_id'] = self.inference_count
                    
                    # Send to result queue
                    if not self.result_queue.full():
                        self.result_queue.put(result)
                    
                    # Update shared memory
                    self.shared_memory.update_ai_results(result)
                    
                    # Performance monitoring
                    self.performance_monitor.update_metrics({
                        'ai_fps': 1.0 / inference_time if inference_time > 0 else 0,
                        'inference_time': inference_time,
                        'queue_size': self.analysis_queue.qsize()
                    })
                
                else:
                    time.sleep(0.001)  # 1ms sleep when no work
                    
            except Exception as e:
                print(f"Error in AI processing loop: {e}")
                time.sleep(0.01)
    
    def _communication_loop(self):
        """Handle communication with other services"""
        while self.is_running:
            try:
                # Check for incoming messages
                if self.socket.poll(timeout=10):  # 10ms timeout
                    message = self.socket.recv_json()
                    response = self._handle_message(message)
                    self.socket.send_json(response)
                
            except Exception as e:
                print(f"Error in AI communication loop: {e}")
                time.sleep(0.01)
    
    def _handle_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Handle incoming service messages"""
        command = message.get('command', '')
        
        if command == 'analyze_image':
            return self._analyze_image(message.get('params', {}))
        elif command == 'analyze_mesh':
            return self._analyze_mesh(message.get('params', {}))
        elif command == 'get_status':
            return self._get_status()
        elif command == 'get_performance':
            return self._get_performance_metrics()
        else:
            return {'status': 'error', 'message': f'Unknown command: {command}'}
    
    def _analyze_image(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze 2D image for dental features"""
        try:
            # Get image data from shared memory or params
            image_id = params.get('image_id')
            image_data = params.get('image_data')
            
            if image_data is None and image_id:
                image_data = self.shared_memory.get_image_data(image_id)
            
            if image_data is None:
                return {'status': 'error', 'message': 'No image data provided'}
            
            # Convert to numpy array if needed
            if isinstance(image_data, list):
                image_data = np.array(image_data, dtype=np.uint8)
            
            # Queue analysis request
            analysis_request = {
                'type': 'image',
                'data': image_data,
                'params': params
            }
            
            if not self.analysis_queue.full():
                self.analysis_queue.put(analysis_request)
                return {'status': 'success', 'message': 'Analysis queued'}
            else:
                return {'status': 'error', 'message': 'Analysis queue full'}
                
        except Exception as e:
            return {'status': 'error', 'message': f'Failed to analyze image: {e}'}
    
    def _analyze_mesh(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze 3D mesh for dental features"""
        try:
            # Get mesh data from shared memory or params
            mesh_id = params.get('mesh_id')
            mesh_data = params.get('mesh_data')
            
            if mesh_data is None and mesh_id:
                mesh_data = self.shared_memory.get_mesh_data(mesh_id)
            
            if mesh_data is None:
                return {'status': 'error', 'message': 'No mesh data provided'}
            
            # Queue analysis request
            analysis_request = {
                'type': 'mesh',
                'data': mesh_data,
                'params': params
            }
            
            if not self.analysis_queue.full():
                self.analysis_queue.put(analysis_request)
                return {'status': 'success', 'message': 'Analysis queued'}
            else:
                return {'status': 'error', 'message': 'Analysis queue full'}
                
        except Exception as e:
            return {'status': 'error', 'message': f'Failed to analyze mesh: {e}'}
    
    def _perform_analysis(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Perform AI analysis on data"""
        analysis_type = request['type']
        data = request['data']
        params = request.get('params', {})
        
        if analysis_type == 'image':
            return self._analyze_image_data(data, params)
        elif analysis_type == 'mesh':
            return self._analyze_mesh_data(data, params)
        else:
            return {'status': 'error', 'message': f'Unknown analysis type: {analysis_type}'}
    
    def _analyze_image_data(self, image: np.ndarray, params: Dict[str, Any]) -> Dict[str, Any]:
        """Perform AI analysis on image data"""
        try:
            results = {
                'status': 'success',
                'segmentation': None,
                'detection': None,
                'clinical_analysis': None
            }
            
            # Tooth segmentation
            if self.segmentation_session:
                segmentation_result = self._run_segmentation_model(image)
                results['segmentation'] = segmentation_result
            else:
                # Fallback to classical computer vision
                segmentation_result = self._classical_segmentation(image)
                results['segmentation'] = segmentation_result
            
            # Feature detection
            if self.detection_session:
                detection_result = self._run_detection_model(image)
                results['detection'] = detection_result
            else:
                # Fallback to classical detection
                detection_result = self._classical_detection(image)
                results['detection'] = detection_result
            
            # Clinical analysis
            if self.config.ai.clinical_analysis_enabled:
                clinical_result = self._perform_clinical_analysis(image, segmentation_result)
                results['clinical_analysis'] = clinical_result
            
            return results
            
        except Exception as e:
            return {'status': 'error', 'message': f'Image analysis failed: {e}'}
    
    def _run_segmentation_model(self, image: np.ndarray) -> Dict[str, Any]:
        """Run neural network segmentation model"""
        try:
            # Preprocess image (matching analysis findings)
            input_size = self.model_configs['segmentation_v5']['input_size']
            processed_image = self._preprocess_image(image, input_size)
            
            # Run inference
            input_name = self.segmentation_session.get_inputs()[0].name
            output = self.segmentation_session.run(None, {input_name: processed_image})
            
            # Post-process results
            segmentation_mask = self._postprocess_segmentation(output[0])
            
            # Extract individual teeth
            teeth = self._extract_individual_teeth(segmentation_mask)
            
            return {
                'mask': segmentation_mask.tolist(),
                'teeth_count': len(teeth),
                'teeth': teeth,
                'confidence': np.mean([tooth['confidence'] for tooth in teeth]) if teeth else 0.0
            }
            
        except Exception as e:
            print(f"Segmentation model error: {e}")
            return self._classical_segmentation(image)
    
    def _classical_segmentation(self, image: np.ndarray) -> Dict[str, Any]:
        """Fallback classical computer vision segmentation"""
        try:
            # Convert to different color spaces
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            
            # Segment teeth (bright, low saturation regions)
            brightness = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            _, bright_mask = cv2.threshold(brightness, 120, 255, cv2.THRESH_BINARY)
            _, low_sat_mask = cv2.threshold(hsv[:,:,1], 80, 255, cv2.THRESH_BINARY_INV)
            
            teeth_mask = bright_mask & low_sat_mask
            
            # Clean up mask
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            teeth_mask = cv2.morphologyEx(teeth_mask, cv2.MORPH_OPEN, kernel)
            teeth_mask = cv2.morphologyEx(teeth_mask, cv2.MORPH_CLOSE, kernel)
            
            # Find individual teeth
            contours, _ = cv2.findContours(teeth_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            teeth = []
            for i, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                if 100 < area < 5000:  # Reasonable tooth size
                    x, y, w, h = cv2.boundingRect(contour)
                    center = (x + w//2, y + h//2)
                    
                    teeth.append({
                        'id': i + 1,
                        'center': center,
                        'area': area,
                        'bounding_box': [x, y, w, h],
                        'confidence': 0.8  # Classical method confidence
                    })
            
            return {
                'mask': teeth_mask.tolist(),
                'teeth_count': len(teeth),
                'teeth': teeth,
                'confidence': 0.8,
                'method': 'classical_cv'
            }
            
        except Exception as e:
            print(f"Classical segmentation error: {e}")
            return {'mask': [], 'teeth_count': 0, 'teeth': [], 'confidence': 0.0}
    
    def _classical_detection(self, image: np.ndarray) -> Dict[str, Any]:
        """Fallback classical computer vision detection"""
        try:
            # Simple feature detection using edge detection and contours
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            features = []
            for i, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                if area > 50:  # Minimum feature size
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    features.append({
                        'id': i,
                        'type': 'edge_feature',
                        'bounding_box': [x, y, w, h],
                        'area': area,
                        'confidence': 0.6
                    })
            
            return {
                'features': features,
                'feature_count': len(features),
                'method': 'classical_cv'
            }
            
        except Exception as e:
            print(f"Classical detection error: {e}")
            return {'features': [], 'feature_count': 0}
    
    def _perform_clinical_analysis(self, image: np.ndarray, segmentation: Dict[str, Any]) -> Dict[str, Any]:
        """Perform clinical analysis based on segmentation results"""
        try:
            analysis = {
                'tooth_count': segmentation.get('teeth_count', 0),
                'quality_score': segmentation.get('confidence', 0.0),
                'findings': [],
                'recommendations': []
            }
            
            # Basic quality assessment
            if analysis['tooth_count'] < 20:
                analysis['findings'].append('Incomplete scan - missing teeth detected')
                analysis['recommendations'].append('Continue scanning to capture all teeth')
            
            if analysis['quality_score'] < 0.7:
                analysis['findings'].append('Low segmentation confidence')
                analysis['recommendations'].append('Improve lighting or camera positioning')
            
            # Check for potential issues
            teeth = segmentation.get('teeth', [])
            if teeth:
                areas = [tooth['area'] for tooth in teeth]
                if areas:
                    area_std = np.std(areas)
                    area_mean = np.mean(areas)
                    
                    for tooth in teeth:
                        if tooth['area'] < area_mean - 2 * area_std:
                            analysis['findings'].append(f"Small tooth detected at position {tooth['center']}")
                        elif tooth['area'] > area_mean + 2 * area_std:
                            analysis['findings'].append(f"Large tooth detected at position {tooth['center']}")
            
            return analysis
            
        except Exception as e:
            print(f"Clinical analysis error: {e}")
            return {'tooth_count': 0, 'quality_score': 0.0, 'findings': [], 'recommendations': []}
    
    def _preprocess_image(self, image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """Preprocess image for neural network input"""
        # Resize to target size
        resized = cv2.resize(image, target_size)
        
        # Normalize to [0, 1]
        normalized = resized.astype(np.float32) / 255.0
        
        # Add batch dimension and transpose to NCHW format
        if len(normalized.shape) == 3:
            normalized = np.transpose(normalized, (2, 0, 1))  # HWC to CHW
        
        normalized = np.expand_dims(normalized, axis=0)  # Add batch dimension
        
        return normalized
    
    def _postprocess_segmentation(self, output: np.ndarray) -> np.ndarray:
        """Post-process segmentation model output"""
        # Apply softmax if needed
        if output.shape[1] > 1:  # Multi-class
            output = np.exp(output) / np.sum(np.exp(output), axis=1, keepdims=True)
            segmentation = np.argmax(output, axis=1)
        else:  # Binary
            segmentation = (output > 0.5).astype(np.uint8)
        
        # Remove batch dimension
        segmentation = segmentation.squeeze(0)
        
        return segmentation
    
    def _extract_individual_teeth(self, segmentation_mask: np.ndarray) -> List[Dict[str, Any]]:
        """Extract individual teeth from segmentation mask"""
        teeth = []
        
        # Find unique tooth IDs (excluding background)
        unique_ids = np.unique(segmentation_mask)
        unique_ids = unique_ids[unique_ids > 0]  # Remove background
        
        for tooth_id in unique_ids:
            tooth_mask = (segmentation_mask == tooth_id).astype(np.uint8)
            
            # Find contours
            contours, _ = cv2.findContours(tooth_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Get largest contour (main tooth region)
                largest_contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest_contour)
                
                if area > 50:  # Minimum tooth size
                    # Calculate properties
                    moments = cv2.moments(largest_contour)
                    if moments['m00'] > 0:
                        center_x = int(moments['m10'] / moments['m00'])
                        center_y = int(moments['m01'] / moments['m00'])
                        
                        x, y, w, h = cv2.boundingRect(largest_contour)
                        
                        teeth.append({
                            'id': int(tooth_id),
                            'center': (center_x, center_y),
                            'area': area,
                            'bounding_box': [x, y, w, h],
                            'confidence': 0.9  # High confidence for neural network
                        })
        
        return teeth
    
    def _get_status(self) -> Dict[str, Any]:
        """Get current service status"""
        return {
            'status': 'success',
            'service_running': self.is_running,
            'models_loaded': self.models_loaded,
            'inference_count': self.inference_count,
            'queue_size': self.analysis_queue.qsize(),
            'segmentation_model': self.segmentation_session is not None,
            'detection_model': self.detection_session is not None
        }
    
    def _get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        metrics = self.performance_monitor.get_metrics()
        
        return {
            'status': 'success',
            'metrics': metrics,
            'inference_times': self.inference_times[-100:],  # Last 100 inferences
            'average_inference_time': np.mean(self.inference_times) if self.inference_times else 0
        }

def main():
    """Run AI analysis service as standalone process"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Dental AI Analysis Service')
    parser.add_argument('--port', type=int, default=5556, help='Service port')
    parser.add_argument('--models-dir', type=str, help='Models directory path')
    
    args = parser.parse_args()
    
    # Initialize service
    service = AIAnalysisService(service_port=args.port)
    
    try:
        if service.start_service():
            print("AI Analysis service running. Press Ctrl+C to stop.")
            while True:
                time.sleep(1)
        else:
            print("Failed to start AI analysis service")
    except KeyboardInterrupt:
        print("\nShutting down AI analysis service...")
    finally:
        service.stop_service()

if __name__ == "__main__":
    main()