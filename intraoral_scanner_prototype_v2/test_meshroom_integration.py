#!/usr/bin/env python3
"""
Test script for Meshroom integration with the dental scanning service.
This demonstrates the complete workflow: real-time SLAM + Meshroom reconstruction.
"""

import time
import json
import zmq
import numpy as np
from pathlib import Path

def test_meshroom_workflow():
    """Test the complete Meshroom integration workflow"""
    
    # Setup ZeroMQ client to communicate with scanning service
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://localhost:5555")
    
    print("🦷 Dental Scanner Meshroom Integration Test")
    print("=" * 50)
    
    try:
        # Step 1: Check service status
        print("1. Checking scanning service status...")
        socket.send_json({"command": "get_status"})
        response = socket.recv_json()
        print(f"   Service status: {response.get('status', 'unknown')}")
        
        # Step 2: Start Meshroom session
        print("\n2. Starting Meshroom reconstruction session...")
        socket.send_json({
            "command": "start_meshroom",
            "params": {
                "session_name": "test_dental_scan",
                "quality_preset": "dental_scan"
            }
        })
        response = socket.recv_json()
        print(f"   Meshroom session: {response.get('message', 'unknown')}")
        
        if response.get('status') != 'success':
            print("   ❌ Failed to start Meshroom session")
            return
        
        # Step 3: Start real-time scanning
        print("\n3. Starting real-time dental scanning...")
        socket.send_json({
            "command": "start_scan",
            "params": {
                "scan_id": "meshroom_test_001",
                "enable_meshroom": True
            }
        })
        response = socket.recv_json()
        print(f"   Scanning: {response.get('message', 'unknown')}")
        
        if response.get('status') != 'success':
            print("   ❌ Failed to start scanning")
            return
        
        # Step 4: Monitor scanning and Meshroom status
        print("\n4. Monitoring scanning progress...")
        for i in range(10):  # Monitor for 10 iterations
            time.sleep(2)
            
            # Get scanning status
            socket.send_json({"command": "get_status"})
            scan_status = socket.recv_json()
            
            # Get Meshroom status
            socket.send_json({"command": "meshroom_status"})
            meshroom_status = socket.recv_json()
            
            frames_processed = scan_status.get('frames_processed', 0)
            meshroom_info = meshroom_status.get('meshroom_status', {})
            frames_added = meshroom_info.get('frames_added', 0)
            
            print(f"   Progress: {frames_processed} frames processed, "
                  f"{frames_added} keyframes added to Meshroom")
            
            if i == 5:  # Halfway through
                print("   📷 Adding keyframes to Meshroom reconstruction...")
        
        # Step 5: Stop scanning
        print("\n5. Stopping real-time scanning...")
        socket.send_json({"command": "stop_scan"})
        response = socket.recv_json()
        print(f"   Scan stopped: {response.get('message', 'unknown')}")
        
        # Step 6: Finalize Meshroom reconstruction
        print("\n6. Starting Meshroom reconstruction pipeline...")
        socket.send_json({"command": "stop_meshroom"})
        response = socket.recv_json()
        
        if response.get('status') == 'success':
            mesh_path = response.get('mesh_path')
            print(f"   ✅ Meshroom reconstruction completed!")
            print(f"   📄 Mesh saved to: {mesh_path}")
        else:
            print(f"   ❌ Meshroom reconstruction failed: {response.get('message')}")
        
        print("\n🎉 Meshroom integration test completed!")
        print("\nWorkflow Summary:")
        print("- Real-time SLAM tracking for immediate feedback")
        print("- Keyframe detection and selection for Meshroom")
        print("- Professional photogrammetry reconstruction")
        print("- High-quality mesh output for dental applications")
        
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test error: {e}")
    finally:
        socket.close()
        context.term()

def create_sample_meshroom_config():
    """Create a sample configuration file for Meshroom integration"""
    
    config = {
        "meshroom_integration": {
            "enabled": True,
            "meshroom_path": "/opt/Meshroom-2023.3.0",
            "temp_directory": "/tmp/meshroom_dental",
            "quality_presets": {
                "dental_scan": {
                    "description": "Optimized for intraoral dental scanning",
                    "keyframe_threshold": 0.15,
                    "min_keyframe_distance": 10,
                    "max_images": 200,
                    "mesh_resolution": "high"
                },
                "real_time": {
                    "description": "Fast processing for real-time preview",
                    "keyframe_threshold": 0.25,
                    "min_keyframe_distance": 5,
                    "max_images": 50,
                    "mesh_resolution": "medium"
                },
                "high_quality": {
                    "description": "Maximum quality for final reconstruction",
                    "keyframe_threshold": 0.10,
                    "min_keyframe_distance": 15,
                    "max_images": 500,
                    "mesh_resolution": "ultra"
                }
            }
        },
        "scanning_service": {
            "enable_gpu_tsdf": True,
            "enable_slam": True,
            "enable_meshroom_integration": True,
            "meshroom_session_auto_start": False
        }
    }
    
    config_path = Path("meshroom_config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"📋 Sample Meshroom configuration saved to: {config_path}")
    return config_path

def print_integration_overview():
    """Print an overview of the Meshroom integration features"""
    
    print("\n🔧 Meshroom Integration Features:")
    print("-" * 40)
    print("✅ Real-time SLAM tracking with Visual SLAM")
    print("✅ Intelligent keyframe detection and selection")
    print("✅ Professional AliceVision photogrammetry pipeline")
    print("✅ Multiple quality presets (dental_scan, real_time, high_quality)")
    print("✅ GPU-accelerated TSDF fusion for real-time feedback")
    print("✅ Automatic image preprocessing and optimization")
    print("✅ High-quality mesh output with texture mapping")
    print("✅ Service-based architecture for scalability")
    
    print("\n📊 Technical Stack:")
    print("- Visual SLAM: Enhanced OpenCV-based tracking")
    print("- Meshroom: AliceVision professional 3D reconstruction")
    print("- GPU Processing: PyTorch CUDA acceleration")
    print("- Communication: ZeroMQ service messaging")
    print("- 3D Processing: Open3D point cloud operations")
    
    print("\n🎯 Dental Scanner Applications:")
    print("- Intraoral scanning with real-time preview")
    print("- Professional mesh reconstruction for CAD/CAM")
    print("- Quality validation and measurement")
    print("- Digital impression workflows")

if __name__ == "__main__":
    print_integration_overview()
    
    print("\n" + "=" * 60)
    response = input("Run Meshroom integration test? (y/n): ")
    
    if response.lower() in ['y', 'yes']:
        # Create sample configuration
        create_sample_meshroom_config()
        
        print("\nStarting test in 3 seconds...")
        time.sleep(3)
        
        # Run the test
        test_meshroom_workflow()
    else:
        print("Test skipped. Configuration file created for reference.")
        create_sample_meshroom_config()
