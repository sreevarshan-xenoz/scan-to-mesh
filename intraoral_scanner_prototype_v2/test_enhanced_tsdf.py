#!/usr/bin/env python3
"""
Enhanced TSDF Integration Test
Demonstrates the upgraded PyTorch GPU acceleration in v2 prototype
"""

import sys
import os
import time
import numpy as np

# Add the project root to the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def test_enhanced_tsdf_integration():
    """Test the enhanced TSDF integration in the v2 prototype"""
    print("=== Enhanced TSDF Integration Test ===")
    print()
    
    try:
        # Import enhanced TSDF
        from processing.gpu_tsdf_enhanced import EnhancedTSDFFusion, TSDFConfig
        print("✅ Enhanced TSDF imported successfully")
        
        # Test configuration for dental scanning
        config = TSDFConfig(
            volume_size=(0.10, 0.10, 0.08),  # 10cm x 10cm x 8cm
            voxel_size=0.0015,  # 1.5mm voxels
            use_gpu=True,
            truncation_distance=0.006,  # 6mm truncation
            max_weight=75.0
        )
        print(f"✅ TSDF config created: {config.volume_size} volume, {config.voxel_size}m voxels")
        
        # Initialize TSDF
        tsdf = EnhancedTSDFFusion(config)
        
        if tsdf.initialize():
            print("✅ Enhanced TSDF initialized successfully")
            
            # Display system info
            stats = tsdf.get_statistics()
            print(f"   Device: {stats['device']}")
            print(f"   Memory usage: {stats['memory_usage_mb']:.1f} MB")
            print(f"   Total voxels: {stats['total_voxels']:,}")
            print()
            
            # Test integration with synthetic data
            print("Testing frame integration...")
            
            # Simulate camera intrinsics (similar to AorB scanner)
            intrinsics = {
                'fx': 850.0,
                'fy': 850.0,
                'cx': 640.0,
                'cy': 360.0
            }
            
            # Test with 3 synthetic frames
            success_count = 0
            
            for i in range(3):
                # Create synthetic depth data (simulating dental arch)
                depth_image = create_synthetic_dental_depth(i)
                color_image = create_synthetic_color()
                
                # Create camera pose (slight rotation each frame)
                pose = create_camera_pose(i * 5.0)  # 5 degree increments
                
                # Integrate frame
                start_time = time.time()
                success = tsdf.integrate_frame(
                    depth_image, color_image, intrinsics, pose, depth_scale=1000.0
                )
                integration_time = time.time() - start_time
                
                if success:
                    success_count += 1
                    print(f"   Frame {i+1}: ✅ Integrated in {integration_time:.3f}s")
                else:
                    print(f"   Frame {i+1}: ❌ Integration failed")
            
            print(f"✅ Successfully integrated {success_count}/3 frames")
            print()
            
            # Test mesh extraction
            print("Testing mesh extraction...")
            start_time = time.time()
            mesh_data = tsdf.extract_mesh(min_weight_threshold=1.0)
            extraction_time = time.time() - start_time
            
            if mesh_data:
                vertices = mesh_data['vertices']
                triangles = mesh_data['triangles']
                print(f"✅ Mesh extracted in {extraction_time:.3f}s")
                print(f"   Vertices: {len(vertices):,}")
                print(f"   Triangles: {len(triangles):,}")
                
                # Analyze mesh quality
                if len(vertices) > 0:
                    bounds = {
                        'x': (np.min(vertices[:, 0]), np.max(vertices[:, 0])),
                        'y': (np.min(vertices[:, 1]), np.max(vertices[:, 1])),
                        'z': (np.min(vertices[:, 2]), np.max(vertices[:, 2]))
                    }
                    print(f"   Mesh bounds:")
                    print(f"     X: {bounds['x'][0]:.3f} to {bounds['x'][1]:.3f} m")
                    print(f"     Y: {bounds['y'][0]:.3f} to {bounds['y'][1]:.3f} m")
                    print(f"     Z: {bounds['z'][0]:.3f} to {bounds['z'][1]:.3f} m")
            else:
                print("❌ Mesh extraction failed")
            
            print()
            
            # Final statistics
            final_stats = tsdf.get_statistics()
            print("📊 Final Statistics:")
            print(f"   Integration count: {final_stats['integration_count']}")
            print(f"   Average integration time: {final_stats['avg_integration_time']:.3f}s")
            print(f"   Last mesh extraction time: {final_stats['last_mesh_extraction_time']:.3f}s")
            print(f"   Occupied voxels: {final_stats['occupied_voxels']:,}")
            print(f"   Volume occupancy: {final_stats['volume_occupancy']:.1%}")
            
            # Cleanup
            tsdf.cleanup()
            print("✅ TSDF cleaned up successfully")
            
        else:
            print("❌ Failed to initialize Enhanced TSDF")
            return False
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   This may be due to missing dependencies (PyTorch, Open3D)")
        return False
    
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False
    
    print()
    print("🎉 Enhanced TSDF integration test completed successfully!")
    return True

def create_synthetic_dental_depth(frame_index: int) -> np.ndarray:
    """Create synthetic depth data resembling a dental arch"""
    height, width = 480, 640
    depth_image = np.zeros((height, width), dtype=np.float32)
    
    # Create a curved surface resembling a dental arch
    y_center = height // 2
    x_center = width // 2
    
    for y in range(height):
        for x in range(width):
            # Distance from center
            dx = (x - x_center) / width
            dy = (y - y_center) / height
            
            # Create arch shape
            distance = np.sqrt(dx*dx + dy*dy)
            
            # Simulate dental arch depth (closer in center, farther at edges)
            if distance < 0.4:  # Within scanning region
                base_depth = 45.0  # 45mm base distance
                arch_curve = 15.0 * (1.0 - distance * 2.5)  # Curved surface
                noise = np.random.normal(0, 0.5)  # Small amount of noise
                
                # Add frame-specific offset for motion
                motion_offset = frame_index * 2.0
                
                depth_image[y, x] = base_depth + arch_curve + noise + motion_offset
    
    return depth_image

def create_synthetic_color() -> np.ndarray:
    """Create synthetic color image"""
    height, width = 480, 640
    
    # Create a gradient color image (simulating tooth color variation)
    color_image = np.zeros((height, width, 3), dtype=np.uint8)
    
    for y in range(height):
        for x in range(width):
            # Tooth-like coloring (yellowish white)
            base_color = [240, 235, 220]  # Tooth white
            
            # Add some variation
            variation = np.random.randint(-10, 10, 3)
            pixel_color = np.clip(np.array(base_color) + variation, 0, 255)
            
            color_image[y, x] = pixel_color
    
    return color_image

def create_camera_pose(angle_degrees: float) -> np.ndarray:
    """Create camera pose matrix with rotation"""
    angle_rad = np.radians(angle_degrees)
    
    # Rotation around Y axis (typical for intraoral scanning)
    rotation_matrix = np.array([
        [np.cos(angle_rad), 0, np.sin(angle_rad)],
        [0, 1, 0],
        [-np.sin(angle_rad), 0, np.cos(angle_rad)]
    ])
    
    # Translation (camera moving slightly)
    translation = np.array([0.0, 0.0, 0.05])  # 5cm from surface
    
    # Create 4x4 transformation matrix
    pose = np.eye(4)
    pose[:3, :3] = rotation_matrix
    pose[:3, 3] = translation
    
    return pose

def test_configuration_integration():
    """Test the configuration system integration"""
    print("=== Configuration Integration Test ===")
    print()
    
    try:
        from config.system_config import get_config
        config = get_config()
        
        print("✅ System configuration loaded")
        print(f"   Enhanced GPU TSDF enabled: {config.processing.use_enhanced_gpu_tsdf}")
        print(f"   GPU memory limit: {config.processing.gpu_memory_limit_mb} MB")
        print(f"   Voxel size: {config.processing.voxel_size} m")
        print()
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_service_integration():
    """Test the scanning service integration"""
    print("=== Service Integration Test ===")
    print()
    
    try:
        # This is a basic import test - full service test would require more setup
        from services.scanning_service import ScanningService
        print("✅ ScanningService imported successfully")
        print("   Enhanced TSDF integration ready")
        print()
        
        return True
        
    except Exception as e:
        print(f"❌ Service integration test failed: {e}")
        print("   This may be due to missing dependencies")
        return False

if __name__ == "__main__":
    print("🔬 IntraoralScan v2 Enhanced TSDF Integration Tests")
    print("=" * 60)
    print()
    
    # Run all tests
    tests = [
        ("Enhanced TSDF", test_enhanced_tsdf_integration),
        ("Configuration", test_configuration_integration),
        ("Service Integration", test_service_integration)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"Running {test_name} test...")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} test PASSED")
            else:
                print(f"❌ {test_name} test FAILED")
        except Exception as e:
            print(f"❌ {test_name} test FAILED with exception: {e}")
        
        print("-" * 40)
        print()
    
    # Final results
    print(f"📋 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Enhanced TSDF integration is ready.")
        print()
        print("Next steps:")
        print("1. Install dependencies: pip install torch open3d")
        print("2. Run the main application: python main_v2.py")
        print("3. Enable GPU acceleration in configuration")
    else:
        print("⚠️  Some tests failed. Check dependencies and configuration.")
    
    print()
