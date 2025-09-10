#!/usr/bin/env python3
"""
Simple Error Check - Test basic functionality without heavy dependencies
"""

import sys
import os
import numpy as np

# Add the project root to the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def test_basic_imports():
    """Test basic imports without heavy dependencies"""
    print("=== Basic Import Test ===")
    print()
    
    try:
        print("Testing NumPy...")
        import numpy as np
        print(f"✅ NumPy {np.__version__} imported successfully")
        
        # Test array operations
        arr = np.array([1, 2, 3])
        result = np.dot(arr, arr)
        print(f"✅ NumPy operations working (dot product: {result})")
        
    except Exception as e:
        print(f"❌ NumPy error: {e}")
        return False
    
    try:
        print("\nTesting configuration...")
        from config.system_config import get_config
        config = get_config()
        print("✅ Configuration system working")
        print(f"   Enhanced TSDF: {getattr(config.processing, 'use_enhanced_gpu_tsdf', 'Not set')}")
        
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False
    
    return True

def test_module_imports():
    """Test importing our custom modules"""
    print("=== Module Import Test ===")
    print()
    
    modules_to_test = [
        ("hardware.camera_manager_v2", "CameraManagerV2"),
        ("processing.slam_processor", "SLAMProcessor"),
        ("utils.performance_monitor", "PerformanceMonitor"),
        ("utils.shared_memory", "SharedMemoryManager"),
    ]
    
    success_count = 0
    
    for module_name, class_name in modules_to_test:
        try:
            module = __import__(module_name, fromlist=[class_name])
            cls = getattr(module, class_name)
            print(f"✅ {module_name}.{class_name} imported successfully")
            success_count += 1
        except Exception as e:
            print(f"❌ {module_name}.{class_name} import failed: {e}")
    
    print(f"\n📊 Module imports: {success_count}/{len(modules_to_test)} successful")
    return success_count == len(modules_to_test)

def test_enhanced_tsdf_basic():
    """Test enhanced TSDF basic functionality"""
    print("=== Enhanced TSDF Basic Test ===")
    print()
    
    try:
        print("Testing enhanced TSDF import...")
        from processing.gpu_tsdf_enhanced import EnhancedTSDFFusion, TSDFConfig
        print("✅ Enhanced TSDF classes imported successfully")
        
        print("Creating TSDF configuration...")
        config = TSDFConfig(
            volume_size=(0.05, 0.05, 0.05),  # Small 5cm cube for testing
            voxel_size=0.005,  # 5mm voxels for fast testing
            use_gpu=False,  # Force CPU for testing
        )
        print("✅ TSDF configuration created")
        
        print("Initializing TSDF...")
        tsdf = EnhancedTSDFFusion(config)
        
        if tsdf.initialize():
            print("✅ Enhanced TSDF initialized successfully")
            
            stats = tsdf.get_statistics()
            print(f"   Device: {stats['device']}")
            print(f"   Total voxels: {stats['total_voxels']:,}")
            print(f"   Memory usage: {stats['memory_usage_mb']:.1f} MB")
            
            tsdf.cleanup()
            print("✅ TSDF cleanup successful")
            return True
        else:
            print("❌ TSDF initialization failed")
            return False
            
    except Exception as e:
        print(f"❌ Enhanced TSDF test failed: {e}")
        return False

def test_file_structure():
    """Test that all required files exist"""
    print("=== File Structure Test ===")
    print()
    
    required_files = [
        "main_v2.py",
        "config/system_config.py",
        "processing/gpu_tsdf_enhanced.py",
        "processing/tsdf_fusion_v2.py",
        "processing/slam_processor.py",
        "services/scanning_service.py",
        "hardware/camera_manager_v2.py",
        "utils/performance_monitor.py",
        "utils/shared_memory.py",
        "requirements_enhanced.txt",
        "setup_enhanced.sh",
        "test_enhanced_tsdf.py"
    ]
    
    missing_files = []
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} (missing)")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\n⚠️  Missing {len(missing_files)} files:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        return False
    else:
        print(f"\n✅ All {len(required_files)} required files present")
        return True

def main():
    """Run all error checking tests"""
    print("🔍 IntraoralScan v2 Error Check")
    print("=" * 50)
    print()
    
    tests = [
        ("File Structure", test_file_structure),
        ("Basic Imports", test_basic_imports),
        ("Module Imports", test_module_imports),
        ("Enhanced TSDF Basic", test_enhanced_tsdf_basic),
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
    
    # Summary
    print(f"📋 Error Check Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! No critical errors found.")
        print()
        print("✅ Your enhanced v2 prototype structure is correct")
        print("✅ Basic functionality is working")
        print("✅ Ready for dependency installation")
        print()
        print("Next steps:")
        print("1. Run setup script: ./setup_enhanced.sh")
        print("2. Install dependencies if needed")
        print("3. Test with full dependencies: python test_enhanced_tsdf.py")
    else:
        print("⚠️  Some tests failed. Issues found:")
        if passed >= 2:
            print("• Basic structure is mostly correct")
            print("• Some dependencies may be missing")
            print("• Run './setup_enhanced.sh' to install dependencies")
        else:
            print("• Critical structural issues detected")
            print("• Check file paths and imports")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
