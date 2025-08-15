#!/usr/bin/env python3
"""
Configuration file analyzer for IntraoralScan
"""
import json
import os
from pathlib import Path

def analyze_system_architecture():
    """Analyze system architecture from configuration files"""
    
    results = {
        'service_startup_order': [],
        'communication_channels': {},
        'supported_devices': [],
        'network_endpoints': {},
        'feature_flags': {},
        'system_info': {}
    }
    
    # Analyze launcher configuration for service startup order
    launcher_cfg_path = "IntraoralScan/Bin/config/Launcher/LauncherCfg.json"
    if os.path.exists(launcher_cfg_path):
        with open(launcher_cfg_path, 'r') as f:
            launcher_cfg = json.load(f)
        
        # Extract default startup modules
        for module_list in launcher_cfg.get('startmodulelists', []):
            if module_list['name'] == 'default':
                for module in module_list.get('startmodules', []):
                    service_info = {
                        'name': module['name'],
                        'executable': module['path'],
                        'start_level': module['startLevel'],
                        'exit_level': module['exitLevel'],
                        'must_start': module.get('mustStart', False),
                        'is_gui': module.get('isGUI', False)
                    }
                    results['service_startup_order'].append(service_info)
        
        # Sort by start level
        results['service_startup_order'].sort(key=lambda x: x['start_level'])
    
    # Analyze main system configuration
    main_cfg_path = "IntraoralScan/Bin/config/IntraoralScan/SystemCfg.json"
    if os.path.exists(main_cfg_path):
        with open(main_cfg_path, 'r') as f:
            main_cfg = json.load(f)
        
        # Extract communication tunnel info
        tunnel_info = main_cfg.get('Tunnel', {})
        results['communication_channels']['main_tunnel'] = {
            'channel': tunnel_info.get('Channel'),
            'host': tunnel_info.get('HostAddress'),
            'port': tunnel_info.get('Port'),
            'service_name': tunnel_info.get('ServiceName'),
            'shared_memory': tunnel_info.get('SharedMemoryName'),
            'memory_size': tunnel_info.get('SharedMemorySize')
        }
        
        # Extract server info
        server_info = main_cfg.get('Server', {})
        results['network_endpoints']['http_sockets'] = server_info.get('httpSocket', [])
        
        # Extract system info
        system_info = main_cfg.get('System', {})
        results['system_info'] = {
            'model_code': system_info.get('modelCode'),
            'version': system_info.get('version'),
            'visibility': system_info.get('visibility')
        }
        
        # Extract feature flags
        default_settings = main_cfg.get('Default', {})
        results['feature_flags'] = {
            'full_offline_mode': default_settings.get('enableFullOfflineMode', False),
            'ir_enabled': default_settings.get('IREnabled', True),
            'tough_mode': default_settings.get('ToughMode', False),
            'expo_mode': default_settings.get('expoMode', False),
            'encrypt_log': default_settings.get('EncryptLog', False)
        }
    
    # Analyze device configuration
    device_cfg_path = "IntraoralScan/Bin/config/IntraoralScan/DeviceCfg.json"
    if os.path.exists(device_cfg_path):
        with open(device_cfg_path, 'r') as f:
            device_cfg = json.load(f)
        
        for device in device_cfg.get('supportDevices', []):
            device_info = {
                'name': device.get('name'),
                'soft_model': device.get('softModel'),
                'disabled': device.get('disabled', False),
                'is_native': device.get('isNative', False),
                'disabled_functions': device.get('disableAuthFunctions', [])
            }
            results['supported_devices'].append(device_info)
    
    # Analyze network configuration
    network_cfg_path = "IntraoralScan/Bin/config/DentalNetwork/QuickCloudCfg.json"
    if os.path.exists(network_cfg_path):
        with open(network_cfg_path, 'r') as f:
            network_cfg = json.load(f)
        
        # Extract cloud configuration
        results['network_endpoints']['cloud_config'] = {
            'model_code': network_cfg.get('software', {}).get('modelCode'),
            'version': network_cfg.get('software', {}).get('version'),
            'mqtt_settings': network_cfg.get('mqtt', {}),
            'auto_upgrade': network_cfg.get('autoUpgrade', {})
        }
    
    return results

def generate_architecture_summary():
    """Generate a comprehensive architecture summary"""
    results = analyze_system_architecture()
    
    print("=== INTRAORAL SCAN ARCHITECTURE ANALYSIS ===\n")
    
    print("=== SERVICE STARTUP ORDER ===")
    for service in results['service_startup_order']:
        gui_marker = " [GUI]" if service['is_gui'] else ""
        must_start = " [CRITICAL]" if service['must_start'] else ""
        print(f"Level {service['start_level']:3d}: {service['name']:<20} -> {service['executable']}{gui_marker}{must_start}")
    
    print(f"\n=== COMMUNICATION ARCHITECTURE ===")
    tunnel = results['communication_channels'].get('main_tunnel', {})
    print(f"Main IPC Channel: {tunnel.get('channel')} on {tunnel.get('host')}:{tunnel.get('port')}")
    print(f"Service Hub: {tunnel.get('service_name')}")
    print(f"Shared Memory: {tunnel.get('shared_memory')} ({tunnel.get('memory_size')})")
    print(f"HTTP Sockets: {', '.join(results['network_endpoints'].get('http_sockets', []))}")
    
    print(f"\n=== SYSTEM INFORMATION ===")
    sys_info = results['system_info']
    print(f"Model Code: {sys_info.get('model_code')}")
    print(f"Version: {sys_info.get('version')}")
    print(f"Display Mode: {sys_info.get('visibility')}")
    
    print(f"\n=== FEATURE FLAGS ===")
    flags = results['feature_flags']
    for flag, value in flags.items():
        status = "ENABLED" if value else "DISABLED"
        print(f"{flag.replace('_', ' ').title()}: {status}")
    
    print(f"\n=== SUPPORTED DEVICES ===")
    active_devices = [d for d in results['supported_devices'] if not d['disabled']]
    print(f"Active Devices: {len(active_devices)} of {len(results['supported_devices'])} total")
    for device in active_devices:
        native_marker = " [NATIVE]" if device['is_native'] else ""
        print(f"  {device['name']}: {device['soft_model']}{native_marker}")
        if device['disabled_functions']:
            print(f"    Disabled Functions: {', '.join(device['disabled_functions'][:3])}{'...' if len(device['disabled_functions']) > 3 else ''}")
    
    return results

if __name__ == "__main__":
    results = generate_architecture_summary()
    
    # Save detailed results
    with open('analysis_output/architecture_analysis.json', 'w') as f:
        json.dump(results, f, indent=2)