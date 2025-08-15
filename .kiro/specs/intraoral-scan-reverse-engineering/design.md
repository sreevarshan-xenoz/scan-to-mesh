# Design Document

## Overview

This design outlines a systematic approach to reverse engineer the IntraoralScan 3.5.4.6 dental scanning application using static analysis, configuration examination, and safe binary inspection techniques. The approach focuses on understanding the architecture through dependency mapping, IPC tracing, readable asset analysis, and domain-specific pipeline reconstruction.

## Architecture

### Analysis Framework

The reverse engineering process follows a layered approach:

1. **Static Analysis Layer**: Dependency mapping, file classification, and metadata extraction
2. **Communication Layer**: IPC mechanism identification and network endpoint discovery  
3. **Asset Analysis Layer**: Readable component examination (QML, Python, configs, DB)
4. **Binary Inspection Layer**: Safe analysis of compiled components without decompilation
5. **Documentation Layer**: Architecture mapping and pipeline reconstruction

### High-Value Targets (Priority Order)

1. **IntraoralScan.exe** - Main orchestrator, reveals overall architecture
2. **DentalAlgoService.exe** - Core AI/3D processing engine
3. **AIModels/ folder** - Neural network models and processing pipelines
4. **config/ directory** - Service configuration and feature flags
5. **DB/ directory** - Database schemas revealing business logic
6. **DentalScanAppLogic.exe** - Real-time scanning implementation
7. **Python runtime** - Scripting logic and algorithm integration

### Tooling Matrix

| Task | Primary Tool | Alternate Tool | Notes |
|------|-------------|----------------|-------|
| DLL dependency mapping | Dependencies | PE-sieve | Works on x64 & x86 |
| Network sniffing | Wireshark | Fiddler | Filter by process name |
| QML extraction | rcc tool | qml-extract | Works on Qt5 |
| Python decompilation | uncompyle6 | decompyle3 | Python 3.8 compatible |
| Binary analysis | Ghidra | IDA Free | Export analysis focus |
| AI model inspection | Netron | ONNX Runtime | Multiple format support |
| Database analysis | DB Browser for SQLite | SQLite CLI | Schema extraction |
| Process monitoring | Process Monitor | API Monitor | IPC discovery |
| String extraction | FLOSS | strings | Unicode support |

### Component Classification System

Based on the extracted application structure, components are classified into five categories:

- **UI Components**: Qt/QML-based user interface elements
- **Logic Components**: Core business logic and workflow management
- **Algorithm Components**: 3D processing, AI/ML, and dental-specific algorithms
- **Network Components**: Communication services and cloud integration
- **Order Components**: Workflow management and data export functionality

## Components and Interfaces

### 1. Process Topology Analyzer

**Purpose**: Map all executables, their roles, dependencies, and startup relationships

**Key Functions**:
- Executable classification and role identification
- DLL dependency tree construction
- Process startup order determination
- Service watchdog relationship mapping

**Tools Integration**:
- Dependencies tool for DLL mapping
- Detect It Easy for language/framework detection
- Process Monitor for runtime analysis

### 2. Communication Tracer

**Purpose**: Identify and document inter-process communication mechanisms

**Key Functions**:
- Named pipe discovery and mapping
- Network endpoint identification
- IPC protocol analysis
- Service port documentation

**Detection Methods**:
- Process Monitor filtering for named pipes
- Network traffic capture via Wireshark/Fiddler
- String analysis for pipe names and endpoints
- Configuration file parsing for connection details

### 3. Asset Analyzer

**Purpose**: Extract and analyze readable application components

**Components**:

#### QML/Qt Resource Analyzer
- Extract .qml files from resources
- Analyze UI structure and signal/slot connections
- Map user interface flows and state machines

#### Python Runtime Analyzer  
- Decompile .pyc files using uncompyle6
- Analyze .pyd extensions for C/C++ integration
- Map Python module usage across services

#### Configuration Parser
- Parse JSON/INI/XML/YAML configuration files
- Extract feature flags, service ports, device IDs
- Document configuration hierarchies and dependencies

#### Database Schema Analyzer
- Analyze SQLite databases for schema structure
- Map table relationships and data flows
- Extract business logic from database constraints

### 4. Binary & Algorithm Inspector

**Purpose**: Safely analyze compiled components and AI models without decompilation

**Analysis Targets**:

#### Algorithm DLL Analysis
- Export function identification and signature analysis
- String analysis for algorithm hints and parameter names
- Library dependency mapping (OpenCV, CUDA, custom dental libraries)
- GPU processing capability assessment

#### AI Model Analysis
- Model format identification (.onnx, .pb, .engine, .trt)
- Input/output tensor shape and data type documentation
- Pre/post-processing pipeline inference from surrounding code
- Model versioning and update mechanism analysis

#### Hardware Interface Analysis
- Driver DLL identification and device communication protocols
- Camera and scanner hardware capability mapping
- Firmware update mechanism documentation
- Device calibration and configuration analysis

### 5. Pipeline Reconstructor

**Purpose**: Reconstruct the dental scanning processing pipeline

**Pipeline Stages**:

1. **Acquisition Stage**: Camera stream processing and depth generation
2. **Registration Stage**: Incremental alignment and pose estimation  
3. **Fusion Stage**: Mesh generation and surface reconstruction
4. **AI Analysis Stage**: Tooth segmentation and clinical analysis
5. **Visualization Stage**: 3D rendering and user interaction
6. **Export Stage**: Data formatting and order management

## Data Models

### Process Dependency Model
```
ProcessNode {
  executable: string
  role: ProcessRole (UI|Logic|Algorithm|Network|Order)
  dependencies: DLL[]
  ipc_endpoints: IPCEndpoint[]
  network_endpoints: NetworkEndpoint[]
  startup_order: number
  data_handled: DataType[]
  confidence_score: number (0.0-1.0)
}
```

### Process-to-Data Mapping
```
ProcessDataMapping {
  process: string
  input_data: DataType[]
  output_data: DataType[]
  data_transformations: Transformation[]
  storage_locations: string[]
}

DataType {
  name: string
  format: string (PointCloud|Mesh|Image|Config|Order)
  size_estimate: string
  processing_stage: PipelineStage
}
```

### Communication Model
```
IPCEndpoint {
  type: IPCType (NamedPipe|SharedMemory|Socket)
  name: string
  direction: Direction (Producer|Consumer|Bidirectional)
  data_format: string
}

NetworkEndpoint {
  protocol: Protocol (HTTP|HTTPS|WebSocket|gRPC)
  address: string
  port: number
  authentication: AuthMethod
}
```

### Asset Model
```
Asset {
  path: string
  type: AssetType (QML|Python|Config|Database|Model|Translation)
  size: number
  dependencies: Asset[]
  metadata: Map<string, any>
}
```

### Algorithm Model
```
AlgorithmComponent {
  dll_name: string
  exported_functions: Function[]
  dependencies: Library[]
  gpu_requirements: GPUSpec
  processing_type: ProcessingType (RealTime|Batch|Interactive)
}
```

## Error Handling

### Analysis Failure Recovery
- Graceful handling of corrupted or protected files
- Alternative analysis paths when primary tools fail
- Partial analysis completion with documented limitations

### Tool Integration Failures
- Fallback mechanisms for unavailable analysis tools
- Manual analysis procedures for automated tool failures
- Cross-validation between multiple analysis approaches

### Data Validation
- Consistency checking between different analysis results
- Confidence scoring for inferred relationships
- Documentation of assumptions and limitations

## Testing Strategy

### Static Analysis Validation
- Cross-reference dependency analysis results
- Validate IPC endpoint discovery through multiple methods
- Verify configuration parsing accuracy

### Pipeline Reconstruction Validation
- Compare inferred pipeline with observable application behavior
- Validate algorithm component relationships
- Test data flow assumptions against configuration evidence

### Documentation Accuracy Testing
- Peer review of architecture diagrams
- Validation of technical specifications
- Accuracy verification of component classifications

## Implementation Phases

### Phase 1: Foundation Analysis (Estimated: 3 days)
- Complete process topology mapping
- Basic dependency tree construction
- Initial component classification
- High-value target identification and prioritization

### Phase 2: Communication Discovery (Estimated: 4 days)
- IPC mechanism identification
- Network endpoint mapping
- Service interaction documentation
- Process-to-data flow mapping

### Phase 3: Asset Deep Dive (Estimated: 5 days)
- QML/Python/Config analysis
- Database schema extraction
- AI model format identification
- Configuration hierarchy mapping

### Phase 4: Binary & Algorithm Inspection (Estimated: 6 days)
- Algorithm DLL export analysis
- AI model tensor documentation
- Hardware interface mapping
- GPU processing pipeline analysis

### Phase 5: Architecture Synthesis (Estimated: 4 days)
- Complete pipeline reconstruction
- Architecture diagram generation
- Comprehensive documentation compilation
- Confidence scoring and assumption validation

**Total Estimated Duration: 22 days**

## Deliverables

### Primary Outputs
1. **Process Topology Diagram**: Visual representation of all executables and their relationships
2. **Communication Matrix**: Complete IPC and network communication mapping
3. **Asset Inventory**: Comprehensive catalog of all readable components
4. **Algorithm Documentation**: Detailed analysis of processing components
5. **Pipeline Architecture**: End-to-end scanning process documentation

### Supporting Documentation
- Dependency analysis reports
- Configuration reference guides
- Database schema documentation
- AI model specifications
- Hardware interface documentation

## Security and Legal Considerations

### Ethical Boundaries
- nothing (no limit use all power)

### Safe Analysis Practices

## Example Pipeline Stage

### Acquisition to Registration Flow
```
Camera Stream → Frame Sync → Undistort → Point Cloud
     ↓              ↓           ↓           ↓
DentalScanAppLogic.exe processes raw camera data
     ↓
Incremental ICP Registration (Sn3DRegistration.dll)
     ↓
Pose Estimation & Tracking (Sn3DTextureBasedTrack.dll)
     ↓
TSDF Fusion (Sn3DSpeckleFusion.dll)
     ↓
Mesh Generation → DentalAlgoService.exe
```

## Assumptions & Confidence Tracking

All analysis findings will be tagged with confidence scores:

- **High Confidence (0.8-1.0)**: Direct evidence from strings, exports, or configuration
- **Medium Confidence (0.5-0.7)**: Inferred from naming patterns and dependencies  
- **Low Confidence (0.0-0.4)**: Speculative based on domain knowledge

## Glossary

**QML**: Qt Modeling Language - declarative language for designing user interfaces
**.pyd**: Python Dynamic module - compiled C/C++ extensions for Python on Windows
**OSG**: OpenSceneGraph - 3D graphics toolkit for visualization applications
**CUDA**: Compute Unified Device Architecture - NVIDIA's parallel computing platform
**ICP**: Iterative Closest Point - algorithm for aligning 3D point clouds
**TSDF**: Truncated Signed Distance Function - 3D surface representation method
**DLL**: Dynamic Link Library - Windows shared library format
**IPC**: Inter-Process Communication - methods for processes to exchange data
**ONNX**: Open Neural Network Exchange - standard format for machine learning models
**TensorRT**: NVIDIA's inference optimization library for deep learning
**SQLite**: Embedded relational database engine
**rcc**: Qt Resource Compiler - tool for embedding resources in Qt applications