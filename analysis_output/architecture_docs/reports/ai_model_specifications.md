# AI Model Specifications

Generated: 2025-08-15T23:26:35.119155

## Model Inventory

- **Total Inferred Models**: 11

### Models by Type

- Dental AI: 1 models
- Detection: 5 models
- Classification: 1 models
- Segmentation: 3 models
- Unknown AI Model: 1 models

### Model Details

- **Sn3DDentalAI.dll**
  - Type: Dental AI
  - Stage: ai_analysis
  - Confidence: 0.7

- **Sn3DDentalBoxDet.dll**
  - Type: Detection
  - Stage: ai_analysis
  - Confidence: 0.35

- **Sn3DDentalOralCls.dll**
  - Type: Classification
  - Stage: ai_analysis
  - Confidence: 0.35

- **Sn3DDentalRealTimeSemSeg.dll**
  - Type: Segmentation
  - Stage: ai_analysis
  - Confidence: 0.35

- **Sn3DInfraredCariesDet.dll**
  - Type: Detection
  - Stage: ai_analysis
  - Confidence: 0.35

- **Sn3DLandmarksDetForCli.dll**
  - Type: Detection
  - Stage: ai_analysis
  - Confidence: 0.35

- **Sn3DCBCTSegLite.dll**
  - Type: Segmentation
  - Stage: unknown
  - Confidence: 0.35

- **Sn3DDentalAICBCTSeg.dll**
  - Type: Segmentation
  - Stage: unknown
  - Confidence: 0.35

- **Sn3DDentalAbutmentCloudDet.dll**
  - Type: Detection
  - Stage: unknown
  - Confidence: 0.35

- **Sn3DDentalGeometryFeatureDet.dll**
  - Type: Detection
  - Stage: unknown
  - Confidence: 0.35

- **Sn3DAIOutUnity.dll**
  - Type: Unknown AI Model
  - Stage: visualization
  - Confidence: 0.35

## Algorithm Analysis

- **Total Algorithm DLLs**: 62
- **High Confidence DLLs**: 0
- **Total Exported Functions**: 450

### Processing Stages

#### Acquisition (2 DLLs)

- Sn3DPhotometricStereo.dll (confidence: 0.35)
- Sn3DRealtimeScan.dll (confidence: 0.7)

#### Ai_Analysis (6 DLLs)

- Sn3DDentalAI.dll (confidence: 0.7)
- Sn3DDentalBoxDet.dll (confidence: 0.35)
- Sn3DDentalOralCls.dll (confidence: 0.35)
- Sn3DDentalRealTimeSemSeg.dll (confidence: 0.35)
- Sn3DInfraredCariesDet.dll (confidence: 0.35)
- Sn3DLandmarksDetForCli.dll (confidence: 0.35)

#### Algorithms (10 DLLs)

- algorithm1.dll (confidence: 0.7)
- algorithm2.dll (confidence: 0.7)
- algorithmHlj.dll (confidence: 0.35)
- algorithmLy.dll (confidence: 0.35)
- algorithmLzy.dll (confidence: 0.7)
- algorithmMigrate.dll (confidence: 0.35)
- algorithmSaj.dll (confidence: 0.35)
- algorithmZbt.dll (confidence: 0.35)
- algorithmZys.dll (confidence: 0.35)
- algorithmlyc.dll (confidence: 0.35)

#### Calibration (1 DLLs)

- Sn3DCalibrationJR.dll (confidence: 0.7)

#### Dental_Specific (5 DLLs)

- Sn3DCrownGen.dll (confidence: 0.35)
- Sn3DDental.dll (confidence: 0.35)
- Sn3DDentalDesktop.dll (confidence: 0.35)
- Sn3DDentalOral.dll (confidence: 0.35)
- Sn3DOrthoEx.dll (confidence: 0.35)

#### Fusion (3 DLLs)

- Sn3DPhaseBuild.dll (confidence: 0.35)
- Sn3DSpeckle.dll (confidence: 0.35)
- Sn3DSpeckleFusion.dll (confidence: 0.7)

#### Mesh_Processing (4 DLLs)

- Sn3DCork.dll (confidence: 0.35)
- Sn3DDraco.dll (confidence: 0.35)
- Sn3DMagic.dll (confidence: 0.7)
- Sn3DTooling.dll (confidence: 0.35)

#### Registration (3 DLLs)

- Sn3DRegistration.dll (confidence: 0.7)
- Sn3DScanSlam.dll (confidence: 0.35)
- Sn3DTextureBasedTrack.dll (confidence: 0.35)

#### Unknown (25 DLLs)

- Sn3DAVXImp.dll (confidence: 0.35)
- Sn3DApplicationCommon.dll (confidence: 0.35)
- Sn3DApplicationRenderKit.dll (confidence: 0.35)
- Sn3DApplicationUtility.dll (confidence: 0.35)
- Sn3DCBCTLoader.dll (confidence: 0.35)
- Sn3DCBCTSegLite.dll (confidence: 0.35)
- Sn3DColorCorrect.dll (confidence: 0.35)
- Sn3DCrypto.dll (confidence: 0.35)
- Sn3DDentalAICBCTSeg.dll (confidence: 0.35)
- Sn3DDentalAbutmentCloudDet.dll (confidence: 0.35)
- Sn3DDentalGeometryFeatureDet.dll (confidence: 0.35)
- Sn3DDentalMeshParameterize.dll (confidence: 0.35)
- Sn3DDigital.dll (confidence: 0.35)
- Sn3DFaceParsing.dll (confidence: 0.35)
- Sn3DFaceScan.dll (confidence: 0.35)
- Sn3DGeometricTrackFusion.dll (confidence: 0.35)
- Sn3DInspect.dll (confidence: 0.35)
- Sn3DInspection.dll (confidence: 0.35)
- Sn3DLineCode.dll (confidence: 0.35)
- Sn3DOralExamCCP.dll (confidence: 0.35)
- Sn3DOralExamResidualCrown.dll (confidence: 0.35)
- Sn3DOralExamWedgeDefect.dll (confidence: 0.35)
- Sn3DSparseSolver.dll (confidence: 0.35)
- Sn3DTextureSlam.dll (confidence: 0.35)
- Sn3DVulkanWrapper.dll (confidence: 0.35)

#### Visualization (3 DLLs)

- Sn3DAIOutUnity.dll (confidence: 0.35)
- Sn3DFace.dll (confidence: 0.35)
- Sn3DFaceUnity.dll (confidence: 0.35)

## Processing Pipeline

### Function Types Distribution

- processing: 156 functions
- initialization: 19 functions
- calibration: 31 functions
- utility: 35 functions
- registration: 12 functions
- analysis: 73 functions
- fusion: 103 functions
- geometry: 20 functions
- gpu_compute: 1 functions

### Algorithm Stages

- calibration: 52 functions
- ai_analysis: 93 functions
- mesh_processing: 101 functions
- acquisition: 50 functions
- registration: 50 functions
- fusion: 65 functions
- general: 39 functions

## Tensor Documentation

- **Functions with Tensor Hints**: 2

### Common Tensor Patterns

- Supports batch processing: 2 occurrences

## Confidence Assessment

- **Average DLL Confidence**: 0.401
- **Average Function Confidence**: 0.835

### Analysis Completeness

- Dll Analysis: Complete
- Function Analysis: Complete
- Tensor Analysis: Inferred
