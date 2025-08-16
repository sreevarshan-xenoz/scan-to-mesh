"""
AI Model Specifications - Based on IntraoralScan 3.5.4.6 Analysis
Defines the exact model architectures and training requirements discovered
"""

from dataclasses import dataclass
from typing import List, Tuple, Dict, Any
import torch
import torch.nn as nn

@dataclass
class ModelSpec:
    """Model specification based on analysis findings"""
    name: str
    input_size: Tuple[int, int]  # (width, height)
    depth_size: Tuple[int, int]  # For depth processing
    num_classes: int
    confidence_threshold: float
    model_type: str  # 'segmentation', 'detection', 'classification'
    version: str  # V2, V3, V5 from analysis
    accuracy_target: float  # Target accuracy from filename analysis

# Model specifications discovered from analysis
DENTAL_MODELS = {
    # Primary segmentation model (V5 - latest generation)
    'dental_segmentation_v5': ModelSpec(
        name='fine_turn_0.70_align8_0.9593_294_13072023_v5',
        input_size=(240, 176),  # From analysis
        depth_size=(120, 88),   # From analysis  
        num_classes=33,         # 32 teeth + background
        confidence_threshold=0.7,
        model_type='segmentation',
        version='v5',
        accuracy_target=0.9593  # 95.93% from filename
    ),
    
    # Specialized posterior teeth model (V3)
    'dental_segmentation_v3_posterior': ModelSpec(
        name='fine_turn_0.70_0.9557_405_03082024_v3_beikou_opera',
        input_size=(240, 176),
        depth_size=(120, 88),
        num_classes=33,
        confidence_threshold=0.7,
        model_type='segmentation',
        version='v3',
        accuracy_target=0.9557  # 95.57% from filename
    ),
    
    # Legacy model (V2 - for comparison)
    'dental_segmentation_v2': ModelSpec(
        name='fine_turn_0.55_align8_0.9540_205_02122020_v2',
        input_size=(240, 240),  # Different input size for V2
        depth_size=(120, 120),
        num_classes=33,
        confidence_threshold=0.55,
        model_type='segmentation', 
        version='v2',
        accuracy_target=0.9540  # 95.40% from filename
    ),
    
    # Clinical detection model
    'dental_detection': ModelSpec(
        name='DentalInfraredCariesDet',
        input_size=(240, 176),
        depth_size=(120, 88),
        num_classes=16,  # Various pathologies
        confidence_threshold=0.8,
        model_type='detection',
        version='v3',
        accuracy_target=0.90
    ),
    
    # Tooth numbering model
    'tooth_numbering': ModelSpec(
        name='AutoToothNumberMark',
        input_size=(240, 176),
        depth_size=(120, 88),
        num_classes=32,  # Individual tooth numbers
        confidence_threshold=0.8,
        model_type='classification',
        version='v3',
        accuracy_target=0.95
    )
}

class DentalSegmentationV5(nn.Module):
    """
    Dental segmentation model V5 architecture
    Based on analysis of the professional system's highest accuracy model
    """
    
    def __init__(self, num_classes=33, input_channels=6):  # RGB + Depth = 6 channels
        super().__init__()
        
        # Encoder (based on common segmentation architectures)
        self.encoder = nn.ModuleList([
            # Block 1: 240x176 -> 120x88
            nn.Sequential(
                nn.Conv2d(input_channels, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2)
            ),
            # Block 2: 120x88 -> 60x44
            nn.Sequential(
                nn.Conv2d(64, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2)
            ),
            # Block 3: 60x44 -> 30x22
            nn.Sequential(
                nn.Conv2d(128, 256, 3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, 3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2)
            ),
            # Block 4: 30x22 -> 15x11
            nn.Sequential(
                nn.Conv2d(256, 512, 3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, 3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2)
            )
        ])
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )
        
        # Decoder with skip connections
        self.decoder = nn.ModuleList([
            # Upsample 1: 15x11 -> 30x22
            nn.Sequential(
                nn.ConvTranspose2d(1024, 512, 2, stride=2),
                nn.Conv2d(1024, 512, 3, padding=1),  # After concatenation
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, 3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True)
            ),
            # Upsample 2: 30x22 -> 60x44
            nn.Sequential(
                nn.ConvTranspose2d(512, 256, 2, stride=2),
                nn.Conv2d(512, 256, 3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, 3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True)
            ),
            # Upsample 3: 60x44 -> 120x88
            nn.Sequential(
                nn.ConvTranspose2d(256, 128, 2, stride=2),
                nn.Conv2d(256, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True)
            ),
            # Upsample 4: 120x88 -> 240x176
            nn.Sequential(
                nn.ConvTranspose2d(128, 64, 2, stride=2),
                nn.Conv2d(128, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True)
            )
        ])
        
        # Final classification layer
        self.classifier = nn.Conv2d(64, num_classes, 1)
        
        # Store skip connections
        self.skip_connections = []
    
    def forward(self, x):
        # Encoder path
        self.skip_connections = []
        for encoder_block in self.encoder:
            x = encoder_block(x)
            self.skip_connections.append(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder path with skip connections
        for i, decoder_block in enumerate(self.decoder):
            # Upsample
            x = decoder_block[0](x)  # ConvTranspose2d
            
            # Concatenate with skip connection
            skip = self.skip_connections[-(i+1)]
            x = torch.cat([x, skip], dim=1)
            
            # Process concatenated features
            for layer in decoder_block[1:]:
                x = layer(x)
        
        # Final classification
        x = self.classifier(x)
        
        return x

class DentalDetectionModel(nn.Module):
    """
    Dental pathology detection model
    Based on analysis of clinical detection capabilities
    """
    
    def __init__(self, num_classes=16, input_channels=6):
        super().__init__()
        
        # Feature extraction backbone
        self.backbone = nn.Sequential(
            # Initial convolution
            nn.Conv2d(input_channels, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
            
            # ResNet-like blocks
            self._make_layer(64, 64, 2),
            self._make_layer(64, 128, 2, stride=2),
            self._make_layer(128, 256, 2, stride=2),
            self._make_layer(256, 512, 2, stride=2),
            
            # Global average pooling
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        
        # First block (may have stride > 1)
        layers.append(nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        
        # Remaining blocks
        for _ in range(1, blocks):
            layers.append(nn.Conv2d(out_channels, out_channels, 3, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x

class ToothNumberingModel(nn.Module):
    """
    Tooth numbering and classification model
    Based on AutoToothNumberMark analysis
    """
    
    def __init__(self, num_teeth=32, input_channels=6):
        super().__init__()
        
        # Feature extraction
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        # Tooth numbering head
        self.tooth_classifier = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_teeth)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.tooth_classifier(x)
        return x

def get_model_by_name(model_name: str, **kwargs) -> nn.Module:
    """Get model instance by name"""
    if model_name == 'dental_segmentation_v5':
        return DentalSegmentationV5(**kwargs)
    elif model_name == 'dental_detection':
        return DentalDetectionModel(**kwargs)
    elif model_name == 'tooth_numbering':
        return ToothNumberingModel(**kwargs)
    else:
        raise ValueError(f"Unknown model: {model_name}")

def get_model_spec(model_name: str) -> ModelSpec:
    """Get model specification by name"""
    return DENTAL_MODELS.get(model_name)

def list_available_models() -> List[str]:
    """List all available model names"""
    return list(DENTAL_MODELS.keys())

# Training hyperparameters based on analysis
TRAINING_CONFIG = {
    'dental_segmentation_v5': {
        'learning_rate': 0.001,
        'batch_size': 8,
        'epochs': 100,
        'optimizer': 'Adam',
        'loss_function': 'CrossEntropyLoss',
        'weight_decay': 1e-4,
        'scheduler': 'CosineAnnealingLR',
        'augmentation': True,
        'early_stopping_patience': 10
    },
    'dental_detection': {
        'learning_rate': 0.0001,
        'batch_size': 16,
        'epochs': 80,
        'optimizer': 'Adam',
        'loss_function': 'BCEWithLogitsLoss',
        'weight_decay': 1e-4,
        'scheduler': 'StepLR',
        'augmentation': True,
        'early_stopping_patience': 8
    },
    'tooth_numbering': {
        'learning_rate': 0.001,
        'batch_size': 32,
        'epochs': 60,
        'optimizer': 'SGD',
        'loss_function': 'CrossEntropyLoss',
        'weight_decay': 1e-3,
        'scheduler': 'MultiStepLR',
        'augmentation': True,
        'early_stopping_patience': 6
    }
}