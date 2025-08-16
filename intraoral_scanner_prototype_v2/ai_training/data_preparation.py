"""
Data Preparation for Dental AI Models
Creates training datasets based on professional system requirements
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
import json
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

class DentalDataset(Dataset):
    """
    Dental dataset for training segmentation and detection models
    Based on the input specifications discovered in analysis
    """
    
    def __init__(self, 
                 data_dir: str,
                 mode: str = 'train',  # 'train', 'val', 'test'
                 model_type: str = 'segmentation',  # 'segmentation', 'detection', 'classification'
                 input_size: Tuple[int, int] = (240, 176),
                 depth_size: Tuple[int, int] = (120, 88),
                 augment: bool = True):
        
        self.data_dir = Path(data_dir)
        self.mode = mode
        self.model_type = model_type
        self.input_size = input_size
        self.depth_size = depth_size
        self.augment = augment and mode == 'train'
        
        # Load data paths and annotations
        self.samples = self._load_samples()
        
        # Setup augmentations
        self.transforms = self._setup_transforms()
        
        print(f"Loaded {len(self.samples)} samples for {mode} mode")
    
    def _load_samples(self) -> List[Dict[str, Any]]:
        """Load dataset samples from directory structure"""
        samples = []
        
        # Expected directory structure:
        # data_dir/
        #   ├── images/           # RGB images
        #   ├── depth/            # Depth images  
        #   ├── masks/            # Segmentation masks (for segmentation)
        #   ├── annotations/      # Detection annotations (for detection)
        #   └── metadata.json     # Dataset metadata
        
        images_dir = self.data_dir / 'images'
        depth_dir = self.data_dir / 'depth'
        masks_dir = self.data_dir / 'masks'
        annotations_dir = self.data_dir / 'annotations'
        
        if not images_dir.exists():
            raise ValueError(f"Images directory not found: {images_dir}")
        
        # Load metadata if available
        metadata_file = self.data_dir / 'metadata.json'
        metadata = {}
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
        
        # Get all image files
        image_files = list(images_dir.glob('*.png')) + list(images_dir.glob('*.jpg'))
        
        for img_path in sorted(image_files):
            sample_id = img_path.stem
            
            # Check if corresponding files exist
            depth_path = depth_dir / f"{sample_id}.png"
            mask_path = masks_dir / f"{sample_id}.png"
            annotation_path = annotations_dir / f"{sample_id}.json"
            
            sample = {
                'id': sample_id,
                'image_path': str(img_path),
                'depth_path': str(depth_path) if depth_path.exists() else None,
                'mask_path': str(mask_path) if mask_path.exists() else None,
                'annotation_path': str(annotation_path) if annotation_path.exists() else None,
                'metadata': metadata.get(sample_id, {})
            }
            
            # Filter based on model type requirements
            if self.model_type == 'segmentation' and sample['mask_path'] is None:
                continue
            elif self.model_type == 'detection' and sample['annotation_path'] is None:
                continue
            
            samples.append(sample)
        
        return samples
    
    def _setup_transforms(self) -> A.Compose:
        """Setup data augmentation transforms"""
        if self.augment:
            # Training augmentations (matching professional system robustness)
            transform_list = [
                A.Resize(self.input_size[1], self.input_size[0]),  # height, width
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.3),
                A.GaussianBlur(blur_limit=3, p=0.2),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
                A.Rotate(limit=15, p=0.3),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.3),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ]
        else:
            # Validation/test transforms
            transform_list = [
                A.Resize(self.input_size[1], self.input_size[0]),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ]
        
        return A.Compose(transform_list)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        
        # Load RGB image
        image = cv2.imread(sample['image_path'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load depth image if available
        depth = None
        if sample['depth_path']:
            depth = cv2.imread(sample['depth_path'], cv2.IMREAD_GRAYSCALE)
            depth = depth.astype(np.float32) / 255.0  # Normalize to [0, 1]
        else:
            # Create dummy depth if not available
            depth = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
        
        # Combine RGB and depth into 6-channel input (matching analysis)
        if len(image.shape) == 3:
            # Stack RGB + depth to create 6-channel input
            combined_input = np.concatenate([
                image,
                np.stack([depth, depth, depth], axis=2)  # Replicate depth to 3 channels
            ], axis=2)
        else:
            combined_input = image
        
        result = {'image': combined_input, 'sample_id': sample['id']}
        
        # Load target based on model type
        if self.model_type == 'segmentation':
            mask = cv2.imread(sample['mask_path'], cv2.IMREAD_GRAYSCALE)
            result['mask'] = mask
            
        elif self.model_type == 'detection':
            # Load detection annotations
            with open(sample['annotation_path'], 'r') as f:
                annotations = json.load(f)
            result['annotations'] = annotations
            
        elif self.model_type == 'classification':
            # Load classification labels
            labels = sample['metadata'].get('labels', [])
            result['labels'] = labels
        
        # Apply transforms
        if self.model_type == 'segmentation':
            transformed = self.transforms(image=combined_input, mask=result['mask'])
            result['image'] = transformed['image']
            result['mask'] = torch.from_numpy(transformed['mask']).long()
        else:
            transformed = self.transforms(image=combined_input)
            result['image'] = transformed['image']
        
        return result

def create_synthetic_dataset(output_dir: str, num_samples: int = 1000):
    """
    Create synthetic dental dataset for training when real data is not available
    Generates realistic dental scan images with corresponding masks
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create directory structure
    (output_path / 'images').mkdir(exist_ok=True)
    (output_path / 'depth').mkdir(exist_ok=True)
    (output_path / 'masks').mkdir(exist_ok=True)
    (output_path / 'annotations').mkdir(exist_ok=True)
    
    print(f"Creating synthetic dataset with {num_samples} samples...")
    
    metadata = {}
    
    for i in range(num_samples):
        sample_id = f"synthetic_{i:06d}"
        
        # Generate synthetic dental image
        image, depth, mask, annotations = _generate_synthetic_dental_sample()
        
        # Save files
        cv2.imwrite(str(output_path / 'images' / f"{sample_id}.png"), 
                   cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(output_path / 'depth' / f"{sample_id}.png"), 
                   (depth * 255).astype(np.uint8))
        cv2.imwrite(str(output_path / 'masks' / f"{sample_id}.png"), mask)
        
        # Save annotations
        with open(output_path / 'annotations' / f"{sample_id}.json", 'w') as f:
            json.dump(annotations, f)
        
        # Add to metadata
        metadata[sample_id] = {
            'tooth_count': len(annotations.get('teeth', [])),
            'scan_type': 'synthetic',
            'quality': 'high'
        }
        
        if (i + 1) % 100 == 0:
            print(f"Generated {i + 1}/{num_samples} samples")
    
    # Save metadata
    with open(output_path / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✓ Synthetic dataset created at {output_path}")

def _generate_synthetic_dental_sample() -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """Generate a single synthetic dental sample"""
    # Image dimensions matching analysis
    height, width = 176, 240
    
    # Create base oral cavity image
    image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Generate background (oral cavity)
    image[:, :] = [80, 60, 60]  # Dark reddish background
    
    # Generate teeth (white/cream colored regions)
    num_teeth = np.random.randint(20, 32)  # Realistic tooth count
    teeth_annotations = []
    mask = np.zeros((height, width), dtype=np.uint8)
    
    for tooth_id in range(1, num_teeth + 1):
        # Random tooth position and size
        center_x = np.random.randint(20, width - 20)
        center_y = np.random.randint(20, height - 20)
        tooth_width = np.random.randint(8, 20)
        tooth_height = np.random.randint(12, 25)
        
        # Draw tooth (ellipse)
        tooth_color = [
            np.random.randint(200, 255),  # R
            np.random.randint(200, 255),  # G  
            np.random.randint(180, 240)   # B (slightly less blue for realistic color)
        ]
        
        cv2.ellipse(image, (center_x, center_y), (tooth_width//2, tooth_height//2), 
                   0, 0, 360, tooth_color, -1)
        
        # Add to mask
        cv2.ellipse(mask, (center_x, center_y), (tooth_width//2, tooth_height//2),
                   0, 0, 360, tooth_id, -1)
        
        # Add annotation
        teeth_annotations.append({
            'tooth_id': tooth_id,
            'center': [center_x, center_y],
            'size': [tooth_width, tooth_height],
            'confidence': 1.0
        })
    
    # Generate gums (pink regions)
    gum_color = [150, 100, 120]
    for _ in range(5):  # Add some gum regions
        center_x = np.random.randint(0, width)
        center_y = np.random.randint(0, height)
        radius = np.random.randint(15, 40)
        cv2.circle(image, (center_x, center_y), radius, gum_color, -1)
    
    # Generate synthetic depth map
    depth = np.random.uniform(0.3, 0.8, (height, width)).astype(np.float32)
    
    # Add some noise and blur for realism
    noise = np.random.normal(0, 0.02, image.shape).astype(np.uint8)
    image = np.clip(image.astype(int) + noise, 0, 255).astype(np.uint8)
    
    # Blur slightly for realism
    image = cv2.GaussianBlur(image, (3, 3), 0)
    depth = cv2.GaussianBlur(depth, (3, 3), 0)
    
    annotations = {
        'teeth': teeth_annotations,
        'tooth_count': len(teeth_annotations),
        'image_size': [width, height]
    }
    
    return image, depth, mask, annotations

def split_dataset(data_dir: str, train_ratio: float = 0.7, val_ratio: float = 0.15, test_ratio: float = 0.15):
    """Split dataset into train/validation/test sets"""
    dataset = DentalDataset(data_dir, mode='train', augment=False)
    
    # Get all sample IDs
    sample_ids = [sample['id'] for sample in dataset.samples]
    
    # Split into train/val/test
    train_ids, temp_ids = train_test_split(sample_ids, test_size=(1-train_ratio), random_state=42)
    val_ids, test_ids = train_test_split(temp_ids, test_size=(test_ratio/(val_ratio+test_ratio)), random_state=42)
    
    # Save splits
    splits = {
        'train': train_ids,
        'validation': val_ids,
        'test': test_ids
    }
    
    splits_file = Path(data_dir) / 'dataset_splits.json'
    with open(splits_file, 'w') as f:
        json.dump(splits, f, indent=2)
    
    print(f"Dataset split saved to {splits_file}")
    print(f"Train: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)}")
    
    return splits

def create_dataloaders(data_dir: str, 
                      model_type: str = 'segmentation',
                      batch_size: int = 8,
                      num_workers: int = 4) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/validation/test dataloaders"""
    
    # Load dataset splits
    splits_file = Path(data_dir) / 'dataset_splits.json'
    if splits_file.exists():
        with open(splits_file, 'r') as f:
            splits = json.load(f)
    else:
        print("No dataset splits found, creating new splits...")
        splits = split_dataset(data_dir)
    
    # Create datasets
    train_dataset = DentalDataset(data_dir, mode='train', model_type=model_type, augment=True)
    val_dataset = DentalDataset(data_dir, mode='val', model_type=model_type, augment=False)
    test_dataset = DentalDataset(data_dir, mode='test', model_type=model_type, augment=False)
    
    # Filter samples based on splits
    train_dataset.samples = [s for s in train_dataset.samples if s['id'] in splits['train']]
    val_dataset.samples = [s for s in val_dataset.samples if s['id'] in splits['validation']]
    test_dataset.samples = [s for s in test_dataset.samples if s['id'] in splits['test']]
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                           num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    
    return train_loader, val_loader, test_loader

def visualize_sample(dataset: DentalDataset, idx: int = 0):
    """Visualize a dataset sample"""
    sample = dataset[idx]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image (first 3 channels)
    image = sample['image'][:3].permute(1, 2, 0).numpy()
    image = (image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]))
    image = np.clip(image, 0, 1)
    axes[0].imshow(image)
    axes[0].set_title('RGB Image')
    axes[0].axis('off')
    
    # Depth (last 3 channels, take first one)
    depth = sample['image'][3].numpy()
    axes[1].imshow(depth, cmap='viridis')
    axes[1].set_title('Depth Map')
    axes[1].axis('off')
    
    # Mask (if available)
    if 'mask' in sample:
        mask = sample['mask'].numpy()
        axes[2].imshow(mask, cmap='tab20')
        axes[2].set_title('Segmentation Mask')
        axes[2].axis('off')
    else:
        axes[2].text(0.5, 0.5, 'No mask available', ha='center', va='center')
        axes[2].set_title('Mask')
        axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Example usage
    print("Creating synthetic dataset...")
    create_synthetic_dataset("data/synthetic_dental", num_samples=1000)
    
    print("Splitting dataset...")
    split_dataset("data/synthetic_dental")
    
    print("Creating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        "data/synthetic_dental", 
        model_type='segmentation',
        batch_size=8
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Visualize a sample
    dataset = DentalDataset("data/synthetic_dental", mode='train', model_type='segmentation')
    visualize_sample(dataset, idx=0)