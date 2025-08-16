"""
Model Training Pipeline - Train dental AI models based on professional specifications
Implements training for segmentation, detection, and classification models
"""

import os
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import matplotlib.pyplot as plt
from tqdm import tqdm
import wandb  # For experiment tracking (optional)

from model_specifications import (
    get_model_by_name, get_model_spec, TRAINING_CONFIG,
    DentalSegmentationV5, DentalDetectionModel, ToothNumberingModel
)
from data_preparation import create_dataloaders, DentalDataset

class ModelTrainer:
    """
    Professional model trainer for dental AI models
    Based on training patterns discovered in analysis
    """
    
    def __init__(self, 
                 model_name: str,
                 data_dir: str,
                 output_dir: str = "models/trained",
                 device: str = "auto",
                 use_wandb: bool = False):
        
        self.model_name = model_name
        self.data_dir = data_dir
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Device setup
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Get model specifications
        self.model_spec = get_model_spec(model_name)
        if self.model_spec is None:
            raise ValueError(f"Unknown model: {model_name}")
        
        self.training_config = TRAINING_CONFIG.get(model_name, {})
        
        # Initialize model
        self.model = self._create_model()
        self.model.to(self.device)
        
        # Training components
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        
        # Tracking
        self.use_wandb = use_wandb
        self.writer = None
        self.best_metric = 0.0
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_metric': [],
            'val_metric': []
        }
        
        # Setup logging
        self._setup_logging()
    
    def _create_model(self) -> nn.Module:
        """Create model instance"""
        model_kwargs = {
            'num_classes': self.model_spec.num_classes,
            'input_channels': 6  # RGB + Depth = 6 channels
        }
        
        if self.model_spec.model_type == 'segmentation':
            return DentalSegmentationV5(**model_kwargs)
        elif self.model_spec.model_type == 'detection':
            return DentalDetectionModel(**model_kwargs)
        elif self.model_spec.model_type == 'classification':
            return ToothNumberingModel(num_teeth=self.model_spec.num_classes, **model_kwargs)
        else:
            raise ValueError(f"Unknown model type: {self.model_spec.model_type}")
    
    def _setup_logging(self):
        """Setup logging and experiment tracking"""
        # TensorBoard
        log_dir = self.output_dir / "logs" / f"{self.model_name}_{int(time.time())}"
        self.writer = SummaryWriter(log_dir)
        
        # Weights & Biases (optional)
        if self.use_wandb:
            wandb.init(
                project="dental-ai-training",
                name=f"{self.model_name}_{int(time.time())}",
                config={
                    'model_name': self.model_name,
                    'model_spec': self.model_spec.__dict__,
                    'training_config': self.training_config,
                    'device': str(self.device)
                }
            )
    
    def _setup_training_components(self):
        """Setup optimizer, scheduler, and loss function"""
        # Optimizer
        optimizer_name = self.training_config.get('optimizer', 'Adam')
        lr = self.training_config.get('learning_rate', 0.001)
        weight_decay = self.training_config.get('weight_decay', 1e-4)
        
        if optimizer_name == 'Adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'SGD':
            self.optimizer = optim.SGD(self.model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
        
        # Scheduler
        scheduler_name = self.training_config.get('scheduler', 'CosineAnnealingLR')
        epochs = self.training_config.get('epochs', 100)
        
        if scheduler_name == 'CosineAnnealingLR':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs)
        elif scheduler_name == 'StepLR':
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.1)
        elif scheduler_name == 'MultiStepLR':
            self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[30, 60], gamma=0.1)
        
        # Loss function
        loss_name = self.training_config.get('loss_function', 'CrossEntropyLoss')
        
        if loss_name == 'CrossEntropyLoss':
            self.criterion = nn.CrossEntropyLoss()
        elif loss_name == 'BCEWithLogitsLoss':
            self.criterion = nn.BCEWithLogitsLoss()
        elif loss_name == 'DiceLoss':
            self.criterion = self._dice_loss
        else:
            raise ValueError(f"Unknown loss function: {loss_name}")
    
    def _dice_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Dice loss for segmentation"""
        smooth = 1e-6
        
        # Apply softmax to predictions
        pred = torch.softmax(pred, dim=1)
        
        # Convert target to one-hot
        target_one_hot = torch.zeros_like(pred)
        target_one_hot.scatter_(1, target.unsqueeze(1), 1)
        
        # Calculate Dice coefficient
        intersection = (pred * target_one_hot).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))
        
        dice = (2 * intersection + smooth) / (union + smooth)
        dice_loss = 1 - dice.mean()
        
        return dice_loss
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        total_metric = 0.0
        num_batches = len(train_loader)
        
        pbar = tqdm(train_loader, desc="Training")
        
        for batch_idx, batch in enumerate(pbar):
            # Move data to device
            images = batch['image'].to(self.device)
            
            if self.model_spec.model_type == 'segmentation':
                targets = batch['mask'].to(self.device)
            else:
                # For detection/classification, targets would be different
                targets = batch.get('labels', torch.zeros(images.size(0))).to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            
            # Calculate loss
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Calculate metrics
            metric = self._calculate_metric(outputs, targets)
            
            # Update totals
            total_loss += loss.item()
            total_metric += metric
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'Metric': f"{metric:.4f}"
            })
        
        avg_loss = total_loss / num_batches
        avg_metric = total_metric / num_batches
        
        return avg_loss, avg_metric
    
    def validate_epoch(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        total_metric = 0.0
        num_batches = len(val_loader)
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc="Validation")
            
            for batch in pbar:
                # Move data to device
                images = batch['image'].to(self.device)
                
                if self.model_spec.model_type == 'segmentation':
                    targets = batch['mask'].to(self.device)
                else:
                    targets = batch.get('labels', torch.zeros(images.size(0))).to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                
                # Calculate loss and metrics
                loss = self.criterion(outputs, targets)
                metric = self._calculate_metric(outputs, targets)
                
                total_loss += loss.item()
                total_metric += metric
                
                pbar.set_postfix({
                    'Loss': f"{loss.item():.4f}",
                    'Metric': f"{metric:.4f}"
                })
        
        avg_loss = total_loss / num_batches
        avg_metric = total_metric / num_batches
        
        return avg_loss, avg_metric
    
    def _calculate_metric(self, outputs: torch.Tensor, targets: torch.Tensor) -> float:
        """Calculate appropriate metric based on model type"""
        if self.model_spec.model_type == 'segmentation':
            # IoU (Intersection over Union) for segmentation
            return self._calculate_iou(outputs, targets)
        else:
            # Accuracy for detection/classification
            return self._calculate_accuracy(outputs, targets)
    
    def _calculate_iou(self, outputs: torch.Tensor, targets: torch.Tensor) -> float:
        """Calculate IoU for segmentation"""
        with torch.no_grad():
            # Get predictions
            preds = torch.argmax(outputs, dim=1)
            
            # Calculate IoU for each class and average
            ious = []
            for class_id in range(self.model_spec.num_classes):
                pred_mask = (preds == class_id)
                target_mask = (targets == class_id)
                
                intersection = (pred_mask & target_mask).sum().float()
                union = (pred_mask | target_mask).sum().float()
                
                if union > 0:
                    iou = intersection / union
                    ious.append(iou.item())
            
            return np.mean(ious) if ious else 0.0
    
    def _calculate_accuracy(self, outputs: torch.Tensor, targets: torch.Tensor) -> float:
        """Calculate accuracy for classification"""
        with torch.no_grad():
            if len(outputs.shape) > 1 and outputs.shape[1] > 1:
                # Multi-class
                preds = torch.argmax(outputs, dim=1)
            else:
                # Binary
                preds = (torch.sigmoid(outputs) > 0.5).float()
            
            correct = (preds == targets).sum().float()
            total = targets.size(0)
            
            return (correct / total).item()
    
    def train(self, 
              train_loader: DataLoader,
              val_loader: DataLoader,
              epochs: Optional[int] = None) -> Dict[str, Any]:
        """Main training loop"""
        
        if epochs is None:
            epochs = self.training_config.get('epochs', 100)
        
        # Setup training components
        self._setup_training_components()
        
        # Early stopping
        patience = self.training_config.get('early_stopping_patience', 10)
        best_val_metric = 0.0
        patience_counter = 0
        
        print(f"Starting training for {epochs} epochs...")
        print(f"Model: {self.model_name}")
        print(f"Target accuracy: {self.model_spec.accuracy_target:.1%}")
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            
            # Train
            train_loss, train_metric = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_metric = self.validate_epoch(val_loader)
            
            # Update scheduler
            if self.scheduler:
                self.scheduler.step()
            
            # Log metrics
            self._log_metrics(epoch, train_loss, train_metric, val_loss, val_metric)
            
            # Save best model
            if val_metric > best_val_metric:
                best_val_metric = val_metric
                self._save_checkpoint(epoch, val_metric, is_best=True)
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= patience:
                print(f"Early stopping triggered after {patience} epochs without improvement")
                break
            
            # Regular checkpoint
            if (epoch + 1) % 10 == 0:
                self._save_checkpoint(epoch, val_metric, is_best=False)
        
        # Final results
        results = {
            'best_val_metric': best_val_metric,
            'target_accuracy': self.model_spec.accuracy_target,
            'achieved_target': best_val_metric >= self.model_spec.accuracy_target,
            'training_history': self.training_history
        }
        
        print(f"\nTraining completed!")
        print(f"Best validation metric: {best_val_metric:.4f}")
        print(f"Target accuracy: {self.model_spec.accuracy_target:.4f}")
        print(f"Target achieved: {results['achieved_target']}")
        
        return results
    
    def _log_metrics(self, epoch: int, train_loss: float, train_metric: float, 
                    val_loss: float, val_metric: float):
        """Log training metrics"""
        
        # Update history
        self.training_history['train_loss'].append(train_loss)
        self.training_history['val_loss'].append(val_loss)
        self.training_history['train_metric'].append(train_metric)
        self.training_history['val_metric'].append(val_metric)
        
        # TensorBoard
        self.writer.add_scalar('Loss/Train', train_loss, epoch)
        self.writer.add_scalar('Loss/Validation', val_loss, epoch)
        self.writer.add_scalar('Metric/Train', train_metric, epoch)
        self.writer.add_scalar('Metric/Validation', val_metric, epoch)
        self.writer.add_scalar('Learning_Rate', self.optimizer.param_groups[0]['lr'], epoch)
        
        # Weights & Biases
        if self.use_wandb:
            wandb.log({
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_metric': train_metric,
                'val_metric': val_metric,
                'learning_rate': self.optimizer.param_groups[0]['lr']
            })
        
        # Console output
        print(f"Train Loss: {train_loss:.4f}, Train Metric: {train_metric:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Metric: {val_metric:.4f}")
    
    def _save_checkpoint(self, epoch: int, metric: float, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'metric': metric,
            'model_spec': self.model_spec.__dict__,
            'training_config': self.training_config,
            'training_history': self.training_history
        }
        
        # Save regular checkpoint
        checkpoint_path = self.output_dir / f"{self.model_name}_epoch_{epoch}.pth"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.output_dir / f"{self.model_name}_best.pth"
            torch.save(checkpoint, best_path)
            print(f"✓ Best model saved: {best_path}")
    
    def export_to_onnx(self, checkpoint_path: str, output_path: str):
        """Export trained model to ONNX format"""
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(1, 6, self.model_spec.input_size[1], self.model_spec.input_size[0])
        dummy_input = dummy_input.to(self.device)
        
        # Export to ONNX
        torch.onnx.export(
            self.model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        print(f"✓ Model exported to ONNX: {output_path}")

def train_all_models(data_dir: str, output_dir: str = "models/trained"):
    """Train all dental AI models"""
    
    models_to_train = [
        'dental_segmentation_v5',
        'dental_detection', 
        'tooth_numbering'
    ]
    
    results = {}
    
    for model_name in models_to_train:
        print(f"\n{'='*50}")
        print(f"Training {model_name}")
        print(f"{'='*50}")
        
        try:
            # Create trainer
            trainer = ModelTrainer(
                model_name=model_name,
                data_dir=data_dir,
                output_dir=output_dir,
                use_wandb=False  # Set to True if you want to use W&B
            )
            
            # Get model spec for appropriate data loader
            model_spec = get_model_spec(model_name)
            
            # Create data loaders
            train_loader, val_loader, test_loader = create_dataloaders(
                data_dir=data_dir,
                model_type=model_spec.model_type,
                batch_size=trainer.training_config.get('batch_size', 8)
            )
            
            # Train model
            result = trainer.train(train_loader, val_loader)
            results[model_name] = result
            
            # Export to ONNX
            best_checkpoint = Path(output_dir) / f"{model_name}_best.pth"
            onnx_path = Path(output_dir) / f"{model_name}.onnx"
            
            if best_checkpoint.exists():
                trainer.export_to_onnx(str(best_checkpoint), str(onnx_path))
            
        except Exception as e:
            print(f"Error training {model_name}: {e}")
            results[model_name] = {'error': str(e)}
    
    # Save overall results
    results_file = Path(output_dir) / "training_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*50}")
    print("Training Summary")
    print(f"{'='*50}")
    
    for model_name, result in results.items():
        if 'error' in result:
            print(f"{model_name}: ERROR - {result['error']}")
        else:
            achieved = result.get('achieved_target', False)
            best_metric = result.get('best_val_metric', 0.0)
            target = result.get('target_accuracy', 0.0)
            
            status = "✓ PASSED" if achieved else "✗ FAILED"
            print(f"{model_name}: {status} ({best_metric:.3f} / {target:.3f})")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Dental AI Models')
    parser.add_argument('--data-dir', type=str, required=True, help='Dataset directory')
    parser.add_argument('--output-dir', type=str, default='models/trained', help='Output directory')
    parser.add_argument('--model', type=str, help='Specific model to train')
    parser.add_argument('--create-synthetic', action='store_true', help='Create synthetic dataset')
    
    args = parser.parse_args()
    
    # Create synthetic dataset if requested
    if args.create_synthetic:
        from data_preparation import create_synthetic_dataset
        print("Creating synthetic dataset...")
        create_synthetic_dataset(args.data_dir, num_samples=2000)
    
    # Train models
    if args.model:
        # Train specific model
        trainer = ModelTrainer(
            model_name=args.model,
            data_dir=args.data_dir,
            output_dir=args.output_dir
        )
        
        model_spec = get_model_spec(args.model)
        train_loader, val_loader, test_loader = create_dataloaders(
            data_dir=args.data_dir,
            model_type=model_spec.model_type,
            batch_size=trainer.training_config.get('batch_size', 8)
        )
        
        trainer.train(train_loader, val_loader)
    else:
        # Train all models
        train_all_models(args.data_dir, args.output_dir)