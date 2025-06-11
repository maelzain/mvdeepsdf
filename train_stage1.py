#!/usr/bin/env python3
"""
train_mv_stage1.py

STAGE 1 TRAINING: Pre-train the SDF decoder on watertight CAD models.
This trains the SDF decoder that will be used in Stage 2.

Usage:
    python train_mv_stage1.py --cars_dir ~/Desktop/Mahdi/cars --output_dir ./experiments/stage1
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Import from your deep_sdf module
from deep_sdf.mv_deepsdf import MVDeepSDF
from deep_sdf.data import ShapeNetDatasetMV
from torch.utils.data import DataLoader


class Stage1Trainer:
    """Trainer for Stage 1: SDF decoder pre-training"""
    
    def __init__(self, config, device):
        self.config = config
        self.device = device
        
        # Create model (we only train the SDF decoder in stage 1)
        self.model = MVDeepSDF(latent_dim=config['latent_dim']).to(device)
        self.sdf_decoder = self.model.sdf_decoder
        
        # Loss and optimizer
        self.criterion = nn.L1Loss()  # L1 loss like original DeepSDF
        self.optimizer = optim.Adam(
            self.sdf_decoder.parameters(),
            lr=config['learning_rate'],
            weight_decay=config.get('weight_decay', 1e-6)
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )
        
        # Setup directories
        self.output_dir = Path(config['output_dir'])
        self.checkpoints_dir = self.output_dir / 'checkpoints'
        self.logs_dir = self.output_dir / 'logs'
        
        for dir_path in [self.output_dir, self.checkpoints_dir, self.logs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Tensorboard writer
        self.writer = SummaryWriter(self.logs_dir / 'stage1')
        
        # Training state
        self.epoch = 0
        self.best_loss = float('inf')
        
        print(f"Stage 1 Trainer initialized on {device}")
        print(f"Output directory: {self.output_dir}")
    
    def save_checkpoint(self, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_loss': self.best_loss,
            'config': self.config
        }
        
        # Save latest
        latest_path = self.checkpoints_dir / 'latest.pth'
        torch.save(checkpoint, latest_path)
        
        # Save best
        if is_best:
            best_path = self.checkpoints_dir / 'best.pth'
            torch.save(checkpoint, best_path)
            print(f"✓ Best checkpoint saved with loss: {self.best_loss:.6f}")
    
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = len(train_loader)
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {self.epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Get data
            query_points = batch['query_points'].to(self.device)  # (B, M, 3)
            sdf_values = batch['sdf_values'].to(self.device)      # (B, M)
            
            batch_size = query_points.shape[0]
            
            # Generate random latent codes (in practice these would be optimized)
            latent_codes = torch.randn(batch_size, self.config['latent_dim'], device=self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            predicted_sdf = self.sdf_decoder(query_points, latent_codes)  # (B, M, 1)
            predicted_sdf = predicted_sdf.squeeze(-1)  # (B, M)
            
            # Compute loss
            loss = self.criterion(predicted_sdf, sdf_values)
            
            # Add latent regularization
            latent_reg = torch.mean(torch.norm(latent_codes, dim=-1)) * self.config.get('latent_reg_weight', 1e-4)
            total_loss_batch = loss + latent_reg
            
            # Backward pass
            total_loss_batch.backward()
            torch.nn.utils.clip_grad_norm_(self.sdf_decoder.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.6f}',
                'LatReg': f'{latent_reg.item():.6f}',
                'LR': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
            })
            
            # Log to tensorboard
            global_step = self.epoch * num_batches + batch_idx
            self.writer.add_scalar('Train/Loss', loss.item(), global_step)
            self.writer.add_scalar('Train/LatentReg', latent_reg.item(), global_step)
        
        return total_loss / num_batches
    
    def validate_epoch(self, val_loader):
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        num_batches = len(val_loader)
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                query_points = batch['query_points'].to(self.device)
                sdf_values = batch['sdf_values'].to(self.device)
                batch_size = query_points.shape[0]
                
                # Generate random latent codes
                latent_codes = torch.randn(batch_size, self.config['latent_dim'], device=self.device)
                
                # Forward pass
                predicted_sdf = self.sdf_decoder(query_points, latent_codes)
                predicted_sdf = predicted_sdf.squeeze(-1)
                
                # Compute loss
                loss = self.criterion(predicted_sdf, sdf_values)
                total_loss += loss.item()
        
        return total_loss / num_batches
    
    def train(self, train_loader, val_loader, num_epochs):
        """Main training loop"""
        print(f"Starting Stage 1 training for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            self.epoch = epoch + 1
            
            # Train and validate
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate_epoch(val_loader)
            
            # Update learning rate
            self.scheduler.step(val_loss)
            
            # Log results
            self.writer.add_scalar('Epoch/TrainLoss', train_loss, self.epoch)
            self.writer.add_scalar('Epoch/ValLoss', val_loss, self.epoch)
            
            print(f"Epoch {self.epoch:4d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
            
            # Save checkpoint
            is_best = val_loss < self.best_loss
            if is_best:
                self.best_loss = val_loss
            
            if self.epoch % self.config.get('save_every', 10) == 0 or is_best:
                self.save_checkpoint(is_best)
        
        print("✓ Stage 1 training completed!")
        self.writer.close()


def create_cars_loaders(cars_dir, batch_size=32, num_workers=4, num_sdf_samples=16384):
    """Create train/val/test data loaders for car data"""
    
    # Custom dataset class for cars directory
    class CarsDataset(torch.utils.data.Dataset):
        def __init__(self, cars_dir, split='train', num_sdf_samples=16384):
            self.cars_dir = Path(cars_dir)
            self.split = split
            self.num_sdf_samples = num_sdf_samples
            
            # Find all car models
            self.mesh_list = self._load_mesh_list()
            print(f"Loaded {len(self.mesh_list)} car meshes for {split} split")
        
        def _load_mesh_list(self):
            """Find all mesh files"""
            mesh_list = []
            
            for subfolder in self.cars_dir.iterdir():
                if subfolder.is_dir():
                    # Look for mesh files
                    for mesh_file in ['model.obj', 'model.npz', 'model.ply', 'model_normalized.obj']:
                        mesh_path = subfolder / mesh_file
                        if mesh_path.exists():
                            mesh_list.append(str(mesh_path))
                            break
            
            # Split data
            total_meshes = len(mesh_list)
            if self.split == 'train':
                mesh_list = mesh_list[:int(0.8 * total_meshes)]
            elif self.split == 'val':
                mesh_list = mesh_list[int(0.8 * total_meshes):int(0.9 * total_meshes)]
            else:  # test
                mesh_list = mesh_list[int(0.9 * total_meshes):]
            
            return mesh_list
        
        def __len__(self):
            return len(self.mesh_list)
        
        def __getitem__(self, idx):
            mesh_path = self.mesh_list[idx]
            
            try:
                from deep_sdf.data import sample_sdf_points
                query_points, sdf_values = sample_sdf_points(mesh_path, self.num_sdf_samples)
                
                return {
                    'query_points': query_points,     # (num_sdf_samples, 3)
                    'sdf_values': sdf_values,         # (num_sdf_samples,)
                    'mesh_path': mesh_path
                }
            except Exception as e:
                print(f"Error loading {mesh_path}: {e}")
                return {
                    'query_points': torch.randn(self.num_sdf_samples, 3),
                    'sdf_values': torch.randn(self.num_sdf_samples),
                    'mesh_path': mesh_path
                }
    
    datasets = {}
    data_loaders = {}
    
    for split in ['train', 'val', 'test']:
        datasets[split] = CarsDataset(
            cars_dir=cars_dir, 
            split=split, 
            num_sdf_samples=num_sdf_samples
        )
        data_loaders[split] = DataLoader(
            datasets[split],
            batch_size=batch_size,
            shuffle=(split == 'train'),
            num_workers=num_workers,
            pin_memory=True,
            drop_last=(split == 'train')
        )
    
    return data_loaders


def main():
    parser = argparse.ArgumentParser(description='MV-DeepSDF Stage 1 Training')
    parser.add_argument('--cars_dir', type=str, default='~/Desktop/Mahdi/cars',
                       help='Path to cars directory')
    parser.add_argument('--output_dir', type=str, default='./experiments/stage1',
                       help='Output directory for experiments')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=100,
                       help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--latent_dim', type=int, default=256,
                       help='Latent code dimension')
    
    args = parser.parse_args()
    
    # Expand the ~ in the path
    cars_dir = os.path.expanduser(args.cars_dir)
    
    # Create config
    config = {
        'cars_dir': cars_dir,
        'output_dir': args.output_dir,
        'batch_size': args.batch_size,
        'num_epochs': args.num_epochs,
        'learning_rate': args.learning_rate,
        'latent_dim': args.latent_dim,
        'weight_decay': 1e-6,
        'latent_reg_weight': 1e-4,
        'num_workers': 4,
        'save_every': 10,
        'num_sdf_samples': 16384
    }
    
    # Setup device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create data loaders
    print("Creating data loaders...")
    data_loaders = create_cars_loaders(
        cars_dir=config['cars_dir'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        num_sdf_samples=config['num_sdf_samples']
    )
    
    print(f"Train samples: {len(data_loaders['train'].dataset)}")
    print(f"Val samples: {len(data_loaders['val'].dataset)}")
    
    # Create trainer
    trainer = Stage1Trainer(config, device)
    
    # Save configuration
    config_path = trainer.output_dir / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Start training
    trainer.train(
        train_loader=data_loaders['train'],
        val_loader=data_loaders['val'],
        num_epochs=config['num_epochs']
    )


if __name__ == "__main__":
    main()