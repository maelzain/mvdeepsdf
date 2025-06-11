#!/usr/bin/env python3
"""
train_mv_stage2.py

STAGE 2 TRAINING: Train the MV-DeepSDF fusion network.
This trains the multi-view fusion network with frozen SDF decoder.

Usage:
    python train_mv_stage2.py --data_dir /path/to/multi_sweep_data --stage1_checkpoint ./experiments/stage1/checkpoints/best.pth --output_dir ./experiments/stage2
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
from deep_sdf.data import create_mv_data_loaders


class Stage2Trainer:
    """Trainer for Stage 2: MV-DeepSDF fusion network training"""
    
    def __init__(self, config, device):
        self.config = config
        self.device = device
        
        # Create model
        self.model = MVDeepSDF(latent_dim=config['latent_dim']).to(device)
        
        # Load pre-trained Stage 1 SDF decoder
        if 'stage1_checkpoint' in config:
            self.load_stage1_checkpoint(config['stage1_checkpoint'])
        
        # Freeze SDF decoder
        for param in self.model.sdf_decoder.parameters():
            param.requires_grad = False
        
        # Optionally freeze DeepSDF encoder
        if config.get('freeze_deepsdf_encoder', True):
            for param in self.model.fusion_network.deepsdf_encoder.parameters():
                param.requires_grad = False
        
        # Loss functions
        self.latent_criterion = nn.MSELoss()
        self.sdf_criterion = nn.L1Loss()
        
        # Optimizer (only trainable parameters)
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = optim.Adam(
            trainable_params,
            lr=config['learning_rate'],
            weight_decay=config.get('weight_decay', 1e-6)
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=config.get('lr_step_size', 10),
            gamma=config.get('lr_gamma', 0.5)
        )
        
        # Setup directories
        self.output_dir = Path(config['output_dir'])
        self.checkpoints_dir = self.output_dir / 'checkpoints'
        self.logs_dir = self.output_dir / 'logs'
        self.reconstructions_dir = self.output_dir / 'reconstructions'
        
        for dir_path in [self.output_dir, self.checkpoints_dir, self.logs_dir, self.reconstructions_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Tensorboard writer
        self.writer = SummaryWriter(self.logs_dir / 'stage2')
        
        # Training state
        self.epoch = 0
        self.best_loss = float('inf')
        
        print(f"Stage 2 Trainer initialized on {device}")
        print(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")
        print(f"Output directory: {self.output_dir}")
    
    def load_stage1_checkpoint(self, checkpoint_path):
        """Load pre-trained Stage 1 SDF decoder"""
        if not os.path.exists(checkpoint_path):
            print(f"Warning: Stage 1 checkpoint not found: {checkpoint_path}")
            return
        
        print(f"Loading Stage 1 checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Extract SDF decoder weights
        model_state = checkpoint['model_state_dict']
        sdf_decoder_state = {k.replace('sdf_decoder.', ''): v 
                           for k, v in model_state.items() 
                           if k.startswith('sdf_decoder.')}
        
        # Load into current model
        self.model.sdf_decoder.load_state_dict(sdf_decoder_state, strict=False)
        print("✓ Stage 1 SDF decoder weights loaded")
    
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
        
        # Save epoch checkpoint
        if self.epoch % self.config.get('save_every', 5) == 0:
            epoch_path = self.checkpoints_dir / f'epoch_{self.epoch:04d}.pth'
            torch.save(checkpoint, epoch_path)
    
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        # Keep SDF decoder in eval mode since it's frozen
        self.model.sdf_decoder.eval()
        
        total_loss = 0.0
        total_latent_loss = 0.0
        total_sdf_loss = 0.0
        num_batches = len(train_loader)
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {self.epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Get data
            multi_sweep_pcs = batch['multi_sweep_pcs'].to(self.device)  # (B, num_sweeps, N, 3)
            gt_latent = batch['gt_latent'].to(self.device)              # (B, 256)
            query_points = batch['query_points'].to(self.device)        # (B, M, 3)
            sdf_values = batch['sdf_values'].to(self.device)           # (B, M)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            results = self.model(multi_sweep_pcs, query_points, freeze_deepsdf=True)
            predicted_latent = results['predicted_latent']  # (B, 256)
            predicted_sdf = results['sdf_values'].squeeze(-1)  # (B, M)
            
            # Compute losses
            latent_loss = self.latent_criterion(predicted_latent, gt_latent)
            sdf_loss = self.sdf_criterion(predicted_sdf, sdf_values)
            
            # Combined loss (weight SDF loss lower as in paper)
            sdf_weight = self.config.get('sdf_loss_weight', 0.1)
            total_loss_batch = latent_loss + sdf_weight * sdf_loss
            
            # Backward pass
            total_loss_batch.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad], 
                max_norm=1.0
            )
            
            self.optimizer.step()
            
            # Update metrics
            total_loss += total_loss_batch.item()
            total_latent_loss += latent_loss.item()
            total_sdf_loss += sdf_loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'Total': f'{total_loss_batch.item():.6f}',
                'Latent': f'{latent_loss.item():.6f}',
                'SDF': f'{sdf_loss.item():.6f}',
                'LR': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
            })
            
            # Log to tensorboard
            global_step = self.epoch * num_batches + batch_idx
            self.writer.add_scalar('Train/TotalLoss', total_loss_batch.item(), global_step)
            self.writer.add_scalar('Train/LatentLoss', latent_loss.item(), global_step)
            self.writer.add_scalar('Train/SDFLoss', sdf_loss.item(), global_step)
        
        avg_loss = total_loss / num_batches
        avg_latent_loss = total_latent_loss / num_batches
        avg_sdf_loss = total_sdf_loss / num_batches
        
        return avg_loss, avg_latent_loss, avg_sdf_loss
    
    def validate_epoch(self, val_loader):
        """Validate for one epoch"""
        self.model.eval()
        
        total_loss = 0.0
        total_latent_loss = 0.0
        total_sdf_loss = 0.0
        num_batches = len(val_loader)
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                multi_sweep_pcs = batch['multi_sweep_pcs'].to(self.device)
                gt_latent = batch['gt_latent'].to(self.device)
                query_points = batch['query_points'].to(self.device)
                sdf_values = batch['sdf_values'].to(self.device)
                
                # Forward pass
                results = self.model(multi_sweep_pcs, query_points, freeze_deepsdf=True)
                predicted_latent = results['predicted_latent']
                predicted_sdf = results['sdf_values'].squeeze(-1)
                
                # Compute losses
                latent_loss = self.latent_criterion(predicted_latent, gt_latent)
                sdf_loss = self.sdf_criterion(predicted_sdf, sdf_values)
                sdf_weight = self.config.get('sdf_loss_weight', 0.1)
                total_loss_batch = latent_loss + sdf_weight * sdf_loss
                
                total_loss += total_loss_batch.item()
                total_latent_loss += latent_loss.item()
                total_sdf_loss += sdf_loss.item()
        
        avg_loss = total_loss / num_batches
        avg_latent_loss = total_latent_loss / num_batches
        avg_sdf_loss = total_sdf_loss / num_batches
        
        return avg_loss, avg_latent_loss, avg_sdf_loss
    
    def reconstruct_samples(self, val_loader, num_samples=3):
        """Reconstruct a few samples for visualization"""
        self.model.eval()
        reconstructed_samples = []
        
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                if i >= num_samples:
                    break
                
                multi_sweep_pcs = batch['multi_sweep_pcs'].to(self.device)
                instance_id = batch['instance_id'][0]  # First sample in batch
                
                # Take first sample
                sample_sweeps = multi_sweep_pcs[:1]  # (1, num_sweeps, N, 3)
                
                # Predict latent code
                results = self.model(sample_sweeps, freeze_deepsdf=True)
                predicted_latent = results['predicted_latent']
                
                # Extract mesh
                try:
                    mesh = self.model.extract_mesh(predicted_latent, resolution=64)
                    
                    # Save mesh
                    mesh_path = self.reconstructions_dir / f'epoch_{self.epoch:04d}_{instance_id}.ply'
                    mesh.export(str(mesh_path))
                    
                    reconstructed_samples.append({
                        'instance_id': instance_id,
                        'mesh_path': str(mesh_path),
                        'num_vertices': len(mesh.vertices),
                        'num_faces': len(mesh.faces)
                    })
                    
                except Exception as e:
                    print(f"Failed to reconstruct {instance_id}: {e}")
        
        return reconstructed_samples
    
    def train(self, train_loader, val_loader, num_epochs):
        """Main training loop"""
        print(f"Starting Stage 2 training for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            self.epoch = epoch + 1
            
            # Train and validate
            train_loss, train_latent_loss, train_sdf_loss = self.train_epoch(train_loader)
            val_loss, val_latent_loss, val_sdf_loss = self.validate_epoch(val_loader)
            
            # Update learning rate
            self.scheduler.step()
            
            # Log epoch results
            self.writer.add_scalar('Epoch/TrainLoss', train_loss, self.epoch)
            self.writer.add_scalar('Epoch/ValLoss', val_loss, self.epoch)
            
            print(f"Epoch {self.epoch:4d} | "
                  f"Train: {train_loss:.6f} (L:{train_latent_loss:.6f}, S:{train_sdf_loss:.6f}) | "
                  f"Val: {val_loss:.6f} (L:{val_latent_loss:.6f}, S:{val_sdf_loss:.6f})")
            
            # Save checkpoint
            is_best = val_loss < self.best_loss
            if is_best:
                self.best_loss = val_loss
            
            self.save_checkpoint(is_best)
            
            # Reconstruct samples for visualization
            if self.epoch % self.config.get('reconstruct_every', 5) == 0:
                print("Reconstructing sample meshes...")
                reconstructed = self.reconstruct_samples(val_loader)
                for sample in reconstructed:
                    print(f"  {sample['instance_id']}: {sample['num_vertices']} vertices")
        
        print("✓ Stage 2 training completed!")
        self.writer.close()


def main():
    parser = argparse.ArgumentParser(description='MV-DeepSDF Stage 2 Training')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to preprocessed multi-sweep data')
    parser.add_argument('--stage1_checkpoint', type=str, required=True,
                       help='Path to Stage 1 checkpoint')
    parser.add_argument('--output_dir', type=str, default='./experiments/stage2',
                       help='Output directory for experiments')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=20,
                       help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-5,
                       help='Learning rate')
    parser.add_argument('--latent_dim', type=int, default=256,
                       help='Latent code dimension')
    
    args = parser.parse_args()
    
    # Create config
    config = {
        'data_dir': args.data_dir,
        'stage1_checkpoint': args.stage1_checkpoint,
        'output_dir': args.output_dir,
        'batch_size': args.batch_size,
        'num_epochs': args.num_epochs,
        'learning_rate': args.learning_rate,
        'latent_dim': args.latent_dim,
        'weight_decay': 1e-6,
        'sdf_loss_weight': 0.1,
        'lr_step_size': 10,
        'lr_gamma': 0.5,
        'num_workers': 4,
        'save_every': 5,
        'reconstruct_every': 5,
        'num_sweeps': 6,
        'num_points': 256,
        'num_sdf_samples': 10000,
        'freeze_deepsdf_encoder': True
    }
    
    # Setup device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create data loaders
    print("Creating data loaders...")
    data_loaders = create_mv_data_loaders(
        data_dir=config['data_dir'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        num_sweeps=config['num_sweeps'],
        num_points=config['num_points'],
        num_sdf_samples=config['num_sdf_samples']
    )
    
    print(f"Train samples: {len(data_loaders['train'].dataset)}")
    print(f"Val samples: {len(data_loaders['val'].dataset)}")
    
    # Create trainer
    trainer = Stage2Trainer(config, device)
    
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