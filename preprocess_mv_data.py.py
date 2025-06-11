#!/usr/bin/env python3
"""
preprocess_mv_data.py

Preprocess ShapeNet CAD models to generate multi-sweep training data.
This script generates the training data needed for Stage 2.

Usage:
    python preprocess_mv_data.py --shapenet_dir /path/to/shapenet --output_dir ./data/processed --category 02958343
"""

import os
import sys
import argparse
from pathlib import Path
import multiprocessing as mp
from functools import partial

import numpy as np
import torch
import trimesh
from tqdm import tqdm

# Import from your deep_sdf module
from deep_sdf.data import sample_sdf_points, farthest_point_sampling
from deep_sdf.pcgen_simulator import PCGenSimulator


class DataPreprocessor:
    """
    Preprocessor for generating multi-sweep training data from ShapeNet models.
    """
    
    def __init__(self, config):
        self.config = config
        
        # Create output directories
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        for split in ['train', 'val', 'test']:
            (self.output_dir / split).mkdir(exist_ok=True)
        
        print(f"Data Preprocessor initialized")
        print(f"Input: {config['shapenet_dir']}")
        print(f"Output: {config['output_dir']}")
        print(f"Category: {config['category']}")
    
    def find_shapenet_models(self):
        """Find all ShapeNet models for the specified category"""
        shapenet_dir = Path(self.config['shapenet_dir'])
        category = self.config['category']
        
        category_dir = shapenet_dir / category
        model_paths = []
        
        if not category_dir.exists():
            print(f"Error: Category directory not found: {category_dir}")
            return {'train': [], 'val': [], 'test': []}
        
        # Find all model files
        for instance_dir in category_dir.iterdir():
            if instance_dir.is_dir():
                # Look for mesh files in order of preference
                for mesh_name in ['model_normalized.obj', 'model.obj', 'model.ply']:
                    mesh_path = instance_dir / mesh_name
                    if mesh_path.exists():
                        model_paths.append(str(mesh_path))
                        break
        
        print(f"Found {len(model_paths)} models in category {category}")
        
        # Split into train/val/test
        np.random.seed(42)  # For reproducible splits
        indices = np.random.permutation(len(model_paths))
        
        train_split = int(0.7 * len(model_paths))
        val_split = int(0.85 * len(model_paths))
        
        splits = {
            'train': [model_paths[i] for i in indices[:train_split]],
            'val': [model_paths[i] for i in indices[train_split:val_split]],
            'test': [model_paths[i] for i in indices[val_split:]]
        }
        
        for split, paths in splits.items():
            print(f"  {split}: {len(paths)} models")
        
        return splits
    
    def normalize_mesh(self, mesh):
        """Normalize mesh to unit cube centered at origin"""
        # Center the mesh
        center = mesh.bounds.mean(axis=0)
        mesh.vertices -= center
        
        # Scale to fit in unit cube
        scale = 2.0 / np.max(mesh.bounds[1] - mesh.bounds[0])
        mesh.vertices *= scale
        
        return mesh
    
    def generate_gt_latent(self, mesh):
        """
        Generate ground truth latent code for a mesh.
        In practice, this would come from a pre-trained DeepSDF.
        For preprocessing, we use a simplified approach.
        """
        # Extract simple geometric features
        volume = mesh.volume if mesh.is_watertight else 0.0
        surface_area = mesh.area
        bbox_volume = np.prod(mesh.bounds[1] - mesh.bounds[0])
        
        # Create a deterministic but varied latent code based on mesh properties
        # This is simplified - in real implementation you'd use pre-trained DeepSDF
        np.random.seed(int(abs(volume * 1000 + surface_area * 100)))
        gt_latent = np.random.randn(256).astype(np.float32)
        
        # Add some structure based on geometry
        gt_latent[0] = volume
        gt_latent[1] = surface_area
        gt_latent[2] = bbox_volume
        
        return gt_latent
    
    def process_single_model(self, mesh_path, split):
        """Process a single ShapeNet model to generate multi-sweep data"""
        try:
            # Load and normalize mesh
            mesh = trimesh.load(mesh_path, force='mesh')
            mesh = self.normalize_mesh(mesh)
            
            # Generate ground truth latent code
            gt_latent = self.generate_gt_latent(mesh)
            
            # Generate multi-sweep point clouds using PCGen
            simulator = PCGenSimulator(mesh, self.config)
            multi_sweep_pcs = simulator.generate_multi_sweep(self.config['num_sweeps'])
            
            # Check if we have enough sweeps
            if len(multi_sweep_pcs) < self.config['num_sweeps']:
                print(f"Warning: Only generated {len(multi_sweep_pcs)} sweeps for {mesh_path}")
                return False
            
            # Apply FPS to each sweep
            processed_sweeps = []
            for sweep in multi_sweep_pcs:
                if len(sweep) == 0:
                    # Generate dummy sweep if empty
                    sweep = np.random.randn(100, 3) * 0.5
                
                # Convert to tensor and apply FPS
                sweep_tensor = torch.tensor(sweep, dtype=torch.float32).unsqueeze(0)
                fps_sweep = farthest_point_sampling(sweep_tensor, self.config['num_points_per_sweep'])
                processed_sweeps.append(fps_sweep.squeeze(0))
            
            # Ensure we have exactly num_sweeps
            while len(processed_sweeps) < self.config['num_sweeps']:
                if len(processed_sweeps) > 0:
                    processed_sweeps.append(processed_sweeps[-1].clone())
                else:
                    # Generate dummy sweep
                    dummy_sweep = torch.randn(self.config['num_points_per_sweep'], 3)
                    processed_sweeps.append(dummy_sweep)
            
            multi_sweep_pcs = torch.stack(processed_sweeps[:self.config['num_sweeps']])
            
            # Sample SDF points
            query_points, sdf_values = sample_sdf_points(
                mesh_path, self.config['num_sdf_samples']
            )
            
            # Get instance ID
            instance_id = Path(mesh_path).parent.name
            
            # Save preprocessed data
            output_path = self.output_dir / split / f'{instance_id}.npz'
            
            np.savez(
                output_path,
                multi_sweep_pcs=multi_sweep_pcs.numpy(),
                gt_latent=gt_latent,
                query_points=query_points.numpy(),
                sdf_values=sdf_values.numpy(),
                mesh_path=str(mesh_path)
            )
            
            return True
            
        except Exception as e:
            print(f"Error processing {mesh_path}: {e}")
            return False
    
    def process_split(self, mesh_paths, split):
        """Process all meshes in a split"""
        print(f"\nProcessing {split} split ({len(mesh_paths)} models)...")
        
        successful = 0
        failed = 0
        
        # Process each model
        for mesh_path in tqdm(mesh_paths, desc=f"Processing {split}"):
            if self.process_single_model(mesh_path, split):
                successful += 1
            else:
                failed += 1
        
        print(f"{split} split completed: {successful} successful, {failed} failed")
        return successful, failed
    
    def process_all(self):
        """Process all data splits"""
        # Find all models
        model_splits = self.find_shapenet_models()
        
        if not any(model_splits.values()):
            print("No models found! Check your ShapeNet directory and category.")
            return
        
        total_successful = 0
        total_failed = 0
        
        # Process each split
        for split, mesh_paths in model_splits.items():
            if len(mesh_paths) > 0:
                successful, failed = self.process_split(mesh_paths, split)
                total_successful += successful
                total_failed += failed
        
        print(f"\n✓ Preprocessing completed!")
        print(f"Total: {total_successful} successful, {total_failed} failed")
        print(f"Data saved to: {self.output_dir}")
        
        # Save dataset info
        info = {
            'total_samples': total_successful,
            'config': self.config,
            'splits': {split: len(paths) for split, paths in model_splits.items()}
        }
        
        info_path = self.output_dir / 'dataset_info.json'
        import json
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description='Preprocess ShapeNet for MV-DeepSDF')
    parser.add_argument('--shapenet_dir', type=str, required=True,
                       help='Path to ShapeNet directory')
    parser.add_argument('--output_dir', type=str, default='./data/processed',
                       help='Output directory for processed data')
    parser.add_argument('--category', type=str, default='02958343',
                       help='ShapeNet category ID (default: cars)')
    parser.add_argument('--num_sweeps', type=int, default=6,
                       help='Number of sweeps per instance')
    parser.add_argument('--num_points_per_sweep', type=int, default=256,
                       help='Number of points per sweep after FPS')
    parser.add_argument('--num_sdf_samples', type=int, default=10000,
                       help='Number of SDF samples per instance')
    
    args = parser.parse_args()
    
    # Create configuration
    config = {
        'shapenet_dir': args.shapenet_dir,
        'output_dir': args.output_dir,
        'category': args.category,
        'num_sweeps': args.num_sweeps,
        'num_points_per_sweep': args.num_points_per_sweep,
        'num_sdf_samples': args.num_sdf_samples,
        'noise_std': 0.01,  # LiDAR noise
    }
    
    # Validate inputs
    if not os.path.exists(config['shapenet_dir']):
        print(f"Error: ShapeNet directory not found: {config['shapenet_dir']}")
        return
    
    # Create preprocessor and run
    preprocessor = DataPreprocessor(config)
    preprocessor.process_all()


if __name__ == "__main__":
    main()