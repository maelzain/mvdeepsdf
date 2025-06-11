#!/usr/bin/env python3
"""
evaluate_mv.py

Evaluate trained MV-DeepSDF model on test data.
Computes metrics like Asymmetric Chamfer Distance (ACD) and Recall.

Usage:
    python evaluate_mv.py --checkpoint ./experiments/stage2/checkpoints/best.pth --data_dir ./data/processed --output_dir ./results
"""

import os
import sys
import argparse
import json
from pathlib import Path

import torch
import numpy as np
import trimesh
from tqdm import tqdm
from scipy.spatial.distance import cdist

# Import from your deep_sdf module
from deep_sdf.mv_deepsdf import MVDeepSDF
from deep_sdf.data import create_mv_data_loaders


def asymmetric_chamfer_distance(pred_points, gt_points):
    """
    Compute Asymmetric Chamfer Distance (ACD).
    
    Args:
        pred_points: Predicted points (N, 3)
        gt_points: Ground truth points (M, 3)
        
    Returns:
        ACD value (float)
    """
    if len(pred_points) == 0 or len(gt_points) == 0:
        return float('inf')
    
    # Compute pairwise distances
    distances = cdist(gt_points, pred_points)
    
    # Find minimum distance for each ground truth point
    min_distances = np.min(distances, axis=1)
    
    # Compute ACD (mean squared distance)
    acd = np.mean(min_distances ** 2)
    
    return acd


def compute_recall(pred_points, gt_points, threshold=0.1):
    """
    Compute recall metric.
    
    Args:
        pred_points: Predicted points (N, 3)
        gt_points: Ground truth points (M, 3)
        threshold: Distance threshold
        
    Returns:
        Recall value (float)
    """
    if len(pred_points) == 0 or len(gt_points) == 0:
        return 0.0
    
    # Compute pairwise distances
    distances = cdist(gt_points, pred_points)
    
    # Find minimum distance for each ground truth point
    min_distances = np.min(distances, axis=1)
    
    # Count points within threshold
    recall = np.mean(min_distances < threshold)
    
    return recall


class MVDeepSDFEvaluator:
    """Evaluator for MV-DeepSDF model"""
    
    def __init__(self, config, device):
        self.config = config
        self.device = device
        
        # Load model
        self.model = MVDeepSDF(latent_dim=config['latent_dim']).to(device)
        self.load_checkpoint(config['checkpoint'])
        self.model.eval()
        
        # Setup output directory
        self.output_dir = Path(config['output_dir'])
        self.meshes_dir = self.output_dir / 'reconstructed_meshes'
        self.results_dir = self.output_dir / 'results'
        
        for dir_path in [self.output_dir, self.meshes_dir, self.results_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        print(f"Evaluator initialized")
        print(f"Model loaded from: {config['checkpoint']}")
        print(f"Output directory: {self.output_dir}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        print(f"✓ Model loaded from epoch {checkpoint.get('epoch', 'unknown')}")
    
    def reconstruct_mesh(self, multi_sweep_pcs, instance_id):
        """Reconstruct mesh from multi-sweep point clouds"""
        with torch.no_grad():
            # Predict latent code
            results = self.model(multi_sweep_pcs, freeze_deepsdf=True)
            predicted_latent = results['predicted_latent']
            
            # Extract mesh
            mesh = self.model.extract_mesh(
                predicted_latent, 
                resolution=self.config.get('mesh_resolution', 64)
            )
            
            # Save mesh
            if len(mesh.vertices) > 0:
                mesh_path = self.meshes_dir / f'{instance_id}.ply'
                mesh.export(str(mesh_path))
                return mesh, str(mesh_path)
            else:
                print(f"Warning: Empty mesh for {instance_id}")
                return None, None
    
    def evaluate_sample(self, batch):
        """Evaluate a single sample"""
        multi_sweep_pcs = batch['multi_sweep_pcs'].to(self.device)  # (B, num_sweeps, N, 3)
        instance_id = batch['instance_id'][0]  # First sample in batch
        
        # Take first sample in batch
        sample_sweeps = multi_sweep_pcs[:1]  # (1, num_sweeps, N, 3)
        
        # Reconstruct mesh
        mesh, mesh_path = self.reconstruct_mesh(sample_sweeps, instance_id)
        
        if mesh is None:
            return {
                'instance_id': instance_id,
                'acd': float('inf'),
                'recall': 0.0,
                'mesh_path': None,
                'num_vertices': 0,
                'num_faces': 0,
                'success': False
            }
        
        # Sample points from reconstructed mesh
        try:
            pred_points, _ = trimesh.sample.sample_surface(mesh, 30000)
        except:
            pred_points = mesh.vertices
        
        # Get ground truth points (stacked multi-sweep)
        gt_sweep = multi_sweep_pcs[0].cpu().numpy()  # (num_sweeps, N, 3)
        gt_points = gt_sweep.reshape(-1, 3)  # (num_sweeps * N, 3)
        
        # Compute metrics
        acd = asymmetric_chamfer_distance(pred_points, gt_points)
        recall = compute_recall(pred_points, gt_points, threshold=0.1)
        
        return {
            'instance_id': instance_id,
            'acd': acd,
            'recall': recall,
            'mesh_path': mesh_path,
            'num_vertices': len(mesh.vertices),
            'num_faces': len(mesh.faces),
            'success': True
        }
    
    def evaluate_dataset(self, test_loader):
        """Evaluate entire test dataset"""
        print("Evaluating test dataset...")
        
        results = []
        successful_reconstructions = 0
        
        # Evaluate each sample
        for batch in tqdm(test_loader, desc="Evaluating"):
            try:
                result = self.evaluate_sample(batch)
                results.append(result)
                
                if result['success']:
                    successful_reconstructions += 1
                
            except Exception as e:
                print(f"Error evaluating {batch['instance_id'][0]}: {e}")
                results.append({
                    'instance_id': batch['instance_id'][0],
                    'acd': float('inf'),
                    'recall': 0.0,
                    'mesh_path': None,
                    'num_vertices': 0,
                    'num_faces': 0,
                    'success': False
                })
        
        # Compute overall statistics
        successful_results = [r for r in results if r['success']]
        
        if len(successful_results) > 0:
            acd_values = [r['acd'] for r in successful_results if r['acd'] != float('inf')]
            recall_values = [r['recall'] for r in successful_results]
            
            stats = {
                'total_samples': len(results),
                'successful_reconstructions': successful_reconstructions,
                'success_rate': successful_reconstructions / len(results),
                'acd_mean': np.mean(acd_values) if acd_values else float('inf'),
                'acd_median': np.median(acd_values) if acd_values else float('inf'),
                'acd_std': np.std(acd_values) if acd_values else 0.0,
                'recall_mean': np.mean(recall_values) if recall_values else 0.0,
                'recall_std': np.std(recall_values) if recall_values else 0.0,
                'config': self.config
            }
        else:
            stats = {
                'total_samples': len(results),
                'successful_reconstructions': 0,
                'success_rate': 0.0,
                'acd_mean': float('inf'),
                'acd_median': float('inf'),
                'acd_std': 0.0,
                'recall_mean': 0.0,
                'recall_std': 0.0,
                'config': self.config
            }
        
        # Save detailed results
        detailed_results_path = self.results_dir / 'detailed_results.json'
        with open(detailed_results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save summary statistics
        stats_path = self.results_dir / 'evaluation_stats.json'
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        # Print results
        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        print(f"Total samples: {stats['total_samples']}")
        print(f"Successful reconstructions: {stats['successful_reconstructions']}")
        print(f"Success rate: {stats['success_rate']:.2%}")
        print(f"ACD Mean: {stats['acd_mean']:.6f}")
        print(f"ACD Median: {stats['acd_median']:.6f}")
        print(f"Recall Mean: {stats['recall_mean']:.4f}")
        print(f"Recall Std: {stats['recall_std']:.4f}")
        print(f"\nResults saved to: {self.results_dir}")
        
        return stats


def main():
    parser = argparse.ArgumentParser(description='Evaluate MV-DeepSDF Model')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to test data directory')
    parser.add_argument('--output_dir', type=str, default='./results',
                       help='Output directory for results')
    parser.add_argument('--latent_dim', type=int, default=256,
                       help='Latent code dimension')
    parser.add_argument('--mesh_resolution', type=int, default=64,
                       help='Mesh extraction resolution')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size')
    
    args = parser.parse_args()
    
    # Create configuration
    config = {
        'checkpoint': args.checkpoint,
        'data_dir': args.data_dir,
        'output_dir': args.output_dir,
        'latent_dim': args.latent_dim,
        'mesh_resolution': args.mesh_resolution,
        'batch_size': args.batch_size,
        'num_workers': 4,
        'num_sweeps': 6,
        'num_points': 256,
        'num_sdf_samples': 10000
    }
    
    # Setup device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create test data loader
    print("Loading test data...")
    data_loaders = create_mv_data_loaders(
        data_dir=config['data_dir'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        num_sweeps=config['num_sweeps'],
        num_points=config['num_points'],
        num_sdf_samples=config['num_sdf_samples']
    )
    
    test_loader = data_loaders['test']
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Create evaluator
    evaluator = MVDeepSDFEvaluator(config, device)
    
    # Run evaluation
    stats = evaluator.evaluate_dataset(test_loader)


if __name__ == "__main__":
    main()