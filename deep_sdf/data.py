#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import glob
import logging
import numpy as np
import os
import random
import torch
import torch.utils.data

import deep_sdf.workspace as ws


def get_instance_filenames(data_source, split):
    npzfiles = []
    for dataset in split:
        for class_name in split[dataset]:
            for instance_name in split[dataset][class_name]:
                instance_filename = os.path.join(
                    dataset, class_name, instance_name + ".npz"
                )
                if not os.path.isfile(
                    os.path.join(data_source, ws.sdf_samples_subdir, instance_filename)
                ):
                    # raise RuntimeError(
                    #     'Requested non-existent file "' + instance_filename + "'"
                    # )
                    logging.warning(
                        "Requested non-existent file '{}'".format(instance_filename)
                    )
                npzfiles += [instance_filename]
    return npzfiles


class NoMeshFileError(RuntimeError):
    """Raised when a mesh file is not found in a shape directory"""

    pass


class MultipleMeshFileError(RuntimeError):
    """"Raised when a there a multiple mesh files in a shape directory"""

    pass


def find_mesh_in_directory(shape_dir):
    mesh_filenames = list(glob.iglob(shape_dir + "/**/*.obj")) + list(
        glob.iglob(shape_dir + "/*.obj")
    )
    if len(mesh_filenames) == 0:
        raise NoMeshFileError()
    elif len(mesh_filenames) > 1:
        raise MultipleMeshFileError()
    return mesh_filenames[0]


def remove_nans(tensor):
    tensor_nan = torch.isnan(tensor[:, 3])
    return tensor[~tensor_nan, :]


def read_sdf_samples_into_ram(filename):
    npz = np.load(filename)
    pos_tensor = torch.from_numpy(npz["pos"])
    neg_tensor = torch.from_numpy(npz["neg"])

    return [pos_tensor, neg_tensor]


def unpack_sdf_samples(filename, subsample=None):
    npz = np.load(filename)
    if subsample is None:
        return npz
    pos_tensor = remove_nans(torch.from_numpy(npz["pos"]))
    neg_tensor = remove_nans(torch.from_numpy(npz["neg"]))

    # split the sample into half
    half = int(subsample / 2)

    random_pos = (torch.rand(half) * pos_tensor.shape[0]).long()
    random_neg = (torch.rand(half) * neg_tensor.shape[0]).long()

    sample_pos = torch.index_select(pos_tensor, 0, random_pos)
    sample_neg = torch.index_select(neg_tensor, 0, random_neg)

    samples = torch.cat([sample_pos, sample_neg], 0)

    return samples


def unpack_sdf_samples_from_ram(data, subsample=None):
    if subsample is None:
        return data
    pos_tensor = data[0]
    neg_tensor = data[1]

    # split the sample into half
    half = int(subsample / 2)

    pos_size = pos_tensor.shape[0]
    neg_size = neg_tensor.shape[0]

    pos_start_ind = random.randint(0, pos_size - half)
    sample_pos = pos_tensor[pos_start_ind : (pos_start_ind + half)]

    if neg_size <= half:
        random_neg = (torch.rand(half) * neg_tensor.shape[0]).long()
        sample_neg = torch.index_select(neg_tensor, 0, random_neg)
    else:
        neg_start_ind = random.randint(0, neg_size - half)
        sample_neg = neg_tensor[neg_start_ind : (neg_start_ind + half)]

    samples = torch.cat([sample_pos, sample_neg], 0)

    return samples


class SDFSamples(torch.utils.data.Dataset):
    def __init__(
        self,
        data_source,
        split,
        subsample,
        load_ram=False,
        print_filename=False,
        num_files=1000000,
    ):
        self.subsample = subsample

        self.data_source = data_source
        self.npyfiles = get_instance_filenames(data_source, split)

        logging.debug(
            "using "
            + str(len(self.npyfiles))
            + " shapes from data source "
            + data_source
        )

        self.load_ram = load_ram

        if load_ram:
            self.loaded_data = []
            for f in self.npyfiles:
                filename = os.path.join(self.data_source, ws.sdf_samples_subdir, f)
                npz = np.load(filename)
                pos_tensor = remove_nans(torch.from_numpy(npz["pos"]))
                neg_tensor = remove_nans(torch.from_numpy(npz["neg"]))
                self.loaded_data.append(
                    [
                        pos_tensor[torch.randperm(pos_tensor.shape[0])],
                        neg_tensor[torch.randperm(neg_tensor.shape[0])],
                    ]
                )

    def __len__(self):
        return len(self.npyfiles)

    def __getitem__(self, idx):
        filename = os.path.join(
            self.data_source, ws.sdf_samples_subdir, self.npyfiles[idx]
        )
        if self.load_ram:
            return (
                unpack_sdf_samples_from_ram(self.loaded_data[idx], self.subsample),
                idx,
            )
        else:
            return unpack_sdf_samples(filename, self.subsample), idx


# ============================================================================
# MV-DEEPSDF ADDITIONS - Added for multi-view DeepSDF functionality
# ============================================================================

from torch.utils.data import DataLoader
from pathlib import Path
import trimesh


def farthest_point_sampling(points, num_samples):
    """
    Sample points using Farthest Point Sampling (FPS).
    
    Args:
        points: (B, N, 3) point cloud
        num_samples: number of points to sample
    
    Returns:
        sampled_points: (B, num_samples, 3)
    """
    B, N, _ = points.shape
    device = points.device
    
    if num_samples >= N:
        # If we need more points than available, repeat the last point
        repeat_factor = (num_samples + N - 1) // N
        points_repeated = points.repeat(1, repeat_factor, 1)[:, :num_samples, :]
        return points_repeated
    
    # Initialize
    sampled_indices = torch.zeros(B, num_samples, dtype=torch.long, device=device)
    distances = torch.full((B, N), float('inf'), device=device)
    
    # Start with random point
    sampled_indices[:, 0] = torch.randint(0, N, (B,), device=device)
    
    # FPS algorithm
    for i in range(1, num_samples):
        # Get current points
        current_points = points[torch.arange(B), sampled_indices[:, i-1]].unsqueeze(1)
        
        # Calculate distances
        dists = torch.norm(points - current_points, dim=2)
        distances = torch.min(distances, dists)
        
        # Mark selected points as unavailable
        for b in range(B):
            distances[b, sampled_indices[b, :i]] = -float('inf')
        
        # Select farthest point
        sampled_indices[:, i] = torch.argmax(distances, dim=1)
    
    # Gather sampled points
    batch_indices = torch.arange(B, device=device).unsqueeze(1).expand(-1, num_samples)
    sampled_points = points[batch_indices, sampled_indices]
    
    return sampled_points


def sample_sdf_points(mesh_path, num_points=100000):
    """
    Sample points and their SDF values from a mesh.
    
    Args:
        mesh_path: path to mesh file
        num_points: total number of points to sample
    
    Returns:
        points: (num_points, 3) sampled points
        sdf_values: (num_points,) SDF values
    """
    try:
        mesh = trimesh.load(mesh_path, force='mesh')
        
        # Sample surface points
        surface_points, _ = trimesh.sample.sample_surface(mesh, num_points // 2)
        
        # Sample random points in bounding box
        bbox = mesh.bounds
        bbox_size = np.max(bbox[1] - bbox[0])
        center = np.mean(bbox, axis=0)
        
        # Extend bounding box slightly
        extended_bbox = np.array([
            center - bbox_size * 0.75,
            center + bbox_size * 0.75
        ])
        
        random_points = np.random.uniform(
            extended_bbox[0], extended_bbox[1], 
            (num_points // 2, 3)
        )
        
        # Combine points
        all_points = np.vstack([surface_points, random_points])
        
        # Calculate SDF values
        surface_sdf = np.zeros(len(surface_points))  # Surface points have SDF ≈ 0
        
        # For random points, calculate distance to surface
        try:
            closest_points, distances, _ = trimesh.proximity.closest_point(mesh, random_points)
            contains = mesh.contains(random_points)
            random_sdf = distances.copy()
            random_sdf[contains] *= -1  # Inside points get negative SDF
            
            sdf_values = np.concatenate([surface_sdf, random_sdf])
        except:
            # Fallback if SDF calculation fails
            sdf_values = np.zeros(len(all_points))
        
        return torch.tensor(all_points, dtype=torch.float32), torch.tensor(sdf_values, dtype=torch.float32)
    
    except Exception as e:
        print(f"Error loading mesh {mesh_path}: {e}")
        # Return dummy data
        return torch.randn(num_points, 3), torch.randn(num_points)


class MultiSweepDataset(torch.utils.data.Dataset):
    """
    Dataset for multi-sweep point clouds.
    This is what you'll use for Stage 2 training.
    """
    
    def __init__(self, data_dir, split='train', num_sweeps=6, num_points=256, num_sdf_samples=10000):
        self.data_dir = Path(data_dir)
        self.split = split
        self.num_sweeps = num_sweeps
        self.num_points = num_points
        self.num_sdf_samples = num_sdf_samples
        
        # Find data files
        self.file_list = self._load_file_list()
        print(f"Loaded {len(self.file_list)} samples for {split} split")
    
    def _load_file_list(self):
        """Find all data files for this split"""
        split_dir = self.data_dir / self.split
        if not split_dir.exists():
            split_dir = self.data_dir  # Fallback to main directory
        
        file_list = []
        for pattern in ['*.npz', '*.pt', '*.pth']:
            file_list.extend(glob.glob(str(split_dir / pattern)))
        
        return sorted(file_list)
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        try:
            data_path = self.file_list[idx]
            
            # Load data
            if data_path.endswith('.npz'):
                data = np.load(data_path)
                multi_sweep_pcs = torch.tensor(data['multi_sweep_pcs'], dtype=torch.float32)
                gt_latent = torch.tensor(data['gt_latent'], dtype=torch.float32)
                query_points = torch.tensor(data['query_points'], dtype=torch.float32)
                sdf_values = torch.tensor(data['sdf_values'], dtype=torch.float32)
            else:
                data = torch.load(data_path, map_location='cpu')
                multi_sweep_pcs = data['multi_sweep_pcs']
                gt_latent = data['gt_latent']
                query_points = data['query_points']
                sdf_values = data['sdf_values']
            
            # Apply FPS to each sweep if needed
            if multi_sweep_pcs.shape[1] != self.num_points:
                fps_sweeps = []
                for i in range(len(multi_sweep_pcs)):
                    sweep = multi_sweep_pcs[i].unsqueeze(0)
                    fps_sweep = farthest_point_sampling(sweep, self.num_points).squeeze(0)
                    fps_sweeps.append(fps_sweep)
                multi_sweep_pcs = torch.stack(fps_sweeps)
            
            # Ensure correct number of sweeps
            if len(multi_sweep_pcs) < self.num_sweeps:
                # Pad with last sweep
                last_sweep = multi_sweep_pcs[-1]
                padding = [last_sweep] * (self.num_sweeps - len(multi_sweep_pcs))
                multi_sweep_pcs = torch.cat([multi_sweep_pcs] + padding, dim=0)
            elif len(multi_sweep_pcs) > self.num_sweeps:
                multi_sweep_pcs = multi_sweep_pcs[:self.num_sweeps]
            
            # Subsample SDF points if needed
            if len(query_points) > self.num_sdf_samples:
                indices = torch.randperm(len(query_points))[:self.num_sdf_samples]
                query_points = query_points[indices]
                sdf_values = sdf_values[indices]
            
            instance_id = os.path.splitext(os.path.basename(data_path))[0]
            
            return {
                'multi_sweep_pcs': multi_sweep_pcs,    # (num_sweeps, num_points, 3)
                'gt_latent': gt_latent,                # (256,)
                'query_points': query_points,          # (num_sdf_samples, 3)
                'sdf_values': sdf_values,              # (num_sdf_samples,)
                'instance_id': instance_id
            }
            
        except Exception as e:
            print(f"Error loading {self.file_list[idx]}: {e}")
            # Return dummy data
            return {
                'multi_sweep_pcs': torch.randn(self.num_sweeps, self.num_points, 3),
                'gt_latent': torch.randn(256),
                'query_points': torch.randn(self.num_sdf_samples, 3),
                'sdf_values': torch.randn(self.num_sdf_samples),
                'instance_id': f'dummy_{idx}'
            }


class ShapeNetDatasetMV(torch.utils.data.Dataset):
    """
    Dataset for ShapeNet meshes for MV-DeepSDF.
    This is what you'll use for Stage 1 training.
    """
    
    def __init__(self, shapenet_dir, category='02958343', split='train', num_sdf_samples=16384):
        self.shapenet_dir = Path(shapenet_dir)
        self.category = category
        self.split = split
        self.num_sdf_samples = num_sdf_samples
        
        # Find mesh files
        self.mesh_list = self._load_mesh_list()
        print(f"Loaded {len(self.mesh_list)} meshes for category {category}, split {split}")
    
    def _load_mesh_list(self):
        """Find all mesh files for this category and split"""
        category_dir = self.shapenet_dir / self.category
        mesh_list = []
        
        if category_dir.exists():
            for instance_dir in category_dir.iterdir():
                if instance_dir.is_dir():
                    # Look for mesh files
                    for mesh_file in ['model_normalized.obj', 'model.obj', 'model.ply']:
                        mesh_path = instance_dir / mesh_file
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


def create_mv_data_loaders(data_dir, batch_size=1, num_workers=4, **kwargs):
    """Create train/val/test data loaders for multi-sweep data"""
    datasets = {}
    data_loaders = {}
    
    for split in ['train', 'val', 'test']:
        datasets[split] = MultiSweepDataset(data_dir=data_dir, split=split, **kwargs)
        data_loaders[split] = DataLoader(
            datasets[split],
            batch_size=batch_size,
            shuffle=(split == 'train'),
            num_workers=num_workers,
            pin_memory=True,
            drop_last=(split == 'train')
        )
    
    return data_loaders


def create_shapenet_mv_loaders(shapenet_dir, category='02958343', batch_size=32, num_workers=4, **kwargs):
    """Create train/val/test data loaders for ShapeNet data"""
    datasets = {}
    data_loaders = {}
    
    for split in ['train', 'val', 'test']:
        datasets[split] = ShapeNetDatasetMV(
            shapenet_dir=shapenet_dir, 
            category=category, 
            split=split, 
            **kwargs
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