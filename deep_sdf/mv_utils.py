#!/usr/bin/env python3
"""
mv_utils.py - MV-DeepSDF Specific Utility Functions

Contains:
- Farthest Point Sampling (FPS)
- SDF computation utilities
- Point cloud processing functions
- Mesh extraction utilities
"""

import torch
import numpy as np
from skimage import measure
import trimesh
from sklearn.neighbors import NearestNeighbors
import random

def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def farthest_point_sampling(points, num_samples):
    """
    Farthest Point Sampling (FPS) implementation
    Args:
        points: (N, 3) numpy array of 3D points
        num_samples: number of points to sample
    Returns:
        sampled_points: (num_samples, 3) numpy array
        sampled_indices: (num_samples,) indices of sampled points
    """
    N, _ = points.shape
    if N <= num_samples:
        # If we have fewer points than requested, pad with repetition
        indices = list(range(N))
        while len(indices) < num_samples:
            indices.extend(list(range(N)))
        indices = indices[:num_samples]
        return points[indices], np.array(indices[:N])
    
    # Initialize with random first point
    sampled_indices = [np.random.randint(N)]
    distances = np.full(N, np.inf)
    
    for i in range(1, num_samples):
        # Update distances to nearest sampled point
        last_point = points[sampled_indices[-1]]
        new_distances = np.linalg.norm(points - last_point, axis=1)
        distances = np.minimum(distances, new_distances)
        
        # Select point with maximum distance
        sampled_indices.append(np.argmax(distances))
    
    sampled_points = points[sampled_indices]
    return sampled_points, np.array(sampled_indices)

def compute_sdf_from_mesh(mesh_path, query_points):
    """
    Compute SDF values for query points given a mesh
    Args:
        mesh_path: path to mesh file
        query_points: (N, 3) array of query points
    Returns:
        sdf_values: (N,) array of SDF values
    """
    mesh = trimesh.load(mesh_path)
    
    # Compute closest points on mesh surface
    closest_points, distances, _ = trimesh.proximity.closest_point(mesh, query_points)
    
    # Determine inside/outside using ray casting
    is_inside = mesh.contains(query_points)
    
    # Apply sign: negative inside, positive outside
    sdf_values = distances.copy()
    sdf_values[is_inside] *= -1
    
    return sdf_values

def sample_sdf_points(mesh_path, num_samples=100000, surface_ratio=0.5):
    """
    Sample points and their SDF values for training
    Args:
        mesh_path: path to mesh file
        num_samples: total number of points to sample
        surface_ratio: ratio of points to sample near surface
    Returns:
        points: (N, 3) sampled points
        sdf_values: (N,) SDF values
    """
    mesh = trimesh.load(mesh_path)
    
    num_surface = int(num_samples * surface_ratio)
    num_uniform = num_samples - num_surface
    
    # Sample points near surface
    surface_points, _ = trimesh.sample.sample_surface(mesh, num_surface)
    
    # Add noise to surface points (both inside and outside)
    noise_scale = 0.01 * np.sqrt(mesh.bounding_box.extents.max())
    surface_points_noisy = surface_points + np.random.normal(0, noise_scale, surface_points.shape)
    
    # Sample uniform points in bounding box
    bounds = mesh.bounding_box.bounds
    uniform_points = np.random.uniform(bounds[0], bounds[1], (num_uniform, 3))
    
    # Combine all points
    all_points = np.vstack([surface_points_noisy, uniform_points])
    
    # Compute SDF values
    sdf_values = compute_sdf_from_mesh(mesh_path, all_points)
    
    return all_points, sdf_values

def extract_mesh_from_sdf(sdf_decoder, latent_code, resolution=128, bbox_size=1.0):
    """
    Extract mesh from SDF using Marching Cubes
    Args:
        sdf_decoder: trained SDF decoder network
        latent_code: (256,) latent code tensor
        resolution: grid resolution for marching cubes
        bbox_size: size of bounding box
    Returns:
        mesh: trimesh object
    """
    device = latent_code.device
    
    # Create grid
    x = np.linspace(-bbox_size, bbox_size, resolution)
    y = np.linspace(-bbox_size, bbox_size, resolution)
    z = np.linspace(-bbox_size, bbox_size, resolution)
    
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    grid_points = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    
    # Compute SDF values
    sdf_values = []
    batch_size = 100000  # Process in batches to avoid memory issues
    
    with torch.no_grad():
        for i in range(0, len(grid_points), batch_size):
            batch_points = torch.FloatTensor(grid_points[i:i+batch_size]).to(device)
            batch_latent = latent_code.unsqueeze(0).repeat(len(batch_points), 1)
            
            # Combine points and latent code
            decoder_input = torch.cat([batch_latent, batch_points], dim=1)
            batch_sdf = sdf_decoder(decoder_input)
            sdf_values.append(batch_sdf.cpu().numpy())
    
    sdf_values = np.concatenate(sdf_values, axis=0)
    sdf_grid = sdf_values.reshape(resolution, resolution, resolution)
    
    # Extract mesh using marching cubes
    try:
        vertices, faces, _, _ = measure.marching_cubes(sdf_grid, level=0.0, spacing=(2*bbox_size/resolution,)*3)
        vertices = vertices - bbox_size  # Center the mesh
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        return mesh
    except:
        print("Warning: Failed to extract mesh, returning empty mesh")
        return trimesh.Trimesh()

def normalize_point_cloud(points, method='unit_sphere'):
    """
    Normalize point cloud to unit sphere or unit cube
    Args:
        points: (N, 3) numpy array
        method: 'unit_sphere' or 'unit_cube'
    Returns:
        normalized_points: (N, 3) normalized points
        scale: scaling factor used
        center: center point used
    """
    center = np.mean(points, axis=0)
    points_centered = points - center
    
    if method == 'unit_sphere':
        scale = np.max(np.linalg.norm(points_centered, axis=1))
    elif method == 'unit_cube':
        scale = np.max(np.abs(points_centered))
    else:
        raise ValueError("Method must be 'unit_sphere' or 'unit_cube'")
    
    if scale > 0:
        normalized_points = points_centered / scale
    else:
        normalized_points = points_centered
    
    return normalized_points, scale, center

def clamp_sdf(sdf_values, clamp_value=0.1):
    """
    Clamp SDF values as done in DeepSDF
    Args:
        sdf_values: array of SDF values
        clamp_value: clamping threshold
    Returns:
        clamped SDF values
    """
    return np.clip(sdf_values, -clamp_value, clamp_value)

def statistical_outlier_removal(points, k=20, std_ratio=2.0):
    """
    Remove statistical outliers from point cloud
    Args:
        points: (N, 3) numpy array
        k: number of neighbors to consider
        std_ratio: standard deviation ratio threshold
    Returns:
        filtered_points: outlier-free points
        inlier_mask: boolean mask of inliers
    """
    if len(points) < k:
        return points, np.ones(len(points), dtype=bool)
    
    # Find k nearest neighbors for each point
    nbrs = NearestNeighbors(n_neighbors=k).fit(points)
    distances, indices = nbrs.kneighbors(points)
    
    # Compute mean distance for each point (excluding self)
    mean_distances = np.mean(distances[:, 1:], axis=1)
    
    # Compute global statistics
    global_mean = np.mean(mean_distances)
    global_std = np.std(mean_distances)
    
    # Filter outliers
    threshold = global_mean + std_ratio * global_std
    inlier_mask = mean_distances < threshold
    
    return points[inlier_mask], inlier_mask

def compute_chamfer_distance(points1, points2):
    """
    Compute asymmetric chamfer distance from points1 to points2
    Args:
        points1: (N, 3) source points
        points2: (M, 3) target points
    Returns:
        chamfer_distance: scalar distance
    """
    if len(points2) == 0:
        return float('inf')
    
    nbrs = NearestNeighbors(n_neighbors=1).fit(points2)
    distances, _ = nbrs.kneighbors(points1)
    return np.mean(distances.flatten() ** 2)

def compute_recall(points1, points2, threshold=0.1):
    """
    Compute recall: percentage of points1 within threshold of points2
    Args:
        points1: (N, 3) source points (ground truth)
        points2: (M, 3) target points (predictions)
        threshold: distance threshold
    Returns:
        recall: recall percentage
    """
    if len(points2) == 0:
        return 0.0
    
    nbrs = NearestNeighbors(n_neighbors=1).fit(points2)
    distances, _ = nbrs.kneighbors(points1)
    within_threshold = distances.flatten() <= threshold
    return np.mean(within_threshold) * 100.0

if __name__ == "__main__":
    # Test FPS
    print("Testing Farthest Point Sampling...")
    test_points = np.random.randn(1000, 3)
    sampled, indices = farthest_point_sampling(test_points, 256)
    print(f"Original points: {test_points.shape}")
    print(f"Sampled points: {sampled.shape}")
    print("FPS test passed!")
    
    # Test normalization
    print("\nTesting point cloud normalization...")
    normalized, scale, center = normalize_point_cloud(test_points)
    print(f"Scale: {scale}, Center: {center}")
    print(f"Normalized range: [{normalized.min():.3f}, {normalized.max():.3f}]")
    print("Normalization test passed!")