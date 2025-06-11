"""
deep_sdf/mv_deepsdf.py

MV-DeepSDF network architectures - all the neural networks for multi-view DeepSDF.
This file goes in your existing deep_sdf/ folder.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional
import trimesh
from skimage import measure


class SharedPCNEncoder(nn.Module):
    """The yellow block from the paper - extracts global features from point clouds"""
    
    def __init__(self):
        super().__init__()
        
        # PointNet-style shared MLPs: 3 -> 128 -> 256 -> 512 -> 1024
        self.conv1 = nn.Conv1d(3, 128, 1)
        self.conv2 = nn.Conv1d(128, 256, 1)
        self.conv3 = nn.Conv1d(256, 512, 1)
        self.conv4 = nn.Conv1d(512, 1024, 1)
        
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(512)
        self.bn4 = nn.BatchNorm1d(1024)
        
        self.tanh = nn.Tanh()  # Normalize to [-1, 1] like DeepSDF
    
    def forward(self, x):
        # x: (B, N, 3) -> transpose to (B, 3, N)
        x = x.transpose(1, 2)
        
        # Apply shared MLPs with BatchNorm and ReLU
        x = F.relu(self.bn1(self.conv1(x)))  # (B, 128, N)
        x = F.relu(self.bn2(self.conv2(x)))  # (B, 256, N)
        
        # Global max pooling
        x = F.max_pool1d(x, kernel_size=x.size(2))  # (B, 256, 1)
        x = x.squeeze(2)  # (B, 256)
        
        # Add remaining layers
        x = x.unsqueeze(2)  # (B, 256, 1)
        x = F.relu(self.bn3(self.conv3(x)))  # (B, 512, 1)
        x = F.relu(self.bn4(self.conv4(x)))  # (B, 1024, 1)
        x = x.squeeze(2)  # (B, 1024)
        
        # Normalize to [-1, 1]
        x = self.tanh(x)
        return x


class DeepSDFEncoder(nn.Module):
    """The green block from the paper - generates latent codes from point clouds"""
    
    def __init__(self, latent_dim=256):
        super().__init__()
        
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        
        self.fc1 = nn.Linear(256, 512)
        self.fc2 = nn.Linear(512, latent_dim)
        self.tanh = nn.Tanh()
    
    def forward(self, x):
        # x: (B, N, 3) -> transpose to (B, 3, N)
        x = x.transpose(1, 2)
        
        # Apply convolutions
        x = F.relu(self.bn1(self.conv1(x)))  # (B, 64, N)
        x = F.relu(self.bn2(self.conv2(x)))  # (B, 128, N)
        x = F.relu(self.bn3(self.conv3(x)))  # (B, 256, N)
        
        # Global max pooling
        x = F.max_pool1d(x, kernel_size=x.size(2))  # (B, 256, 1)
        x = x.squeeze(2)  # (B, 256)
        
        # Generate latent code
        x = F.relu(self.fc1(x))
        x = self.tanh(self.fc2(x))  # (B, latent_dim)
        return x


class SDFDecoder(nn.Module):
    """The SDF decoder - takes query points + latent code -> SDF values"""
    
    def __init__(self, latent_dim=256, hidden_dim=512):
        super().__init__()
        
        # Standard DeepSDF decoder architecture
        self.fc1 = nn.Linear(3 + latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim + 3 + latent_dim, hidden_dim)  # Skip connection
        self.fc5 = nn.Linear(hidden_dim, hidden_dim)
        self.fc6 = nn.Linear(hidden_dim, hidden_dim)
        self.fc7 = nn.Linear(hidden_dim, hidden_dim)
        self.fc8 = nn.Linear(hidden_dim, 1)
        
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, query_points, latent_code):
        # query_points: (B, M, 3), latent_code: (B, 256)
        B, M, _ = query_points.shape
        
        # Expand latent code for each query point
        latent_expanded = latent_code.unsqueeze(1).expand(-1, M, -1)  # (B, M, 256)
        
        # Concatenate query points and latent
        x = torch.cat([query_points, latent_expanded], dim=-1)  # (B, M, 259)
        input_tensor = x  # Store for skip connection
        
        # Flatten for processing
        x = x.view(-1, x.size(-1))  # (B*M, 259)
        
        # Forward pass with skip connection at layer 4
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        
        # Skip connection
        input_flat = input_tensor.view(-1, input_tensor.size(-1))
        x = torch.cat([x, input_flat], dim=-1)
        
        x = F.relu(self.fc4(x))
        x = self.dropout(x)
        x = F.relu(self.fc5(x))
        x = self.dropout(x)
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        sdf = self.fc8(x)  # (B*M, 1)
        
        # Reshape back
        sdf = sdf.view(B, M, 1)
        return sdf


class MVDeepSDFFusionNetwork(nn.Module):
    """The red block from the paper - combines multi-sweep information"""
    
    def __init__(self, latent_dim=256):
        super().__init__()
        
        # Components
        self.shared_pcn_encoder = SharedPCNEncoder()  # Yellow block
        self.deepsdf_encoder = DeepSDFEncoder(latent_dim)  # Green block
        
        # Fusion MLP (red block)
        # Input: 1024 (global features) + 256 (latent code) = 1280
        self.fusion_mlp = nn.Sequential(
            nn.Linear(1280, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim)
        )
    
    def forward(self, multi_sweep_pcs, freeze_deepsdf=False):
        # multi_sweep_pcs: (B, num_sweeps, N, 3)
        B, num_sweeps, N, _ = multi_sweep_pcs.shape
        
        element_representations = []
        
        # Process each sweep
        for i in range(num_sweeps):
            sweep_i = multi_sweep_pcs[:, i, :, :]  # (B, N, 3)
            
            # Extract global features (yellow block)
            global_feat = self.shared_pcn_encoder(sweep_i)  # (B, 1024)
            
            # Extract latent code (green block)
            if freeze_deepsdf:
                with torch.no_grad():
                    latent_code = self.deepsdf_encoder(sweep_i)  # (B, 256)
            else:
                latent_code = self.deepsdf_encoder(sweep_i)  # (B, 256)
            
            # Concatenate (element-level representation)
            element_repr = torch.cat([global_feat, latent_code], dim=1)  # (B, 1280)
            element_representations.append(element_repr)
        
        # Stack and average pool (set-level aggregation)
        element_stack = torch.stack(element_representations, dim=1)  # (B, num_sweeps, 1280)
        aggregated = torch.mean(element_stack, dim=1)  # (B, 1280)
        
        # Map to predicted latent code
        predicted_latent = self.fusion_mlp(aggregated)  # (B, 256)
        return predicted_latent


class MVDeepSDF(nn.Module):
    """Complete MV-DeepSDF model - this is what you'll use for training"""
    
    def __init__(self, latent_dim=256):
        super().__init__()
        
        self.fusion_network = MVDeepSDFFusionNetwork(latent_dim)
        self.sdf_decoder = SDFDecoder(latent_dim)
        self.latent_dim = latent_dim
    
    def forward(self, multi_sweep_pcs, query_points=None, freeze_deepsdf=False):
        # Predict latent code from multi-sweep point clouds
        predicted_latent = self.fusion_network(multi_sweep_pcs, freeze_deepsdf)
        
        results = {'predicted_latent': predicted_latent}
        
        # If query points provided, compute SDF values
        if query_points is not None:
            sdf_values = self.sdf_decoder(query_points, predicted_latent)
            results['sdf_values'] = sdf_values
        
        return results
    
    def extract_mesh(self, predicted_latent, resolution=64, bbox_size=1.0):
        """Extract 3D mesh using Marching Cubes"""
        device = predicted_latent.device
        
        # Create 3D grid
        coords = torch.linspace(-bbox_size, bbox_size, resolution, device=device)
        grid_x, grid_y, grid_z = torch.meshgrid(coords, coords, coords, indexing='ij')
        query_points = torch.stack([grid_x, grid_y, grid_z], dim=-1)
        query_points = query_points.view(1, -1, 3)  # (1, res^3, 3)
        
        # Evaluate SDF in batches
        batch_size = 100000
        sdf_values = []
        
        with torch.no_grad():
            for i in range(0, query_points.shape[1], batch_size):
                batch_points = query_points[:, i:i+batch_size]
                batch_sdf = self.sdf_decoder(batch_points, predicted_latent)
                sdf_values.append(batch_sdf.squeeze(0).squeeze(-1))
        
        # Combine results
        sdf_values = torch.cat(sdf_values, dim=0)
        sdf_grid = sdf_values.view(resolution, resolution, resolution).cpu().numpy()
        
        # Extract mesh with Marching Cubes
        try:
            vertices, faces, _, _ = measure.marching_cubes(sdf_grid, level=0.0)
            vertices = vertices / (resolution - 1) * 2 * bbox_size - bbox_size
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
            return mesh
        except:
            return trimesh.Trimesh()  # Return empty mesh if failed


# Test the model
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create model
    model = MVDeepSDF().to(device)
    
    # Test with dummy data
    B, num_sweeps, N = 2, 6, 256
    multi_sweep_pcs = torch.randn(B, num_sweeps, N, 3).to(device)
    query_points = torch.randn(B, 1000, 3).to(device)
    
    # Forward pass
    results = model(multi_sweep_pcs, query_points)
    
    print(f"✓ Model test passed!")
    print(f"Predicted latent shape: {results['predicted_latent'].shape}")
    print(f"SDF values shape: {results['sdf_values'].shape}")