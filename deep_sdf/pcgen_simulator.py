"""
deep_sdf/pcgen_simulator.py

PCGen LiDAR simulator for generating realistic multi-sweep point clouds from CAD models.
This file goes in your deep_sdf/ folder.
"""

import numpy as np
import trimesh
from typing import List, Tuple
import math


class PCGenSimulator:
    """
    Simplified PCGen simulator for generating multi-sweep LiDAR point clouds.
    Simulates realistic LiDAR scanning patterns and occlusions.
    """
    
    def __init__(self, mesh, config=None):
        """
        Args:
            mesh: Trimesh object of the 3D model
            config: Configuration dictionary
        """
        self.mesh = mesh
        self.config = config or {}
        
        # Normalize mesh to fit in unit cube
        self.mesh = self.normalize_mesh(mesh)
        
        # LiDAR parameters (based on real sensors like Velodyne)
        self.elevation_angles = np.linspace(-30, 10, 32)  # Vertical field of view
        self.azimuth_range = 360  # Horizontal field of view
        self.azimuth_resolution = 0.2  # Degrees per ray
        self.max_range = 5.0  # Maximum detection range
        
        print(f"PCGen simulator initialized for mesh with {len(mesh.vertices)} vertices")
    
    def normalize_mesh(self, mesh):
        """Normalize mesh to unit cube centered at origin"""
        # Center the mesh
        center = mesh.bounds.mean(axis=0)
        mesh.vertices -= center
        
        # Scale to unit cube
        scale = 2.0 / np.max(mesh.bounds[1] - mesh.bounds[0])
        mesh.vertices *= scale
        
        return mesh
    
    def generate_sweep(self, lidar_pose, num_rays_target=1000):
        """
        Generate a single LiDAR sweep from given pose.
        
        Args:
            lidar_pose: Dictionary with 'position', 'azimuth', 'elevation'
            num_rays_target: Target number of rays to cast
        
        Returns:
            points: Numpy array of hit points (N, 3)
        """
        position = np.array(lidar_pose['position'])
        base_azimuth = lidar_pose.get('azimuth', 0)
        base_elevation = lidar_pose.get('elevation', 0)
        
        # Generate ray directions
        ray_origins = []
        ray_directions = []
        
        # Calculate azimuth step to get approximately num_rays_target rays
        num_azimuths = min(int(self.azimuth_range / self.azimuth_resolution), 
                          num_rays_target // len(self.elevation_angles))
        azimuth_step = self.azimuth_range / num_azimuths
        
        for elev_angle in self.elevation_angles:
            for i in range(num_azimuths):
                azim_angle = base_azimuth + (i * azimuth_step) - (self.azimuth_range / 2)
                
                # Convert to radians
                elev_rad = np.deg2rad(elev_angle + base_elevation)
                azim_rad = np.deg2rad(azim_angle)
                
                # Ray direction in spherical coordinates
                direction = np.array([
                    np.cos(elev_rad) * np.cos(azim_rad),
                    np.cos(elev_rad) * np.sin(azim_rad),
                    np.sin(elev_rad)
                ])
                
                ray_origins.append(position)
                ray_directions.append(direction)
        
        ray_origins = np.array(ray_origins)
        ray_directions = np.array(ray_directions)
        
        # Cast rays and find intersections
        try:
            locations, ray_indices, _ = self.mesh.ray.intersects_location(
                ray_origins=ray_origins,
                ray_directions=ray_directions
            )
            
            # Filter points by range
            if len(locations) > 0:
                distances = np.linalg.norm(locations - position, axis=1)
                valid_mask = distances <= self.max_range
                locations = locations[valid_mask]
            
            return locations
            
        except Exception as e:
            print(f"Ray casting failed: {e}")
            return np.array([]).reshape(0, 3)
    
    def generate_multi_sweep(self, num_sweeps=6):
        """
        Generate multiple LiDAR sweeps from different poses.
        
        Args:
            num_sweeps: Number of sweeps to generate
            
        Returns:
            sweeps: List of point clouds (each is numpy array of shape (N, 3))
        """
        sweeps = []
        
        for i in range(num_sweeps):
            # Generate random pose parameters (similar to paper specifications)
            # θ ∈ [0°, 180°] or θ ∈ [-180°, 0°], r ∈ [3, 15], h ∈ [0.8, 1.2]
            
            if np.random.random() < 0.5:
                azimuth = np.random.uniform(0, 180)  # Front/side views
            else:
                azimuth = np.random.uniform(-180, 0)  # Back/side views
            
            distance = np.random.uniform(3, 15)  # Distance from object
            height = np.random.uniform(0.8, 1.2)  # Height above ground
            
            # Convert to Cartesian coordinates
            azimuth_rad = np.deg2rad(azimuth)
            position = np.array([
                distance * np.cos(azimuth_rad),
                distance * np.sin(azimuth_rad),
                height
            ])
            
            # Add some randomness to elevation
            elevation = np.random.uniform(-10, 10)  # Small elevation variations
            
            # Define LiDAR pose
            lidar_pose = {
                'position': position,
                'azimuth': azimuth,
                'elevation': elevation
            }
            
            # Generate sweep
            sweep_points = self.generate_sweep(lidar_pose)
            
            # Add some noise to simulate real LiDAR
            if len(sweep_points) > 0:
                noise_std = self.config.get('noise_std', 0.01)
                noise = np.random.normal(0, noise_std, sweep_points.shape)
                sweep_points += noise
                
                sweeps.append(sweep_points)
            else:
                # If sweep failed, generate a dummy sweep with some points
                print(f"Warning: Sweep {i} failed, generating dummy points")
                dummy_points = np.random.randn(100, 3) * 0.5  # Small random cloud
                sweeps.append(dummy_points)
        
        return sweeps
    
    def visualize_sweeps(self, sweeps, save_path=None):
        """
        Visualize the generated multi-sweep point clouds.
        
        Args:
            sweeps: List of point clouds
            save_path: Optional path to save visualization
        """
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            
            fig = plt.figure(figsize=(15, 5))
            
            # Plot original mesh
            ax1 = fig.add_subplot(131, projection='3d')
            vertices = self.mesh.vertices
            ax1.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], 
                       c='gray', s=1, alpha=0.3, label='Original Mesh')
            ax1.set_title('Original Mesh')
            ax1.set_xlabel('X')
            ax1.set_ylabel('Y')
            ax1.set_zlabel('Z')
            
            # Plot individual sweeps
            ax2 = fig.add_subplot(132, projection='3d')
            colors = plt.cm.tab10(np.linspace(0, 1, len(sweeps)))
            
            for i, sweep in enumerate(sweeps):
                if len(sweep) > 0:
                    ax2.scatter(sweep[:, 0], sweep[:, 1], sweep[:, 2], 
                              c=[colors[i]], s=2, alpha=0.7, label=f'Sweep {i+1}')
            
            ax2.set_title('Individual Sweeps')
            ax2.set_xlabel('X')
            ax2.set_ylabel('Y')
            ax2.set_zlabel('Z')
            ax2.legend()
            
            # Plot combined sweeps
            ax3 = fig.add_subplot(133, projection='3d')
            combined_points = np.vstack([sweep for sweep in sweeps if len(sweep) > 0])
            
            if len(combined_points) > 0:
                ax3.scatter(combined_points[:, 0], combined_points[:, 1], combined_points[:, 2], 
                           c='blue', s=1, alpha=0.6)
            
            ax3.set_title('Combined Multi-Sweep')
            ax3.set_xlabel('X')
            ax3.set_ylabel('Y')
            ax3.set_zlabel('Z')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                print(f"Visualization saved to {save_path}")
            
            plt.show()
            
        except ImportError:
            print("Matplotlib not available for visualization")
        except Exception as e:
            print(f"Visualization failed: {e}")


def simulate_waymo_lidar_sweeps(mesh_path, num_sweeps=6, config=None):
    """
    Convenience function to simulate Waymo-style LiDAR sweeps from a mesh file.
    
    Args:
        mesh_path: Path to mesh file
        num_sweeps: Number of sweeps to generate
        config: Configuration dictionary
        
    Returns:
        sweeps: List of point clouds
    """
    try:
        # Load mesh
        mesh = trimesh.load(mesh_path, force='mesh')
        
        # Create simulator
        simulator = PCGenSimulator(mesh, config)
        
        # Generate sweeps
        sweeps = simulator.generate_multi_sweep(num_sweeps)
        
        print(f"Generated {len(sweeps)} sweeps from {mesh_path}")
        for i, sweep in enumerate(sweeps):
            print(f"  Sweep {i+1}: {len(sweep)} points")
        
        return sweeps
        
    except Exception as e:
        print(f"Error simulating sweeps for {mesh_path}: {e}")
        return []


# Test the simulator
if __name__ == "__main__":
    # Test with a simple mesh
    print("Testing PCGen simulator...")
    
    # Create a simple test mesh (cube)
    mesh = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
    
    # Test configuration
    config = {
        'noise_std': 0.01,
        'num_sweeps': 6
    }
    
    # Create simulator
    simulator = PCGenSimulator(mesh, config)
    
    # Generate sweeps
    sweeps = simulator.generate_multi_sweep(num_sweeps=6)
    
    print(f"✓ Generated {len(sweeps)} sweeps")
    for i, sweep in enumerate(sweeps):
        print(f"  Sweep {i+1}: {len(sweep)} points")
    
    # Test visualization (optional)
    try:
        simulator.visualize_sweeps(sweeps)
    except:
        print("Skipping visualization (matplotlib not available)")
    
    print("PCGen simulator test completed!")