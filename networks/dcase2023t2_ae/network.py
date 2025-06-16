import torch
from torch import nn

# First, define the AENet class
class AENet(nn.Module):
    def __init__(self, input_dim, block_size, time_stretch_prob=0.5, time_stretch_range=(0.9, 1.1)):
        super(AENet, self).__init__()
        self.input_dim = input_dim
        self.cov_source = nn.Parameter(torch.zeros(block_size, block_size), requires_grad=False)
        self.cov_target = nn.Parameter(torch.zeros(block_size, block_size), requires_grad=False)
        
        # Time stretch augmentation parameters
        self.time_stretch_prob = time_stretch_prob
        self.time_stretch_range = time_stretch_range
        self.training_mode = True

        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, 128),
            nn.BatchNorm1d(128, momentum=0.01, eps=1e-03),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128, momentum=0.01, eps=1e-03),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128, momentum=0.01, eps=1e-03),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128, momentum=0.01, eps=1e-03),
            nn.ReLU(),
            nn.Linear(128, 8),
            nn.BatchNorm1d(8, momentum=0.01, eps=1e-03),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(8, 128),
            nn.BatchNorm1d(128, momentum=0.01, eps=1e-03),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128, momentum=0.01, eps=1e-03),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128, momentum=0.01, eps=1e-03),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128, momentum=0.01, eps=1e-03),
            nn.ReLU(),
            nn.Linear(128, self.input_dim)
        )
        
        # Initialize covariance matrices for Mahalanobis distance
        self.register_buffer('mean_source', torch.zeros(block_size))
        self.register_buffer('mean_target', torch.zeros(block_size))
        self.register_buffer('inv_cov_source', torch.eye(block_size))
        self.register_buffer('inv_cov_target', torch.eye(block_size))
    
    def forward(self, x, apply_augmentation=True):
        """
        Forward pass with optional time stretch augmentation.
        
        Args:
            x (torch.Tensor): Input data
            apply_augmentation (bool): Whether to apply augmentation
        
        Returns:
            tuple: (reconstructed_x, latent_z)
        """
        x_flat = x.view(-1, self.input_dim)
        
        # Apply time stretch augmentation during training if requested
        if self.training and apply_augmentation and torch.rand(1).item() < self.time_stretch_prob:
            x_aug = self.apply_time_stretch(x_flat)
            z = self.encoder(x_aug)
        else:
            z = self.encoder(x_flat)
            
        reconstructed = self.decoder(z)
        return reconstructed, z
    
    def apply_time_stretch(self, x):
        """
        Apply time stretching to feature vectors.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, feature_dim]
            
        Returns:
            torch.Tensor: Time-stretched features
        """
        import torch.nn.functional as F
        import numpy as np
        
        batch_size = x.size(0)
        feature_dim = x.size(1)
        
        # Process each sample in the batch
        augmented_batch = []
        
        for i in range(batch_size):
            # Apply augmentation with probability
            if torch.rand(1).item() < self.time_stretch_prob:
                # Choose a stretch factor
                stretch_factor = torch.FloatTensor(1).uniform_(*self.time_stretch_range).item()
                
                # Divide the feature vector into segments (assuming time is somehow represented)
                num_segments = 8  # Arbitrary division, adjust based on your data
                segment_length = feature_dim // num_segments
                
                # Process each segment
                stretched_segments = []
                for j in range(num_segments):
                    start_idx = j * segment_length
                    end_idx = start_idx + segment_length if j < num_segments - 1 else feature_dim
                    
                    segment = x[i, start_idx:end_idx]
                    
                    # Stretch the segment
                    new_length = int(segment.size(0) * stretch_factor)
                    new_length = max(1, new_length)  # Ensure at least one element
                    
                    # Use interpolation
                    stretched = F.interpolate(
                        segment.unsqueeze(0).unsqueeze(0),  # [1, 1, segment_length]
                        size=new_length,
                        mode='linear',
                        align_corners=False
                    ).squeeze(0).squeeze(0)  # [new_length]
                    
                    # Resize back to original length
                    resized = F.interpolate(
                        stretched.unsqueeze(0).unsqueeze(0),  # [1, 1, new_length]
                        size=end_idx - start_idx,
                        mode='linear',
                        align_corners=False
                    ).squeeze(0).squeeze(0)  # [segment_length]
                    
                    stretched_segments.append(resized)
                
                # Combine segments
                augmented_sample = torch.cat(stretched_segments)
                augmented_batch.append(augmented_sample)
            else:
                augmented_batch.append(x[i])
        
        return torch.stack(augmented_batch)
    
    def train(self, mode=True):
        """Override train method to set training_mode flag"""
        self.training_mode = mode
        return super().train(mode)
    
    def eval(self):
        """Override eval method to set training_mode flag"""
        self.training_mode = False
        return super().eval()
    
    def compute_mahalanobis_distance(self, z, is_source=True):
        """
        Compute Mahalanobis distance for latent vectors.
        
        Args:
            z (torch.Tensor): Latent vectors
            is_source (bool): Whether to use source or target distribution
        
        Returns:
            torch.Tensor: Mahalanobis distances
        """
        if is_source:
            mean = self.mean_source
            inv_cov = self.inv_cov_source
        else:
            mean = self.mean_target
            inv_cov = self.inv_cov_target
            
        # Center the data
        centered = z - mean.unsqueeze(0)
        
        # Compute Mahalanobis distance: sqrt((x-μ)ᵀΣ⁻¹(x-μ))
        distances = torch.sqrt(torch.sum(
            torch.matmul(centered, inv_cov) * centered, dim=1
        ))
        
        return distances
    
    def update_covariance(self, embeddings, is_source=True):
        """
        Update covariance matrix for source or target domain.
        
        Args:
            embeddings (torch.Tensor): Latent embeddings
            is_source (bool): Whether to update source or target distribution
        """
        # Compute mean
        mean = embeddings.mean(dim=0)
        
        # Center the data
        centered = embeddings - mean.unsqueeze(0)
        
        # Compute covariance matrix
        n_samples = embeddings.size(0)
        cov = torch.matmul(centered.t(), centered) / (n_samples - 1)
        
        # Add small regularization for numerical stability
        cov = cov + torch.eye(cov.size(0), device=cov.device) * 1e-5
        
        # Compute inverse
        try:
            inv_cov = torch.inverse(cov)
        except:
            # Fallback to pseudo-inverse if inverse fails
            inv_cov = torch.pinverse(cov)
        
        # Update parameters
        if is_source:
            self.mean_source.copy_(mean.detach())
            self.cov_source.copy_(cov.detach())
            self.inv_cov_source.copy_(inv_cov.detach())
        else:
            self.mean_target.copy_(mean.detach())
            self.cov_target.copy_(cov.detach())
            self.inv_cov_target.copy_(inv_cov.detach())
    
    def compute_anomaly_score(self, x, use_mahalanobis=True):
        """
        Compute anomaly score for input samples.
        
        Args:
            x (torch.Tensor): Input samples
            use_mahalanobis (bool): Whether to use Mahalanobis distance
        
        Returns:
            torch.Tensor: Anomaly scores
        """
        self.eval()  # Set to evaluation mode
        with torch.no_grad():
            # Get reconstruction and latent representation
            x_flat = x.view(-1, self.input_dim)
            reconstructed, z = self.forward(x_flat, apply_augmentation=False)
            
            if use_mahalanobis:
                # Compute Mahalanobis distance in latent space
                anomaly_scores = self.compute_mahalanobis_distance(z, is_source=False)
            else:
                # Use reconstruction error as anomaly score
                reconstruction_error = torch.mean((reconstructed - x_flat) ** 2, dim=1)
                anomaly_scores = reconstruction_error
                
        return anomaly_scores

# Then, if you need to create an instance, do it after the class definition
# For example:
# model = AENet(input_dim=640, block_size=8)

