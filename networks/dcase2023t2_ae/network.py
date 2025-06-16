import torch
from torch import nn
import numpy as np
from sklearn.neighbors import NearestNeighbors

class AENet(nn.Module):
    def __init__(self, input_dim, block_size):
        super(AENet, self).__init__()
        self.input_dim = input_dim
        self.block_size = block_size
        
        # Parameters for storing covariance matrices
        self.cov_source = nn.Parameter(torch.zeros(block_size, block_size), requires_grad=False)
        self.cov_target = nn.Parameter(torch.zeros(block_size, block_size), requires_grad=False)
        
        # Inverse covariance matrices (precision matrices)
        self.register_buffer('inv_cov_source', torch.zeros(block_size, block_size))
        self.register_buffer('inv_cov_target', torch.zeros(block_size, block_size))

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
            nn.Linear(128, block_size),
            nn.BatchNorm1d(block_size, momentum=0.01, eps=1e-03),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(block_size, 128),
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

    def forward(self, x):
        z = self.encoder(x.view(-1, self.input_dim))
        return self.decoder(z), z
    
    def compute_covariance_matrices(self, source_data, target_data, eps=1e-6):
        """
        Compute covariance matrices for source and target domains
        
        Parameters:
        -----------
        source_data : torch.Tensor
            Data from source domain
        target_data : torch.Tensor
            Data from target domain
        eps : float
            Small value to ensure numerical stability
        """
        # Get reconstructions and latent representations
        with torch.no_grad():
            source_recon, source_z = self(source_data)
            target_recon, target_z = self(target_data)
            
            # Compute reconstruction errors
            source_diff = source_recon - source_data
            target_diff = target_recon - target_data
            
            # Reshape to 2D if needed
            if len(source_diff.shape) > 2:
                source_diff = source_diff.view(source_diff.size(0), -1)
                target_diff = target_diff.view(target_diff.size(0), -1)
            
            # Compute covariance matrices
            source_diff = source_diff - source_diff.mean(dim=0, keepdim=True)
            target_diff = target_diff - target_diff.mean(dim=0, keepdim=True)
            
            # Compute covariance matrices
            self.cov_source.data = (source_diff.T @ source_diff) / (source_diff.size(0) - 1)
            self.cov_target.data = (target_diff.T @ target_diff) / (target_diff.size(0) - 1)
            
            # Add small value to diagonal for numerical stability
            self.cov_source.data += torch.eye(self.cov_source.size(0), device=self.cov_source.device) * eps
            self.cov_target.data += torch.eye(self.cov_target.size(0), device=self.cov_target.device) * eps
            
            # Compute inverse covariance matrices
            self.inv_cov_source = torch.inverse(self.cov_source.data)
            self.inv_cov_target = torch.inverse(self.cov_target.data)
    
    def mahalanobis_distance(self, x, recon, inv_cov):
        """
        Compute Mahalanobis distance between original and reconstructed samples
        
        Parameters:
        -----------
        x : torch.Tensor
            Original samples
        recon : torch.Tensor
            Reconstructed samples
        inv_cov : torch.Tensor
            Inverse covariance matrix
            
        Returns:
        --------
        torch.Tensor
            Mahalanobis distances
        """
        diff = recon - x
        if len(diff.shape) > 2:
            diff = diff.view(diff.size(0), -1)
        
        # Compute Mahalanobis distance
        dist = torch.sum((diff @ inv_cov) * diff, dim=1)
        return dist
    
    def compute_anomaly_score(self, x, mode='mse'):
        """
        Compute anomaly score for input samples
        
        Parameters:
        -----------
        x : torch.Tensor
            Input samples
        mode : str
            'mse': Mean Squared Error (Simple Autoencoder mode)
            'mahalanobis': Mahalanobis distance using source domain covariance
            'selective_mahalanobis': Minimum of source and target Mahalanobis distances
            
        Returns:
        --------
        torch.Tensor
            Anomaly scores
        """
        with torch.no_grad():
            recon, z = self(x)
            
            if mode == 'mse':
                # Simple Autoencoder mode
                diff = recon - x
                if len(diff.shape) > 2:
                    diff = diff.view(diff.size(0), -1)
                scores = torch.sum(diff ** 2, dim=1)
                
            elif mode == 'mahalanobis':
                # Mahalanobis distance using source domain covariance
                scores = self.mahalanobis_distance(x, recon, self.inv_cov_source)
                
            elif mode == 'selective_mahalanobis':
                # Selective Mahalanobis mode - minimum of source and target distances
                source_dist = self.mahalanobis_distance(x, recon, self.inv_cov_source)
                target_dist = self.mahalanobis_distance(x, recon, self.inv_cov_target)
                scores = torch.min(source_dist, target_dist)
                
            else:
                raise ValueError(f"Unknown mode: {mode}")
                
            return scores


class SMOTE:
    """
    Implementation of SMOTE (Synthetic Minority Over-sampling Technique) for PyTorch tensors.
    
    Parameters:
    -----------
    k : int, optional (default=5)
        Number of nearest neighbors to use for generating synthetic samples.
    n_samples : int or float, optional (default=1.0)
        If int, specifies the exact number of synthetic samples to generate.
        If float, specifies the ratio of synthetic samples to generate relative to the original minority class.
    random_state : int, optional (default=None)
        Random seed for reproducibility.
    distance_metric : str, optional (default='euclidean')
        Distance metric to use for finding nearest neighbors.
        Options: 'euclidean', 'mahalanobis'
    """
    
    def __init__(self, k=5, n_samples=1.0, random_state=None, distance_metric='euclidean'):
        self.k = k
        self.n_samples = n_samples
        self.random_state = random_state
        self.distance_metric = distance_metric
        self.inv_cov = None  # For Mahalanobis distance
        
        if random_state is not None:
            torch.manual_seed(random_state)
            np.random.seed(random_state)
    
    def fit_resample(self, X, y, inv_cov=None):
        """
        Generate synthetic samples for the minority class.
        
        Parameters:
        -----------
        X : torch.Tensor
            Feature tensor of shape (n_samples, n_features)
        y : torch.Tensor
            Target tensor of shape (n_samples,)
        inv_cov : torch.Tensor, optional
            Inverse covariance matrix for Mahalanobis distance
            
        Returns:
        --------
        X_resampled : torch.Tensor
            Resampled feature tensor with synthetic samples
        y_resampled : torch.Tensor
            Corresponding target tensor with synthetic samples
        """
        # Set inverse covariance matrix if provided
        if inv_cov is not None:
            self.inv_cov = inv_cov
        
        # Convert tensors to numpy for easier processing
        X_np = X.detach().cpu().numpy()
        y_np = y.detach().cpu().numpy()
        
        # Find minority class
        unique_classes, class_counts = np.unique(y_np, return_counts=True)
        minority_class = unique_classes[np.argmin(class_counts)]
        
        # Get minority class samples
        minority_indices = np.where(y_np == minority_class)[0]
        minority_samples = X_np[minority_indices]
        
        # Determine number of synthetic samples to generate
        if isinstance(self.n_samples, float):
            n_synthetic = int(self.n_samples * len(minority_samples))
        else:
            n_synthetic = self.n_samples
        
        # Find k nearest neighbors for each minority sample
        if self.distance_metric == 'euclidean':
            nn = NearestNeighbors(n_neighbors=self.k + 1)  # +1 because the sample itself is included
            nn.fit(minority_samples)
            distances, indices = nn.kneighbors(minority_samples)
        elif self.distance_metric == 'mahalanobis':
            if self.inv_cov is None:
                raise ValueError("Inverse covariance matrix must be provided for Mahalanobis distance")
            
            # Convert inverse covariance to numpy
            inv_cov_np = self.inv_cov.detach().cpu().numpy()
            
            # Compute pairwise Mahalanobis distances
            n_samples = minority_samples.shape[0]
            distances = np.zeros((n_samples, n_samples))
            for i in range(n_samples):
                for j in range(n_samples):
                    diff = minority_samples[i] - minority_samples[j]
                    distances[i, j] = np.sqrt(diff @ inv_cov_np @ diff.T)
            
            # Get indices of k+1 nearest neighbors
            indices = np.argsort(distances, axis=1)[:, :self.k+1]
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")
        
        # Generate synthetic samples
        synthetic_samples = []
        for i in range(n_synthetic):
            # Randomly select a minority sample
            sample_idx = np.random.randint(0, len(minority_samples))
            sample = minority_samples[sample_idx]
            
            # Randomly select one of its neighbors (excluding itself)
            neighbor_idx = indices[sample_idx][np.random.randint(1, self.k + 1)]
            neighbor = minority_samples[neighbor_idx]
            
            # Generate synthetic sample by interpolation
            alpha = np.random.random()
            synthetic_sample = sample + alpha * (neighbor - sample)
            synthetic_samples.append(synthetic_sample)
        
        # Combine original and synthetic samples
        X_resampled_np = np.vstack([X_np, synthetic_samples])
        y_resampled_np = np.hstack([y_np, np.full(len(synthetic_samples), minority_class)])
        
        # Convert back to PyTorch tensors
        X_resampled = torch.tensor(X_resampled_np, dtype=X.dtype, device=X.device)
        y_resampled = torch.tensor(y_resampled_np, dtype=y.dtype, device=y.device)
        
        return X_resampled, y_resampled
    
    def fit_resample_latent(self, model, X, y, mode='selective_mahalanobis'):
        """
        Generate synthetic samples in the latent space of an autoencoder.
        
        Parameters:
        -----------
        model : AENet
            Trained autoencoder model
        X : torch.Tensor
            Feature tensor of shape (n_samples, n_features)
        y : torch.Tensor
            Target tensor of shape (n_samples,)
        mode : str
            'euclidean': Use Euclidean distance for SMOTE
            'mahalanobis': Use Mahalanobis distance with source covariance
            'selective_mahalanobis': Use minimum of source and target Mahalanobis distances
            
        Returns:
        --------
        X_resampled : torch.Tensor
            Resampled feature tensor with synthetic samples
        y_resampled : torch.Tensor
            Corresponding target tensor with synthetic samples
        """
        # Get latent representations
        model.eval()
        with torch.no_grad():
            _, latent = model(X)
        
        # Set distance metric and inverse covariance matrix based on mode
        if mode == 'euclidean':
            self.distance_metric = 'euclidean'
            inv_cov = None
        elif mode == 'mahalanobis':
            self.distance_metric = 'mahalanobis'
            inv_cov = model.inv_cov_source
        elif mode == 'selective_mahalanobis':
            # For selective Mahalanobis, we'll use source covariance for neighbor finding
            # but will compute both distances when calculating anomaly scores
            self.distance_metric = 'mahalanobis'
            inv_cov = model.inv_cov_source
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        # Apply SMOTE in the latent space
        latent_resampled, y_resampled = self.fit_resample(latent, y, inv_cov)
        
        # Decode the synthetic samples
        with torch.no_grad():
            synthetic_indices = range(len(y), len(y_resampled))
            synthetic_latent = latent_resampled[synthetic_indices]
            synthetic_X = model.decoder(synthetic_latent)
        
        # Combine original and synthetic samples
        X_resampled = torch.cat([X, synthetic_X], dim=0)
        
        return X_resampled, y_resampled


# Example usage:
"""
# Initialize the model
input_dim = 128
block_size = 8  # This should match the latent dimension in the encoder
model = AENet(input_dim, block_size)

# Create some dummy data
X = torch.randn(100, input_dim)
y = torch.cat([torch.zeros(80), torch.ones(20)])  # Imbalanced dataset

# Split into source and target domains
source_indices = torch.where(y == 0)[0][:70]  # 70 samples from class 0
target_indices = torch.cat([torch.where(y == 0)[0][70:], torch.where(y == 1)[0]])  # 10 samples from class 0, 20 from class 1
X_source = X[source_indices]
X_target = X[target_indices]

# Compute covariance matrices for Mahalanobis distance
model.compute_covariance_matrices(X_source, X_target)

# Initialize SMOTE with Mahalanobis distance
smote = SMOTE(k=5, n_samples=60, distance_metric='mahalanobis')

# Apply SMOTE in the latent space with selective Mahalanobis mode
X_resampled, y_resampled = smote.fit_resample_latent(model, X, y, mode='selective_mahalanobis')

# Compute anomaly scores using selective Mahalanobis mode
anomaly_scores = model.compute_anomaly_score(X, mode='selective_mahalanobis')
"""
