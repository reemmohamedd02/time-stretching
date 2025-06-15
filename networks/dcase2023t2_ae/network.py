import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import KMeans
from collections import defaultdict, Counter
import torchaudio
import random

class AdaProjLoss(nn.Module):
    """
    AdaProj Loss: Adaptively scaled angular margin subspace projections
    as described in the FKIE-VUB paper
    """
    def __init__(self, in_features, out_features, s=None):
        super(AdaProjLoss, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s  # Scale parameter (if None, will be adaptively computed)
        
        # Initialize projection matrices (one per class)
        self.projections = nn.Parameter(torch.Tensor(out_features, in_features, in_features // 2))
        nn.init.orthogonal_(self.projections.view(out_features, -1))
        
    def forward(self, embeddings, labels):
        # Normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        # Project embeddings to class-specific subspaces
        batch_size = embeddings.size(0)
        logits = torch.zeros(batch_size, self.out_features, device=embeddings.device)
        
        for i in range(self.out_features):
            # Get projection matrix for class i
            proj_i = self.projections[i]
            
            # Project embeddings to class i subspace
            proj_embeddings = torch.matmul(embeddings.unsqueeze(1), proj_i).squeeze(1)
            
            # Compute norm of projected embeddings (similarity to class i subspace)
            logits[:, i] = torch.norm(proj_embeddings, p=2, dim=1)
        
        # Compute adaptive scale if not provided
        if self.s is None:
            with torch.no_grad():
                # Compute theta (angle) between embeddings and their target subspaces
                target_logits = torch.gather(logits, 1, labels.unsqueeze(1)).squeeze(1)
                theta = torch.acos(torch.clamp(target_logits, -1.0 + 1e-7, 1.0 - 1e-7))
                
                # Compute adaptive scale
                self.s = torch.log(torch.tensor(self.out_features - 1, dtype=torch.float, device=embeddings.device)) / torch.cos(torch.mean(theta))
        
        # Scale logits
        logits = logits * self.s
        
        # Compute cross-entropy loss
        loss = F.cross_entropy(logits, labels)
        
        return logits, loss

class ResidualBlock(nn.Module):
    """
    Residual block for the spectrogram branch
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
            
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class SpectrogramEncoder(nn.Module):
    """
    Modified ResNet architecture for spectrogram processing
    """
    def __init__(self, output_dim=256):
        super(SpectrogramEncoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # 4 residual blocks
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
        # Global average pooling and linear layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, output_dim)
        
    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # Add channel dimension if needed
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
            
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x

class SpectrumEncoder(nn.Module):
    """
    CNN for spectrum processing
    """
    def __init__(self, input_dim, output_dim=256):
        super(SpectrumEncoder, self).__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv1d(64, 128, kernel_size=7, stride=2, padding=3)
        self.bn2 = nn.BatchNorm1d(128)
        
        self.conv3 = nn.Conv1d(128, 256, kernel_size=7, stride=2, padding=3)
        self.bn3 = nn.BatchNorm1d(256)
        
        # Calculate the size after convolutions
        self.flatten_size = 256 * (input_dim // 8)
        
        # Dense layers
        self.fc1 = nn.Linear(self.flatten_size, 512)
        self.bn_fc1 = nn.BatchNorm1d(512)
        
        self.fc2 = nn.Linear(512, 512)
        self.bn_fc2 = nn.BatchNorm1d(512)
        
        self.fc3 = nn.Linear(512, 512)
        self.bn_fc3 = nn.BatchNorm1d(512)
        
        self.fc4 = nn.Linear(512, 512)
        self.bn_fc4 = nn.BatchNorm1d(512)
        
        self.fc5 = nn.Linear(512, output_dim)
        
    def forward(self, x):
        # Add channel dimension if needed
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
            
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        
        x = x.view(x.size(0), -1)  # Flatten
        
        x = self.relu(self.bn_fc1(self.fc1(x)))
        x = self.relu(self.bn_fc2(self.fc2(x)))
        x = self.relu(self.bn_fc3(self.fc3(x)))
        x = self.relu(self.bn_fc4(self.fc4(x)))
        x = self.fc5(x)
        
        return x

class AENet(nn.Module):
    def __init__(self, input_dim, block_size, n_machine_types=7, n_attributes=None):
        super(AENet, self).__init__()
        self.input_dim = input_dim
        self.block_size = block_size
        self.n_machine_types = n_machine_types
        
        # If n_attributes is not provided, assume 3 attributes per machine type
        if n_attributes is None:
            n_attributes = [3] * n_machine_types
        self.n_attributes = n_attributes
        
        # Calculate total number of classes (machine type + attribute combinations)
        self.n_classes = sum([max(1, attr) for attr in n_attributes])
        
        # Parameters for storing covariance matrices
        self.cov_source = nn.Parameter(torch.zeros(block_size, block_size), requires_grad=False)
        self.cov_target = nn.Parameter(torch.zeros(block_size, block_size), requires_grad=False)
        
        # Buffers for inverse covariance matrices
        self.register_buffer('inv_cov_source', torch.zeros(block_size, block_size))
        self.register_buffer('inv_cov_target', torch.zeros(block_size, block_size))
        
        # Cluster centers and counts for k-means
        self.register_buffer('cluster_centers', torch.zeros(32, block_size))
        self.register_buffer('cluster_counts', torch.zeros(32))
        
        # Original encoder and decoder for compatibility
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
        
        # FKIE-VUB dual-branch architecture
        self.spectrogram_encoder = SpectrogramEncoder(output_dim=block_size // 2)
        self.spectrum_encoder = SpectrumEncoder(input_dim=8000, output_dim=block_size // 2)
        
        # Classification head for auxiliary task
        self.classifier = nn.Linear(block_size, self.n_classes)
        
        # SSL classification head for FeatEx
        self.ssl_classifier = nn.Linear(block_size, 2)  # 2 classes: same or different
        
        # AdaProj loss for classification
        self.adaproj = AdaProjLoss(block_size, self.n_classes)
        
        # AdaProj loss for SSL
        self.ssl_adaproj = AdaProjLoss(block_size, 2)
        
    def forward(self, x):
        """
        Forward pass through the autoencoder (for compatibility with original code)
        
        Parameters:
        -----------
        x : torch.Tensor
            Input tensor
            
        Returns:
        --------
        tuple
            (reconstructed_x, latent_z)
        """
        z = self.encoder(x.view(-1, self.input_dim))
        return self.decoder(z), z
    
    def forward_dual(self, spectrogram, spectrum):
        """
        Forward pass through the dual-branch architecture
        
        Parameters:
        -----------
        spectrogram : torch.Tensor
            Spectrogram input
        spectrum : torch.Tensor
            Spectrum input
            
        Returns:
        --------
        tuple
            (latent_z, class_logits, ssl_logits)
        """
        # Get embeddings from both branches
        z_spec = self.spectrogram_encoder(spectrogram)
        z_spect = self.spectrum_encoder(spectrum)
        
        # Concatenate embeddings
        z = torch.cat([z_spec, z_spect], dim=1)
        
        # Get class logits
        class_logits = self.classifier(z)
        
        # Get SSL logits
        ssl_logits = self.ssl_classifier(z)
        
        return z, class_logits, ssl_logits
    
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
        self.eval()  # Set model to evaluation mode
        
        with torch.no_grad():
            # Process source domain data
            _, source_z = self(source_data)
            
            # Process target domain data
            _, target_z = self(target_data)
            
            # Compute mean vectors
            source_mean = source_z.mean(dim=0)
            target_mean = target_z.mean(dim=0)
            
            # Center the data
            source_centered = source_z - source_mean
            target_centered = target_z - target_mean
            
            # Compute covariance matrices
            self.cov_source.data = (source_centered.T @ source_centered) / max(1, source_centered.size(0) - 1)
            self.cov_target.data = (target_centered.T @ target_centered) / max(1, target_centered.size(0) - 1)
            
            # Add small value to diagonal for numerical stability
            self.cov_source.data += torch.eye(self.cov_source.size(0), 
                                             device=self.cov_source.device) * eps
            self.cov_target.data += torch.eye(self.cov_target.size(0), 
                                             device=self.cov_target.device) * eps
            
            # Compute inverse covariance matrices
            try:
                self.inv_cov_source = torch.inverse(self.cov_source.data)
                self.inv_cov_target = torch.inverse(self.cov_target.data)
            except RuntimeError:
                # If matrix is not invertible, use pseudo-inverse
                self.inv_cov_source = torch.pinverse(self.cov_source.data)
                self.inv_cov_target = torch.pinverse(self.cov_target.data)
    
    def mahalanobis_distance(self, x, mean, inv_cov):
        """
        Compute Mahalanobis distance between samples and a distribution
        
        Parameters:
        -----------
        x : torch.Tensor
            Samples
        mean : torch.Tensor
            Mean vector of the distribution
        inv_cov : torch.Tensor
            Inverse covariance matrix of the distribution
            
        Returns:
        --------
        torch.Tensor
            Mahalanobis distances for each sample
        """
        # Center the data
        x_centered = x - mean
        
        # Compute Mahalanobis distance for each sample
        dist = torch.sum((x_centered @ inv_cov) * x_centered, dim=1)
        return dist
    
    def compute_kmeans_clusters(self, source_data, k=32):
        """
        Compute k-means clusters on source domain data
        
        Parameters:
        -----------
        source_data : torch.Tensor
            Source domain data
        k : int
            Number of clusters
            
        Returns:
        --------
        torch.Tensor
            Cluster centers
        """
        self.eval()  # Set model to evaluation mode
        
        with torch.no_grad():
            # Get embeddings
            _, z = self(source_data)
            
            # Convert to numpy for k-means
            z_np = z.cpu().numpy()
            
            # Perform k-means clustering
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(z_np)
            
            # Convert centers to tensor
            self.cluster_centers = torch.tensor(kmeans.cluster_centers_, device=z.device)
            
            # Count samples per cluster
            for i in range(k):
                self.cluster_counts[i] = (labels == i).sum()
            
            return self.cluster_centers, labels
    
    def compute_anomaly_score(self, x, target_data=None, mode='fkie'):
        """
        Compute anomaly score for input samples
        
        Parameters:
        -----------
        x : torch.Tensor
            Input samples
        target_data : torch.Tensor or None
            Target domain data for computing distances
        mode : str
            'mse': Mean Squared Error
            'mahalanobis': Mahalanobis distance
            'selective_mahalanobis': Minimum of source and target Mahalanobis distances
            'fkie': FKIE-VUB approach with k-means and cosine distances
            
        Returns:
        --------
        torch.Tensor
            Anomaly scores for each sample
        """
        self.eval()  # Set model to evaluation mode
        
        with torch.no_grad():
            # Get embeddings
            _, z = self(x)
            
            if mode == 'mse':
                # Simple Autoencoder mode - Mean Squared Error
                recon, _ = self(x)
                diff = recon - x.view(-1, self.input_dim)
                scores = torch.sum(diff ** 2, dim=1)
                
            elif mode == 'mahalanobis':
                # Mahalanobis distance using source domain covariance
                source_mean = z.mean(dim=0)  # Use batch mean as an approximation
                scores = self.mahalanobis_distance(z, source_mean, self.inv_cov_source)
                
            elif mode == 'selective_mahalanobis':
                # Selective Mahalanobis mode - minimum of source and target distances
                source_mean = z.mean(dim=0)
                target_mean = z.mean(dim=0)
                
                source_dist = self.mahalanobis_distance(z, source_mean, self.inv_cov_source)
                target_dist = self.mahalanobis_distance(z, target_mean, self.inv_cov_target)
                
                scores = torch.min(source_dist, target_dist)
                
            elif mode == 'fkie':
                # FKIE-VUB approach with k-means and cosine distances
                
                # Normalize embeddings
                z_norm = F.normalize(z, p=2, dim=1)
                
                # Compute cosine distances to cluster centers
                cluster_centers_norm = F.normalize(self.cluster_centers, p=2, dim=1)
                cluster_distances = 1 - torch.matmul(z_norm, cluster_centers_norm.t())
                
                # Get minimum distance to any cluster center
                min_cluster_dist, _ = torch.min(cluster_distances, dim=1)
                
                # If target data is provided, compute distances to target samples
                if target_data is not None:
                    _, target_z = self(target_data)
                    target_z_norm = F.normalize(target_z, p=2, dim=1)
                    
                    # Compute cosine distances to target samples
                    target_distances = 1 - torch.matmul(z_norm, target_z_norm.t())
                    
                    # Get minimum distance to any target sample
                    min_target_dist, _ = torch.min(target_distances, dim=1)
                    
                    # Take minimum of cluster and target distances
                    scores = torch.min(min_cluster_dist, min_target_dist)
                else:
                    scores = min_cluster_dist
                
            else:
                raise ValueError(f"Unknown mode: {mode}")
                
            return scores
    
    def compute_sample_weights(self, machine_types, attributes):
        """
        Compute balanced sample weights for training
        
        Parameters:
        -----------
        machine_types : torch.Tensor
            Machine type labels
        attributes : torch.Tensor
            Attribute labels
            
        Returns:
        --------
        torch.Tensor
            Sample weights
        """
        device = machine_types.device
        
        # Create class labels (machine type + attribute combinations)
        class_labels = []
        for i in range(len(machine_types)):
            machine_type = machine_types[i].item()
            attribute = attributes[i].item() if attributes is not None else 0
            class_labels.append((machine_type, attribute))
        
        # Count samples per class
        class_counts = Counter(class_labels)
        
        # Compute weights (inverse of class frequency)
        weights = torch.zeros(len(machine_types), device=device)
        for i, label in enumerate(class_labels):
            weights[i] = 1.0 / max(1, class_counts[label])
        
        # Normalize weights per machine type
        for machine_type in torch.unique(machine_types):
            mask = (machine_types == machine_type)
            weights[mask] = weights[mask] / weights[mask].sum()
        
        # Rescale weights to have mean 1
        weights = weights * (len(weights) / weights.sum())
        
        return weights
    
    def apply_mixup(self, x, y, alpha=1.0):
        """
        Apply mixup data augmentation
        
        Parameters:
        -----------
        x : torch.Tensor
            Input data
        y : torch.Tensor
            Labels
        alpha : float
            Mixup coefficient
            
        Returns:
        --------
        tuple
            (mixed_x, mixed_y, lambda)
        """
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
        
        batch_size = x.size(0)
        index = torch.randperm(batch_size, device=x.device)
        
        mixed_x = lam * x + (1 - lam) * x[index]
        mixed_y = y.clone()
        
        return mixed_x, mixed_y, lam, index
    
    def apply_featex(self, z_spec, z_spect, y, prob=0.5):
        """
        Apply Feature Exchange (FeatEx) for self-supervised learning
        
        Parameters:
        -----------
        z_spec : torch.Tensor
            Spectrogram embeddings
        z_spect : torch.Tensor
            Spectrum embeddings
        y : torch.Tensor
            Labels
        prob : float
            Probability of applying FeatEx
            
        Returns:
        --------
        tuple
            (z, ssl_labels)
        """
        batch_size = z_spec.size(0)
        device = z_spec.device
        
        # Initialize SSL labels (0: same, 1: different)
        ssl_labels = torch.zeros(batch_size, dtype=torch.long, device=device)
        
        # Apply FeatEx with probability prob
        mask = torch.rand(batch_size, device=device) < prob
        
        if mask.sum() > 0:
            # Get indices for exchange
            index = torch.randperm(batch_size, device=device)
            
            # Exchange spectrum embeddings
            z_spect_orig = z_spect.clone()
            z_spect[mask] = z_spect_orig[index][mask]
            
            # Set SSL labels (1 for exchanged samples)
            ssl_labels[mask] = 1
        
        # Concatenate embeddings
        z = torch.cat([z_spec, z_spect], dim=1)
        
        return z, ssl_labels
    
    def train_fkie(self, train_data, machine_types, attributes=None, 
                  optimizer=None, num_epochs=10, batch_size=32, 
                  mixup_prob=0.5, featex_prob=0.5, lambda_ssl=1.0):
        """
        Train the model using the FKIE-VUB approach
        
        Parameters:
        -----------
        train_data : dict
            Dictionary with 'spectrograms' and 'spectra' keys
        machine_types : torch.Tensor
            Machine type labels
        attributes : torch.Tensor or None
            Attribute labels
        optimizer : torch.optim.Optimizer or None
            Optimizer for training
        num_epochs : int
            Number of training epochs
        batch_size : int
            Batch size for training
        mixup_prob : float
            Probability of applying mixup
        featex_prob : float
            Probability of applying FeatEx
        lambda_ssl : float
            Weight for SSL loss
        """
        if optimizer is None:
            optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        
        device = next(self.parameters()).device
        
        # Move data to device
        spectrograms = train_data['spectrograms'].to(device)
        spectra = train_data['spectra'].to(device)
        machine_types = machine_types.to(device)
        if attributes is not None:
            attributes = attributes.to(device)
        
        # Compute sample weights
        sample_weights = self.compute_sample_weights(machine_types, attributes)
        
        # Create dataset
        dataset = torch.utils.data.TensorDataset(spectrograms, spectra, machine_types, 
                                               attributes if attributes is not None else torch.zeros_like(machine_types))
        
        for epoch in range(num_epochs):
            self.train()
            
            # Create dataloader
            dataloader = torch.utils.data.DataLoader(
                dataset, batch_size=batch_size, shuffle=True
            )
            
            total_loss = 0
            total_meta_loss = 0
            total_ssl_loss = 0
            
            for batch in dataloader:
                batch_spectrograms, batch_spectra, batch_machine_types, batch_attributes = batch
                
                # Apply mixup with probability mixup_prob
                if random.random() < mixup_prob:
                    # Create class labels (machine type + attribute combinations)
                    batch_labels = []
                    for i in range(len(batch_machine_types)):
                        machine_type = batch_machine_types[i].item()
                        attribute = batch_attributes[i].item() if attributes is not None else 0
                        # Map to class index
                        class_idx = 0
                        for j in range(machine_type):
                            class_idx += max(1, self.n_attributes[j])
                        class_idx += attribute
                        batch_labels.append(class_idx)
                    batch_labels = torch.tensor(batch_labels, device=device)
                    
                    # Apply mixup
                    batch_spectrograms, batch_labels, lam, index = self.apply_mixup(batch_spectrograms, batch_labels)
                    batch_spectra, _, _, _ = self.apply_mixup(batch_spectra, batch_labels, alpha=0)  # No mixup for spectra
                
                # Get embeddings from both branches
                z_spec = self.spectrogram_encoder(batch_spectrograms)
                z_spect = self.spectrum_encoder(batch_spectra)
                
                # Apply FeatEx for SSL
                z, ssl_labels = self.apply_featex(z_spec, z_spect, batch_labels if 'batch_labels' in locals() else None, 
                                                prob=featex_prob)
                
                # Get class labels if not already created by mixup
                if 'batch_labels' not in locals():
                    batch_labels = []
                    for i in range(len(batch_machine_types)):
                        machine_type = batch_machine_types[i].item()
                        attribute = batch_attributes[i].item() if attributes is not None else 0
                        # Map to class index
                        class_idx = 0
                        for j in range(machine_type):
                            class_idx += max(1, self.n_attributes[j])
                        class_idx += attribute
                        batch_labels.append(class_idx)
                    batch_labels = torch.tensor(batch_labels, device=device)
                
                # Compute meta loss (classification)
                meta_logits, meta_loss = self.adaproj(z, batch_labels)
                
                # Apply sample weights to meta loss
                if sample_weights is not None:
                    meta_loss = (meta_loss * sample_weights[batch_labels]).mean()
                
                # Compute SSL loss
                ssl_logits, ssl_loss = self.ssl_adaproj(z, ssl_labels)
                
                # Combine losses
                loss = meta_loss + lambda_ssl * ssl_loss
                
                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                total_meta_loss += meta_loss.item()
                total_ssl_loss += ssl_loss.item()
            
            # Print progress
            print(f'Epoch [{epoch+1}/{num_epochs}], '
                  f'Loss: {total_loss/len(dataloader):.6f}, '
                  f'Meta Loss: {total_meta_loss/len(dataloader):.6f}, '
                  f'SSL Loss: {total_ssl_loss/len(dataloader):.6f}')
        
        return self
    
    def extract_features(self, waveform, sample_rate=16000):
        """
        Extract spectrogram and spectrum features from waveform
        
        Parameters:
        -----------
        waveform : torch.Tensor
            Audio waveform
        sample_rate : int
            Sampling rate
            
        Returns:
        --------
        dict
            Dictionary with 'spectrogram' and 'spectrum' keys
        """
        device = waveform.device
        
        # Pad or truncate to 12 seconds
        target_length = 12 * sample_rate
        if waveform.size(-1) < target_length:
            padding = target_length - waveform.size(-1)
            waveform = F.pad(waveform, (padding // 2, padding - padding // 2))
        elif waveform.size(-1) > target_length:
            waveform = waveform[:, :target_length]
        
        # Extract spectrogram
        spectrogram = torchaudio.transforms.Spectrogram(
            n_fft=1024,
            hop_length=512,
            power=1.0  # magnitude spectrogram
        ).to(device)(waveform)
        
        # Apply temporal mean normalization
        mean = torch.mean(spectrogram, dim=2, keepdim=True)
        spectrogram = spectrogram - mean
        
        # Extract spectrum (magnitude of FFT)
        spectrum = torch.abs(torch.fft.rfft(waveform, dim=-1))
        
        # Keep only frequencies up to 8 kHz
        spectrum = spectrum[:, :int(8000 * waveform.size(-1) / sample_rate)]
        
        return {
            'spectrogram': spectrogram,
            'spectrum': spectrum
        }
