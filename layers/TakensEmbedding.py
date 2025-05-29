import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional, Tuple, Union
import warnings
from scipy.spatial.distance import pdist
from sklearn.feature_selection import mutual_info_regression


class TakensEmbedding(nn.Module):
    """
    Multi-scale Takens delay embedding with adaptive parameter selection
    
    Implements delay coordinate embedding (Takens' theorem) for time series analysis,
    transforming univariate time series into multi-dimensional phase space reconstructions
    that preserve topological properties of the underlying dynamical system.
    
    Mathematical Foundation:
    For a time series x(t), the delay embedding creates vectors:
    v_i = [x_i, x_{i+τ}, x_{i+2τ}, ..., x_{i+(d-1)τ}]
    
    where:
    - τ (tau) is the delay parameter
    - d is the embedding dimension
    - i ranges from 0 to T-(d-1)τ-1
    
    Parameters:
    -----------
    dims : List[int], default=[2, 3, 5, 10]
        Embedding dimensions to compute. Higher dimensions capture more complex dynamics.
    delays : List[int], default=[1, 2, 4, 8] 
        Delay parameters (in time steps). Should be chosen based on autocorrelation structure.
    strategy : str, default='multi_scale'
        Embedding strategy: 'single', 'multi_scale', 'adaptive'
        - 'single': Use first dim/delay pair only
        - 'multi_scale': Compute all dim/delay combinations
        - 'adaptive': Automatically select optimal parameters
    optimization_method : str, default='mutual_info'
        Parameter optimization method: 'mutual_info', 'fnn', 'autocorr'
        - 'mutual_info': Minimize mutual information
        - 'fnn': False nearest neighbors method
        - 'autocorr': First zero crossing of autocorrelation
    cache_embeddings : bool, default=True
        Whether to cache computed embeddings for repeated calls
    device : str, default='auto'
        Device to use for computations ('cpu', 'cuda', 'auto')
    """
    
    def __init__(self, 
                 dims: List[int] = [2, 3, 5, 10], 
                 delays: List[int] = [1, 2, 4, 8], 
                 strategy: str = 'multi_scale',
                 optimization_method: str = 'mutual_info',
                 cache_embeddings: bool = True,
                 device: str = 'auto'):
        super(TakensEmbedding, self).__init__()
        
        # Parameter validation
        self._validate_parameters(dims, delays, strategy, optimization_method)
        
        self.dims = dims
        self.delays = delays
        self.strategy = strategy
        self.optimization_method = optimization_method
        self.cache_embeddings = cache_embeddings
        
        # Device management
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Cache for embeddings and parameters
        self.embedding_cache = {} if cache_embeddings else None
        self.optimal_params_cache = {}
        
        # Precompute parameter combinations for multi_scale strategy
        if strategy == 'multi_scale':
            self.param_combinations = [(d, tau) for d in dims for tau in delays]
        else:
            self.param_combinations = [(dims[0], delays[0])]
        
        # Buffers for efficient computation
        self.register_buffer('_tau_range', torch.arange(max(dims) if dims else 10, device=self.device))
        
        # Statistics tracking
        self.embedding_stats = {
            'computation_count': 0,
            'cache_hits': 0,
            'total_embeddings': 0,
            'memory_usage': 0
        }
    
    def _validate_parameters(self, dims: List[int], delays: List[int], 
                           strategy: str, optimization_method: str):
        """Validate initialization parameters"""
        # Validate dims
        if not dims or not all(isinstance(d, int) and d >= 2 for d in dims):
            raise ValueError("dims must be a list of integers >= 2")
        if max(dims) > 50:
            warnings.warn("Large embedding dimensions (>50) may be computationally expensive")
        
        # Validate delays
        if not delays or not all(isinstance(tau, int) and tau >= 1 for tau in delays):
            raise ValueError("delays must be a list of positive integers")
        if max(delays) > 100:
            warnings.warn("Large delay values (>100) may reduce effective sample size significantly")
        
        # Validate strategy
        valid_strategies = ['single', 'multi_scale', 'adaptive']
        if strategy not in valid_strategies:
            raise ValueError(f"strategy must be one of {valid_strategies}")
        
        # Validate optimization method
        valid_methods = ['mutual_info', 'fnn', 'autocorr']
        if optimization_method not in valid_methods:
            raise ValueError(f"optimization_method must be one of {valid_methods}") 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute Takens delay embeddings for input time series
        
        Parameters:
        -----------
        x : torch.Tensor
            Input time series of shape [batch_size, seq_len] or [batch_size, seq_len, features]
            For multivariate input, each feature is embedded separately
        
        Returns:
        --------
        embeddings : torch.Tensor
            Embedded time series of shape [batch_size, n_embeddings, max_points, max_dim]
            where:
            - n_embeddings = number of (dim, delay) combinations
            - max_points = maximum number of embedding vectors
            - max_dim = maximum embedding dimension
        """
        # Input validation and preprocessing
        if x.dim() == 2:
            x = x.unsqueeze(-1)  # Add feature dimension: [batch, seq_len, 1]
        elif x.dim() == 3:
            pass  # Already in correct format: [batch, seq_len, features]
        else:
            raise ValueError(f"Input must be 2D or 3D tensor, got {x.dim()}D")
        
        batch_size, seq_len, n_features = x.shape
        x = x.to(self.device)
        
        # Generate cache key for this input configuration
        cache_key = self._generate_cache_key(x.shape, x.dtype) if self.cache_embeddings else None
        
        # Check cache first
        if self.cache_embeddings and cache_key in self.embedding_cache:
            self.embedding_stats['cache_hits'] += 1
            return self.embedding_cache[cache_key]
        
        # Strategy-based embedding computation
        if self.strategy == 'adaptive':
            # Optimize parameters for each feature separately
            embeddings = self._adaptive_embedding(x)
        elif self.strategy == 'multi_scale':
            # Compute all parameter combinations
            embeddings = self._multi_scale_embedding(x)
        else:  # strategy == 'single'
            # Use first parameter combination only
            dim, tau = self.param_combinations[0]
            embeddings = self._single_embedding(x, dim, tau)
            embeddings = embeddings.unsqueeze(1)  # Add embedding dimension
        
        # Cache result if enabled
        if self.cache_embeddings and cache_key is not None:
            self.embedding_cache[cache_key] = embeddings.detach()
        
        # Update statistics
        self.embedding_stats['computation_count'] += 1
        self.embedding_stats['total_embeddings'] += embeddings.shape[1]
        self.embedding_stats['memory_usage'] = embeddings.numel() * embeddings.element_size()
        
        return embeddings
    
    def _single_embedding(self, x: torch.Tensor, dim: int, tau: int) -> torch.Tensor:
        """
        Compute single Takens embedding with specified dimension and delay
        
        Mathematical Implementation:
        For each time series x and each starting point i:
        v_i = [x[i], x[i+τ], x[i+2τ], ..., x[i+(d-1)τ]]
        
        Parameters:
        -----------
        x : torch.Tensor
            Input time series [batch_size, seq_len, features]
        dim : int
            Embedding dimension
        tau : int
            Delay parameter
        
        Returns:
        --------
        embedding : torch.Tensor
            Embedded vectors [batch_size, n_points, dim] where n_points = seq_len - (dim-1)*tau
        """
        batch_size, seq_len, n_features = x.shape
        
        # Calculate number of embedding vectors
        n_points = seq_len - (dim - 1) * tau
        if n_points <= 0:
            raise ValueError(f"Insufficient data: seq_len={seq_len}, dim={dim}, tau={tau} "
                           f"requires at least {(dim-1)*tau + 1} time points")
        
        # Memory-efficient embedding computation using advanced indexing
        if self.device.type == 'cuda' and n_points * dim > 10000:
            # GPU-optimized version for large embeddings
            embedding = self._gpu_optimized_embedding(x, dim, tau, n_points)
        else:
            # Standard version for smaller embeddings or CPU
            embedding = self._cpu_embedding(x, dim, tau, n_points)
        
        return embedding
    
    def _cpu_embedding(self, x: torch.Tensor, dim: int, tau: int, n_points: int) -> torch.Tensor:
        """CPU-optimized embedding computation"""
        batch_size, seq_len, n_features = x.shape
        
        # Create index tensor for delay coordinates
        indices = torch.arange(n_points, device=self.device).unsqueeze(1)  # [n_points, 1]
        delays = torch.arange(dim, device=self.device).unsqueeze(0) * tau    # [1, dim]
        embedding_indices = indices + delays  # Broadcasting: [n_points, dim]
        
        # Extract embedding vectors for all features
        embeddings = []
        for feature_idx in range(n_features):
            # Extract delay coordinates: [batch_size, n_points, dim]
            feature_embedding = x[:, embedding_indices, feature_idx]
            embeddings.append(feature_embedding)
        
        # Concatenate features: [batch_size, n_points, dim * n_features]
        if n_features > 1:
            embedding = torch.cat(embeddings, dim=-1)
        else:
            embedding = embeddings[0]
        
        return embedding
    
    def _gpu_optimized_embedding(self, x: torch.Tensor, dim: int, tau: int, n_points: int) -> torch.Tensor:
        """GPU-optimized embedding computation using tensor operations"""
        batch_size, seq_len, n_features = x.shape
        
        # Preallocate output tensor
        embedding = torch.zeros(batch_size, n_points, dim * n_features, 
                              device=self.device, dtype=x.dtype)
        
        # Sliding window approach with vectorized operations
        for d in range(dim):
            start_idx = d * tau
            end_idx = start_idx + n_points
            for feature_idx in range(n_features):
                col_idx = feature_idx * dim + d
                embedding[:, :, col_idx] = x[:, start_idx:end_idx, feature_idx]
        
        return embedding 

    def _multi_scale_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute multiple embeddings at different scales (all dim/delay combinations)
        
        This creates a comprehensive multi-scale representation by computing embeddings
        with different parameters to capture patterns at various temporal scales.
        
        Parameters:
        -----------
        x : torch.Tensor
            Input time series [batch_size, seq_len, features]
        
        Returns:
        --------
        embeddings : torch.Tensor
            Multi-scale embeddings [batch_size, n_combinations, max_points, max_dim]
        """
        batch_size, seq_len, n_features = x.shape
        n_combinations = len(self.param_combinations)
        
        # Calculate maximum dimensions for tensor allocation
        max_dim = max(dim for dim, _ in self.param_combinations) * n_features
        max_points = min(seq_len - (dim - 1) * tau 
                        for dim, tau in self.param_combinations)
        
        if max_points <= 0:
            raise ValueError("No valid embeddings possible with current parameters and sequence length")
        
        # Preallocate output tensor
        embeddings = torch.zeros(batch_size, n_combinations, max_points, max_dim, 
                               device=self.device, dtype=x.dtype)
        
        # Compute each embedding
        for i, (dim, tau) in enumerate(self.param_combinations):
            try:
                # Compute single embedding
                single_emb = self._single_embedding(x, dim, tau)
                n_points = single_emb.shape[1]
                embedding_dim = single_emb.shape[2]
                
                # Store in output tensor (truncate to max_points if needed)
                actual_points = min(n_points, max_points)
                embeddings[:, i, :actual_points, :embedding_dim] = single_emb[:, :actual_points, :]
                
            except ValueError as e:
                warnings.warn(f"Skipping embedding (dim={dim}, tau={tau}): {str(e)}")
                continue
        
        return embeddings
    
    def _adaptive_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Adaptive parameter selection using optimization methods
        
        Automatically determines optimal embedding parameters for each feature
        based on the specified optimization method (mutual_info, fnn, autocorr).
        
        Parameters:
        -----------
        x : torch.Tensor
            Input time series [batch_size, seq_len, features]
        
        Returns:
        --------
        embeddings : torch.Tensor
            Adaptively optimized embeddings [batch_size, n_features, max_points, optimal_dim]
        """
        batch_size, seq_len, n_features = x.shape
        
        # Optimize parameters for each feature
        optimal_embeddings = []
        
        for feature_idx in range(n_features):
            # Extract single feature time series
            feature_data = x[:, :, feature_idx]  # [batch_size, seq_len]
            
            # Generate cache key for this feature
            feature_key = f"adaptive_{feature_idx}_{seq_len}_{x.dtype}"
            
            if feature_key in self.optimal_params_cache:
                optimal_dim, optimal_tau = self.optimal_params_cache[feature_key]
            else:
                # Optimize parameters for this feature
                optimal_dim, optimal_tau = self._optimize_parameters(feature_data[0])  # Use first sample
                self.optimal_params_cache[feature_key] = (optimal_dim, optimal_tau)
            
            # Compute embedding with optimal parameters
            feature_x = feature_data.unsqueeze(-1)  # Add feature dimension
            optimal_embedding = self._single_embedding(feature_x, optimal_dim, optimal_tau)
            optimal_embeddings.append(optimal_embedding)
        
        # Stack embeddings from all features
        if len(optimal_embeddings) > 1:
            # Find common dimensions
            min_points = min(emb.shape[1] for emb in optimal_embeddings)
            max_dim = max(emb.shape[2] for emb in optimal_embeddings)
            
            # Standardize dimensions and concatenate
            standardized_embeddings = []
            for emb in optimal_embeddings:
                # Truncate to min_points and pad dimension if needed
                trunc_emb = emb[:, :min_points, :]
                if trunc_emb.shape[2] < max_dim:
                    padding = torch.zeros(batch_size, min_points, max_dim - trunc_emb.shape[2], 
                                        device=self.device, dtype=x.dtype)
                    trunc_emb = torch.cat([trunc_emb, padding], dim=2)
                standardized_embeddings.append(trunc_emb.unsqueeze(1))
            
            embeddings = torch.cat(standardized_embeddings, dim=1)
        else:
            embeddings = optimal_embeddings[0].unsqueeze(1)
        
        return embeddings
    
    def _optimize_parameters(self, x: torch.Tensor) -> Tuple[int, int]:
        """
        Automatic parameter optimization using specified method
        
        Parameters:
        -----------
        x : torch.Tensor
            Single time series [seq_len]
        
        Returns:
        --------
        optimal_dim : int
            Optimal embedding dimension
        optimal_tau : int
            Optimal delay parameter
        """
        if self.optimization_method == 'mutual_info':
            return self._optimize_mutual_info(x)
        elif self.optimization_method == 'fnn':
            return self._optimize_false_nearest_neighbors(x)
        elif self.optimization_method == 'autocorr':
            return self._optimize_autocorrelation(x)
        else:
            # Fallback to first parameter combination
            return self.dims[0], self.delays[0]
    
    def _optimize_mutual_info(self, x: torch.Tensor) -> Tuple[int, int]:
        """
        Optimize parameters using mutual information criterion
        
        The optimal delay minimizes mutual information between x(t) and x(t+τ),
        indicating independence and good phase space reconstruction.
        """
        x_np = x.detach().cpu().numpy()
        
        # Test different delay values
        mi_scores = []
        test_delays = self.delays[:5]  # Limit to first 5 delays for efficiency
        
        for tau in test_delays:
            if len(x_np) <= tau:
                mi_scores.append(float('inf'))
                continue
            
            # Compute delayed series
            x1 = x_np[:-tau]
            x2 = x_np[tau:]
            
            # Compute mutual information (lower is better)
            try:
                # Reshape for sklearn
                x1_reshaped = x1.reshape(-1, 1)
                mi = mutual_info_regression(x1_reshaped, x2, random_state=42)[0]
                mi_scores.append(mi)
            except:
                mi_scores.append(float('inf'))
        
        # Select delay with minimum mutual information
        optimal_tau_idx = np.argmin(mi_scores)
        optimal_tau = test_delays[optimal_tau_idx]
        
        # For dimension, use False Nearest Neighbors method
        optimal_dim = self._estimate_dimension_fnn(x, optimal_tau)
        
        return optimal_dim, optimal_tau
    
    def _optimize_false_nearest_neighbors(self, x: torch.Tensor) -> Tuple[int, int]:
        """
        Optimize parameters using False Nearest Neighbors method
        
        This method finds the minimum embedding dimension where the number
        of false nearest neighbors drops below a threshold.
        """
        x_np = x.detach().cpu().numpy()
        
        # First, estimate optimal delay using autocorrelation
        optimal_tau = self._estimate_delay_autocorr(x_np)
        
        # Then estimate dimension using FNN
        optimal_dim = self._estimate_dimension_fnn(x, optimal_tau)
        
        return optimal_dim, optimal_tau
    
    def _optimize_autocorrelation(self, x: torch.Tensor) -> Tuple[int, int]:
        """
        Optimize delay using first zero crossing of autocorrelation function
        """
        x_np = x.detach().cpu().numpy()
        optimal_tau = self._estimate_delay_autocorr(x_np)
        
        # Use middle dimension from available options
        optimal_dim = self.dims[len(self.dims) // 2]
        
        return optimal_dim, optimal_tau

    def _estimate_dimension_fnn(self, x: torch.Tensor, tau: int) -> int:
        """
        Estimate embedding dimension using False Nearest Neighbors method
        
        Parameters:
        -----------
        x : torch.Tensor
            Input time series [seq_len]
        tau : int
            Delay parameter
        
        Returns:
        --------
        optimal_dim : int
            Estimated optimal embedding dimension
        """
        x_np = x.detach().cpu().numpy()
        
        # Test different embedding dimensions
        max_test_dim = min(10, len(self.dims))
        fnn_ratios = []
        
        for test_dim in range(2, max_test_dim + 2):
            if len(x_np) < (test_dim - 1) * tau + 1:
                fnn_ratios.append(1.0)  # High FNN ratio for insufficient data
                continue
            
            # Create embedding with current dimension
            n_points = len(x_np) - (test_dim - 1) * tau
            embedding = np.zeros((n_points, test_dim))
            
            for i in range(n_points):
                for j in range(test_dim):
                    embedding[i, j] = x_np[i + j * tau]
            
            # Compute FNN ratio
            try:
                distances = pdist(embedding)
                threshold = np.percentile(distances, 5)  # Use 5th percentile as threshold
                fnn_ratio = np.mean(distances < threshold)
                fnn_ratios.append(fnn_ratio)
            except:
                fnn_ratios.append(1.0)
        
        # Find dimension where FNN ratio stabilizes (drops below threshold)
        optimal_dim = 2
        for i, ratio in enumerate(fnn_ratios):
            if ratio < 0.1:  # Threshold for acceptable FNN ratio
                optimal_dim = i + 2
                break
        
        # Ensure dimension is within available options
        optimal_dim = min(optimal_dim, max(self.dims))
        
        return optimal_dim

    def _estimate_delay_autocorr(self, x: np.ndarray) -> int:
        """
        Estimate delay using first zero crossing of autocorrelation function
        
        Parameters:
        -----------
        x : np.ndarray
            Input time series
        
        Returns:
        --------
        optimal_tau : int
            Estimated optimal delay parameter
        """
        # Compute autocorrelation
        autocorr = np.correlate(x, x, mode='full')
        
        # Find first zero crossing
        autocorr_np = autocorr[len(x)-1:]
        
        # Normalize autocorrelation
        autocorr_np = autocorr_np / autocorr_np[0]
        
        # Find first zero crossing or when autocorr drops below threshold
        optimal_tau_idx = 1
        for i in range(1, min(len(autocorr_np), 50)):  # Limit search to first 50 lags
            if autocorr_np[i] <= 0.1:  # When autocorrelation drops to 10%
                optimal_tau_idx = i
                break
        
        # Ensure delay is within available options
        optimal_tau = min(optimal_tau_idx, max(self.delays))
        optimal_tau = max(optimal_tau, 1)  # Ensure at least 1
        
        return optimal_tau

    def _generate_cache_key(self, shape: Tuple[int, int, int], dtype: torch.dtype) -> str:
        """
        Generate a unique cache key for a given input shape and dtype
        
        Parameters:
        -----------
        shape : Tuple[int, int, int]
            Input tensor shape (batch_size, seq_len, n_features)
        dtype : torch.dtype
            Input tensor data type
        
        Returns:
        --------
        cache_key : str
            Unique cache key
        """
        return f"{shape}-{dtype}-{self.strategy}-{tuple(self.dims)}-{tuple(self.delays)}"

    def _compute_embedding_quality(self, x: torch.Tensor, embedding: torch.Tensor) -> dict:
        """
        Compute embedding quality metrics for validation
        
        Parameters:
        -----------
        x : torch.Tensor
            Original time series [batch_size, seq_len, features]
        embedding : torch.Tensor
            Embedded time series [batch_size, n_points, dim]
        
        Returns:
        --------
        quality_metrics : dict
            Dictionary containing various quality metrics
        """
        batch_size, n_points, dim = embedding.shape
        
        # Convert to numpy for computation
        x_np = x.detach().cpu().numpy()
        emb_np = embedding.detach().cpu().numpy()
        
        quality_metrics = {}
        
        try:
            # Average across batch for metrics
            emb_sample = emb_np[0]  # Use first sample
            
            # 1. Local correlation dimension (estimate of fractal dimension)
            quality_metrics['correlation_dimension'] = self._estimate_correlation_dimension(emb_sample)
            
            # 2. Largest Lyapunov exponent estimate
            quality_metrics['lyapunov_exponent'] = self._estimate_lyapunov_exponent(emb_sample)
            
            # 3. Embedding density (measure of space utilization)
            quality_metrics['embedding_density'] = self._compute_embedding_density(emb_sample)
            
            # 4. Reconstruction error (for known systems)
            quality_metrics['reconstruction_error'] = float(torch.mean(torch.abs(embedding - embedding.mean(dim=1, keepdim=True))))
            
            # 5. Determinism measure
            quality_metrics['determinism'] = self._compute_determinism(emb_sample)
            
        except Exception as e:
            warnings.warn(f"Error computing embedding quality: {str(e)}")
            quality_metrics = {'error': str(e)}
        
        return quality_metrics
    
    def _estimate_correlation_dimension(self, embedding: np.ndarray) -> float:
        """Estimate correlation dimension of the embedding"""
        try:
            # Compute pairwise distances
            distances = pdist(embedding)
            
            # Use correlation sum method
            r_values = np.logspace(-2, 1, 20)
            correlation_sums = []
            
            for r in r_values:
                correlation_sum = np.mean(distances < r)
                correlation_sums.append(correlation_sum + 1e-10)  # Avoid log(0)
            
            # Estimate dimension from slope
            log_r = np.log(r_values)
            log_c = np.log(correlation_sums)
            
            # Linear regression on middle portion
            mid_start, mid_end = len(log_r) // 4, 3 * len(log_r) // 4
            slope, _ = np.polyfit(log_r[mid_start:mid_end], log_c[mid_start:mid_end], 1)
            
            return float(slope)
        except:
            return np.nan
    
    def _estimate_lyapunov_exponent(self, embedding: np.ndarray) -> float:
        """Estimate largest Lyapunov exponent"""
        try:
            n_points, dim = embedding.shape
            
            # Simple estimation using nearest neighbors
            distances = pdist(embedding)
            min_distance = np.min(distances[distances > 0])
            max_distance = np.max(distances)
            
            # Rough estimate
            lyapunov_estimate = np.log(max_distance / min_distance) / n_points
            
            return float(lyapunov_estimate)
        except:
            return np.nan
    
    def _compute_embedding_density(self, embedding: np.ndarray) -> float:
        """Compute space utilization density of embedding"""
        try:
            n_points, dim = embedding.shape
            
            # Compute volume occupied by points
            ranges = np.max(embedding, axis=0) - np.min(embedding, axis=0)
            total_volume = np.prod(ranges)
            
            # Estimate effective volume using nearest neighbor distances
            if n_points > 1:
                distances = pdist(embedding)
                avg_distance = np.mean(distances)
                effective_volume = n_points * (avg_distance ** dim)
                
                density = effective_volume / (total_volume + 1e-10)
            else:
                density = 0.0
            
            return float(density)
        except:
            return np.nan
    
    def _compute_determinism(self, embedding: np.ndarray) -> float:
        """Compute determinism measure based on recurrence"""
        try:
            n_points = len(embedding)
            
            # Compute recurrence matrix
            threshold = np.percentile(pdist(embedding), 10)
            recurrence_matrix = pdist(embedding) < threshold
            
            # Count diagonal lines (deterministic structure)
            min_line_length = 2
            diagonal_lines = 0
            
            # Simple diagonal line counting
            for i in range(len(recurrence_matrix) - min_line_length):
                if recurrence_matrix[i] and recurrence_matrix[i + 1]:
                    diagonal_lines += 1
            
            determinism = diagonal_lines / len(recurrence_matrix)
            return float(determinism)
        except:
            return np.nan
    
    def clear_cache(self):
        """Clear embedding and parameter caches"""
        if self.embedding_cache is not None:
            self.embedding_cache.clear()
        self.optimal_params_cache.clear()
        
        # Reset statistics
        self.embedding_stats = {
            'computation_count': 0,
            'cache_hits': 0,
            'total_embeddings': 0,
            'memory_usage': 0
        }
    
    def get_statistics(self) -> dict:
        """Get computation statistics"""
        stats = self.embedding_stats.copy()
        stats['cache_hit_rate'] = (stats['cache_hits'] / max(stats['computation_count'], 1)) * 100
        stats['memory_usage_mb'] = stats['memory_usage'] / (1024 * 1024)
        return stats
    
    def get_optimal_parameters(self) -> dict:
        """Get cached optimal parameters"""
        return self.optimal_params_cache.copy()
    
    def set_device(self, device: Union[str, torch.device]):
        """Move module to specified device"""
        if isinstance(device, str):
            device = torch.device(device)
        
        self.device = device
        self.to(device)
        
        # Update buffer
        self._tau_range = self._tau_range.to(device)
    
    def __repr__(self) -> str:
        """String representation of the module"""
        return (f"TakensEmbedding(\n"
                f"  dims={self.dims},\n"
                f"  delays={self.delays},\n"
                f"  strategy='{self.strategy}',\n"
                f"  optimization_method='{self.optimization_method}',\n"
                f"  device={self.device},\n"
                f"  param_combinations={len(self.param_combinations)}\n"
                f")")


# Utility functions for external use
def compute_takens_embedding(x: torch.Tensor, 
                           dim: int = 3, 
                           tau: int = 1, 
                           device: str = 'auto') -> torch.Tensor:
    """
    Convenient function to compute single Takens embedding
    
    Parameters:
    -----------
    x : torch.Tensor
        Input time series [batch_size, seq_len] or [seq_len]
    dim : int, default=3
        Embedding dimension
    tau : int, default=1
        Delay parameter
    device : str, default='auto'
        Device for computation
    
    Returns:
    --------
    embedding : torch.Tensor
        Takens embedding [batch_size, n_points, dim]
    """
    # Create temporary TakensEmbedding instance
    embedder = TakensEmbedding(dims=[dim], delays=[tau], strategy='single', device=device)
    
    # Handle 1D input
    if x.dim() == 1:
        x = x.unsqueeze(0)  # Add batch dimension
    
    # Compute embedding
    embedding = embedder(x)
    
    # Remove extra dimensions and return
    return embedding.squeeze(1)  # Remove embedding combination dimension


def estimate_embedding_parameters(x: torch.Tensor, 
                                method: str = 'mutual_info',
                                device: str = 'auto') -> Tuple[int, int]:
    """
    Estimate optimal embedding parameters for time series
    
    Parameters:
    -----------
    x : torch.Tensor
        Input time series [seq_len]
    method : str, default='mutual_info'
        Optimization method ('mutual_info', 'fnn', 'autocorr')
    device : str, default='auto'
        Device for computation
    
    Returns:
    --------
    optimal_dim : int
        Estimated optimal embedding dimension
    optimal_tau : int
        Estimated optimal delay parameter
    """
    # Create temporary TakensEmbedding instance
    embedder = TakensEmbedding(optimization_method=method, device=device)
    
    # Estimate parameters
    optimal_dim, optimal_tau = embedder._optimize_parameters(x)
    
    return optimal_dim, optimal_tau 