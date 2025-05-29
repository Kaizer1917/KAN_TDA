import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional, Tuple, Union, Dict, Any
import warnings
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp

# Optional imports for different backends
try:
    import ripser
    RIPSER_AVAILABLE = True
except ImportError:
    RIPSER_AVAILABLE = False
    warnings.warn("Ripser not available. Install with: pip install ripser")

try:
    import gudhi
    GUDHI_AVAILABLE = True
except ImportError:
    GUDHI_AVAILABLE = False
    warnings.warn("GUDHI not available. Install with: pip install gudhi")

try:
    from gtda.homology import VietorisRipsPersistence
    from gtda.diagrams import PersistenceEntropy, Amplitude, NumberOfPoints
    GIOTTO_AVAILABLE = True
except ImportError:
    GIOTTO_AVAILABLE = False
    warnings.warn("Giotto-tda not available. Install with: pip install giotto-tda")


class PersistenceBackend(ABC):
    """Abstract base class for persistent homology computation backends"""
    
    @abstractmethod
    def compute_persistence(self, 
                          point_cloud: np.ndarray, 
                          max_dimension: int = 2,
                          **kwargs) -> List[np.ndarray]:
        """
        Compute persistence diagrams for a point cloud
        
        Parameters:
        -----------
        point_cloud : np.ndarray
            Point cloud data of shape [n_points, n_features]
        max_dimension : int
            Maximum homological dimension to compute
        **kwargs : dict
            Backend-specific parameters
        
        Returns:
        --------
        diagrams : List[np.ndarray]
            Persistence diagrams for each dimension
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Return backend name"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if backend is available"""
        pass


class RipserBackend(PersistenceBackend):
    """Ripser backend for fast persistent homology computation"""
    
    def __init__(self, metric: str = 'euclidean', n_perm: Optional[int] = None):
        """
        Initialize Ripser backend
        
        Parameters:
        -----------
        metric : str
            Distance metric ('euclidean', 'manhattan', 'cosine', etc.)
        n_perm : int, optional
            Number of permutations for sparsification (None for no limit)
        """
        self.metric = metric
        self.n_perm = n_perm
    
    def compute_persistence(self, 
                          point_cloud: np.ndarray, 
                          max_dimension: int = 2,
                          maxdim: Optional[int] = None,
                          thresh: Optional[float] = None,
                          **kwargs) -> List[np.ndarray]:
        """Compute persistence using Ripser"""
        if not self.is_available():
            raise RuntimeError("Ripser not available")
        
        # Use maxdim parameter if provided, otherwise use max_dimension
        actual_maxdim = maxdim if maxdim is not None else max_dimension
        
        # Adjust n_perm based on point cloud size
        n_points = point_cloud.shape[0]
        if self.n_perm is None or self.n_perm > n_points:
            # Use all points if n_perm is too large or None
            effective_n_perm = None
        else:
            effective_n_perm = self.n_perm
        
        # Prepare parameters for ripser
        ripser_params = {
            'maxdim': actual_maxdim,
            'metric': self.metric
        }
        
        # Add optional parameters only if they are not None
        if effective_n_perm is not None:
            ripser_params['n_perm'] = effective_n_perm
        
        if thresh is not None:
            ripser_params['thresh'] = thresh
        
        # Add any additional kwargs
        ripser_params.update(kwargs)
        
        # Compute persistence diagrams
        result = ripser.ripser(point_cloud, **ripser_params)
        
        # Extract diagrams
        diagrams = result['dgms']
        
        # Ensure we have diagrams for all requested dimensions
        while len(diagrams) <= max_dimension:
            diagrams.append(np.empty((0, 2)))
        
        return diagrams[:max_dimension + 1]
    
    def get_name(self) -> str:
        return "ripser"
    
    def is_available(self) -> bool:
        return RIPSER_AVAILABLE


class GUDHIBackend(PersistenceBackend):
    """GUDHI backend for advanced persistent homology features"""
    
    def __init__(self, metric: str = 'euclidean', max_edge_length: Optional[float] = None):
        """
        Initialize GUDHI backend
        
        Parameters:
        -----------
        metric : str
            Distance metric
        max_edge_length : float, optional
            Maximum edge length for simplicial complex
        """
        self.metric = metric
        self.max_edge_length = max_edge_length
    
    def compute_persistence(self, 
                          point_cloud: np.ndarray, 
                          max_dimension: int = 2,
                          max_edge_length: Optional[float] = None,
                          **kwargs) -> List[np.ndarray]:
        """Compute persistence using GUDHI"""
        if not self.is_available():
            raise RuntimeError("GUDHI not available")
        
        # Use provided max_edge_length or compute a reasonable default
        if max_edge_length is not None:
            edge_length = max_edge_length
        elif self.max_edge_length is not None:
            edge_length = self.max_edge_length
        else:
            # Compute a reasonable default based on data
            from scipy.spatial.distance import pdist
            distances = pdist(point_cloud)
            edge_length = np.percentile(distances, 90)  # Use 90th percentile
        
        # Create Rips complex
        rips_complex = gudhi.RipsComplex(
            points=point_cloud,
            max_edge_length=edge_length
        )
        
        # Create simplex tree
        simplex_tree = rips_complex.create_simplex_tree(max_dimension=max_dimension + 1)
        
        # Compute persistence
        simplex_tree.compute_persistence()
        
        # Extract diagrams by dimension
        diagrams = [[] for _ in range(max_dimension + 1)]
        
        for persistence_pair in simplex_tree.persistence():
            dimension = persistence_pair[0]
            birth = persistence_pair[1][0]
            death = persistence_pair[1][1]
            
            if dimension <= max_dimension:
                diagrams[dimension].append([birth, death])
        
        # Convert to numpy arrays
        diagrams = [np.array(diag) if diag else np.empty((0, 2)) for diag in diagrams]
        
        return diagrams
    
    def get_name(self) -> str:
        return "gudhi"
    
    def is_available(self) -> bool:
        return GUDHI_AVAILABLE


class GiottoBackend(PersistenceBackend):
    """Giotto-tda backend with sklearn-compatible interface"""
    
    def __init__(self, metric: str = 'euclidean', max_edge_length: float = np.inf):
        """
        Initialize Giotto-tda backend
        
        Parameters:
        -----------
        metric : str
            Distance metric
        max_edge_length : float
            Maximum edge length for Vietoris-Rips complex
        """
        self.metric = metric
        self.max_edge_length = max_edge_length
    
    def compute_persistence(self, 
                          point_cloud: np.ndarray, 
                          max_dimension: int = 2,
                          **kwargs) -> List[np.ndarray]:
        """Compute persistence using Giotto-tda"""
        if not self.is_available():
            raise RuntimeError("Giotto-tda not available")
        
        # Create VietorisRips transformer
        vr = VietorisRipsPersistence(
            metric=self.metric,
            max_edge_length=self.max_edge_length,
            homology_dimensions=list(range(max_dimension + 1)),
            n_jobs=1
        )
        
        # Reshape for Giotto-tda (expects 3D array)
        point_cloud_3d = point_cloud.reshape(1, *point_cloud.shape)
        
        # Compute persistence
        diagrams_3d = vr.fit_transform(point_cloud_3d)
        
        # Extract diagrams for single point cloud
        diagrams_2d = diagrams_3d[0]
        
        # Separate by dimension
        diagrams = [[] for _ in range(max_dimension + 1)]
        
        for point in diagrams_2d:
            dimension = int(point[2])  # Third column is dimension
            birth = point[0]
            death = point[1]
            
            if dimension <= max_dimension:
                diagrams[dimension].append([birth, death])
        
        # Convert to numpy arrays
        diagrams = [np.array(diag) if diag else np.empty((0, 2)) for diag in diagrams]
        
        return diagrams
    
    def get_name(self) -> str:
        return "giotto"
    
    def is_available(self) -> bool:
        return GIOTTO_AVAILABLE 


class PersistenceDiagram:
    """
    Class for manipulating and analyzing persistence diagrams
    """
    
    def __init__(self, diagrams: List[np.ndarray], dimensions: Optional[List[int]] = None):
        """
        Initialize persistence diagram
        
        Parameters:
        -----------
        diagrams : List[np.ndarray]
            List of persistence diagrams, one per dimension
        dimensions : List[int], optional
            Dimension labels for each diagram
        """
        self.diagrams = diagrams
        self.dimensions = dimensions if dimensions is not None else list(range(len(diagrams)))
        self.n_dimensions = len(diagrams)
    
    def filter_by_persistence(self, threshold: float = 0.01) -> 'PersistenceDiagram':
        """
        Filter features by minimum persistence threshold
        
        Parameters:
        -----------
        threshold : float
            Minimum persistence (death - birth) to keep feature
        
        Returns:
        --------
        filtered_diagram : PersistenceDiagram
            Filtered persistence diagram
        """
        filtered_diagrams = []
        
        for diagram in self.diagrams:
            if len(diagram) == 0:
                filtered_diagrams.append(diagram)
                continue
            
            # Calculate persistence
            persistence = diagram[:, 1] - diagram[:, 0]
            
            # Filter by threshold
            mask = persistence >= threshold
            filtered_diagram = diagram[mask]
            
            filtered_diagrams.append(filtered_diagram)
        
        return PersistenceDiagram(filtered_diagrams, self.dimensions)
    
    def stable_features(self, noise_level: float = 0.05) -> 'PersistenceDiagram':
        """
        Extract stable features (high persistence relative to noise)
        
        Parameters:
        -----------
        noise_level : float
            Expected noise level in the data
        
        Returns:
        --------
        stable_diagram : PersistenceDiagram
            Diagram with only stable features
        """
        return self.filter_by_persistence(threshold=noise_level)
    
    def diagram_statistics(self) -> Dict[str, Any]:
        """
        Compute statistical summary of persistence diagrams
        
        Returns:
        --------
        stats : Dict[str, Any]
            Dictionary containing various statistics
        """
        stats = {
            'dimensions': self.dimensions,
            'n_features_per_dim': [],
            'max_persistence_per_dim': [],
            'mean_persistence_per_dim': [],
            'total_persistence_per_dim': [],
            'feature_counts': {},
            'persistence_ranges': {}
        }
        
        for i, (dim, diagram) in enumerate(zip(self.dimensions, self.diagrams)):
            if len(diagram) == 0:
                stats['n_features_per_dim'].append(0)
                stats['max_persistence_per_dim'].append(0.0)
                stats['mean_persistence_per_dim'].append(0.0)
                stats['total_persistence_per_dim'].append(0.0)
                stats['feature_counts'][dim] = 0
                stats['persistence_ranges'][dim] = (0.0, 0.0)
                continue
            
            # Calculate persistence
            persistence = diagram[:, 1] - diagram[:, 0]
            
            # Remove infinite persistence (for essential classes)
            finite_persistence = persistence[np.isfinite(persistence)]
            
            stats['n_features_per_dim'].append(len(diagram))
            stats['max_persistence_per_dim'].append(float(np.max(finite_persistence)) if len(finite_persistence) > 0 else 0.0)
            stats['mean_persistence_per_dim'].append(float(np.mean(finite_persistence)) if len(finite_persistence) > 0 else 0.0)
            stats['total_persistence_per_dim'].append(float(np.sum(finite_persistence)) if len(finite_persistence) > 0 else 0.0)
            stats['feature_counts'][dim] = len(diagram)
            
            if len(finite_persistence) > 0:
                stats['persistence_ranges'][dim] = (float(np.min(finite_persistence)), float(np.max(finite_persistence)))
            else:
                stats['persistence_ranges'][dim] = (0.0, 0.0)
        
        # Overall statistics
        all_persistence = []
        for diagram in self.diagrams:
            if len(diagram) > 0:
                persistence = diagram[:, 1] - diagram[:, 0]
                finite_persistence = persistence[np.isfinite(persistence)]
                all_persistence.extend(finite_persistence)
        
        if all_persistence:
            stats['overall_max_persistence'] = float(np.max(all_persistence))
            stats['overall_mean_persistence'] = float(np.mean(all_persistence))
            stats['overall_std_persistence'] = float(np.std(all_persistence))
            stats['total_features'] = len(all_persistence)
        else:
            stats['overall_max_persistence'] = 0.0
            stats['overall_mean_persistence'] = 0.0
            stats['overall_std_persistence'] = 0.0
            stats['total_features'] = 0
        
        return stats
    
    def to_tensor(self, device: str = 'cpu') -> torch.Tensor:
        """
        Convert persistence diagrams to PyTorch tensor
        
        Parameters:
        -----------
        device : str
            Device for tensor ('cpu' or 'cuda')
        
        Returns:
        --------
        tensor : torch.Tensor
            Concatenated persistence diagrams as tensor
        """
        # Concatenate all diagrams with dimension labels
        all_points = []
        
        for dim, diagram in zip(self.dimensions, self.diagrams):
            if len(diagram) > 0:
                # Add dimension as third column
                dim_column = np.full((len(diagram), 1), dim)
                diagram_with_dim = np.concatenate([diagram, dim_column], axis=1)
                all_points.append(diagram_with_dim)
        
        if all_points:
            concatenated = np.concatenate(all_points, axis=0)
        else:
            concatenated = np.empty((0, 3))
        
        return torch.tensor(concatenated, dtype=torch.float32, device=device)
    
    def __len__(self) -> int:
        """Return total number of persistence features"""
        return sum(len(diagram) for diagram in self.diagrams)
    
    def __getitem__(self, dim: int) -> np.ndarray:
        """Get persistence diagram for specific dimension"""
        if dim in self.dimensions:
            idx = self.dimensions.index(dim)
            return self.diagrams[idx]
        else:
            raise KeyError(f"Dimension {dim} not found in diagrams")


class PersistentHomologyComputer(nn.Module):
    """
    Main class for computing persistent homology with multiple backend support
    """
    
    def __init__(self,
                 backend: str = 'auto',
                 max_dimension: int = 2,
                 metric: str = 'euclidean',
                 n_jobs: int = 1,
                 backend_params: Optional[Dict[str, Any]] = None):
        """
        Initialize persistent homology computer
        
        Parameters:
        -----------
        backend : str
            Backend to use ('ripser', 'gudhi', 'giotto', 'auto')
        max_dimension : int
            Maximum homological dimension to compute
        metric : str
            Distance metric for point cloud
        n_jobs : int
            Number of parallel jobs (-1 for all cores)
        backend_params : Dict[str, Any], optional
            Backend-specific parameters
        """
        super(PersistentHomologyComputer, self).__init__()
        
        self.max_dimension = max_dimension
        self.metric = metric
        self.n_jobs = n_jobs if n_jobs != -1 else mp.cpu_count()
        self.backend_params = backend_params or {}
        
        # Initialize backend
        self.backend = self._initialize_backend(backend)
        self.backend_name = self.backend.get_name()
        
        # Statistics tracking
        self.computation_stats = {
            'total_computations': 0,
            'total_time': 0.0,
            'average_time': 0.0,
            'backend_used': self.backend_name
        }
    
    def _initialize_backend(self, backend_name: str) -> PersistenceBackend:
        """Initialize the specified backend"""
        
        if backend_name == 'auto':
            # Choose best available backend (prefer GUDHI for stability)
            if GUDHI_AVAILABLE:
                backend_name = 'gudhi'
            elif RIPSER_AVAILABLE:
                backend_name = 'ripser'
            elif GIOTTO_AVAILABLE:
                backend_name = 'giotto'
            else:
                raise RuntimeError("No persistent homology backend available. "
                                 "Install ripser, gudhi, or giotto-tda")
        
        # Create backend instance with appropriate parameters
        if backend_name == 'ripser':
            if not RIPSER_AVAILABLE:
                raise RuntimeError("Ripser backend not available")
            
            # Filter parameters for Ripser
            ripser_params = {}
            if 'metric' in self.backend_params:
                ripser_params['metric'] = self.backend_params['metric']
            else:
                ripser_params['metric'] = self.metric
            
            if 'n_perm' in self.backend_params:
                ripser_params['n_perm'] = self.backend_params['n_perm']
            
            return RipserBackend(**ripser_params)
        
        elif backend_name == 'gudhi':
            if not GUDHI_AVAILABLE:
                raise RuntimeError("GUDHI backend not available")
            
            # Filter parameters for GUDHI
            gudhi_params = {}
            if 'metric' in self.backend_params:
                gudhi_params['metric'] = self.backend_params['metric']
            else:
                gudhi_params['metric'] = self.metric
            
            if 'max_edge_length' in self.backend_params:
                gudhi_params['max_edge_length'] = self.backend_params['max_edge_length']
            
            return GUDHIBackend(**gudhi_params)
        
        elif backend_name == 'giotto':
            if not GIOTTO_AVAILABLE:
                raise RuntimeError("Giotto-tda backend not available")
            
            # Filter parameters for Giotto
            giotto_params = {}
            if 'metric' in self.backend_params:
                giotto_params['metric'] = self.backend_params['metric']
            else:
                giotto_params['metric'] = self.metric
            
            if 'max_edge_length' in self.backend_params:
                giotto_params['max_edge_length'] = self.backend_params['max_edge_length']
            
            return GiottoBackend(**giotto_params)
        
        else:
            raise ValueError(f"Unknown backend: {backend_name}")
    
    def compute_diagrams(self, 
                        point_cloud: Union[torch.Tensor, np.ndarray], 
                        max_dim: Optional[int] = None,
                        **kwargs) -> PersistenceDiagram:
        """
        Compute persistence diagrams for a point cloud
        
        Parameters:
        -----------
        point_cloud : Union[torch.Tensor, np.ndarray]
            Point cloud data of shape [n_points, n_features]
        max_dim : int, optional
            Override maximum dimension for this computation
        **kwargs : dict
            Additional backend-specific parameters
        
        Returns:
        --------
        diagram : PersistenceDiagram
            Computed persistence diagrams
        """
        # Convert to numpy if needed
        if isinstance(point_cloud, torch.Tensor):
            point_cloud_np = point_cloud.detach().cpu().numpy()
        else:
            point_cloud_np = point_cloud.copy()
        
        # Validate input
        if point_cloud_np.ndim != 2:
            raise ValueError(f"Point cloud must be 2D, got {point_cloud_np.ndim}D")
        
        if point_cloud_np.shape[0] < 2:
            raise ValueError(f"Need at least 2 points, got {point_cloud_np.shape[0]}")
        
        # Use provided max_dim or instance default
        actual_max_dim = max_dim if max_dim is not None else self.max_dimension
        
        # Compute persistence
        start_time = time.time()
        
        try:
            diagrams = self.backend.compute_persistence(
                point_cloud_np, 
                max_dimension=actual_max_dim,
                **kwargs
            )
        except Exception as e:
            raise RuntimeError(f"Persistence computation failed with {self.backend_name}: {str(e)}")
        
        computation_time = time.time() - start_time
        
        # Update statistics
        self.computation_stats['total_computations'] += 1
        self.computation_stats['total_time'] += computation_time
        self.computation_stats['average_time'] = (
            self.computation_stats['total_time'] / self.computation_stats['total_computations']
        )
        
        # Create PersistenceDiagram object
        dimensions = list(range(len(diagrams)))
        return PersistenceDiagram(diagrams, dimensions)
    
    def compute_batch_diagrams(self, 
                              point_clouds: List[Union[torch.Tensor, np.ndarray]],
                              max_dim: Optional[int] = None,
                              **kwargs) -> List[PersistenceDiagram]:
        """
        Compute persistence diagrams for multiple point clouds in parallel
        
        Parameters:
        -----------
        point_clouds : List[Union[torch.Tensor, np.ndarray]]
            List of point clouds to process
        max_dim : int, optional
            Override maximum dimension for this computation
        **kwargs : dict
            Additional backend-specific parameters
        
        Returns:
        --------
        diagrams : List[PersistenceDiagram]
            List of computed persistence diagrams
        """
        if self.n_jobs == 1:
            # Sequential processing
            return [self.compute_diagrams(pc, max_dim, **kwargs) for pc in point_clouds]
        
        # Parallel processing
        def compute_single(point_cloud):
            return self.compute_diagrams(point_cloud, max_dim, **kwargs)
        
        if self.n_jobs <= 4:  # Use ThreadPoolExecutor for small number of jobs
            with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
                diagrams = list(executor.map(compute_single, point_clouds))
        else:  # Use ProcessPoolExecutor for many jobs
            with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
                diagrams = list(executor.map(compute_single, point_clouds))
        
        return diagrams
    
    def _optimize_distance_matrix(self, 
                                 point_cloud: np.ndarray,
                                 metric: str = None,
                                 max_points: int = 1000) -> np.ndarray:
        """
        Compute optimized distance matrix for large point clouds
        
        Parameters:
        -----------
        point_cloud : np.ndarray
            Point cloud data
        metric : str, optional
            Distance metric to use
        max_points : int
            Maximum number of points to process directly
        
        Returns:
        --------
        distances : np.ndarray
            Distance matrix or subsampled point cloud
        """
        n_points = point_cloud.shape[0]
        metric = metric or self.metric
        
        # For small point clouds, compute full distance matrix
        if n_points <= max_points:
            return self._compute_distance_matrix_efficient(point_cloud, metric)
        
        # For large point clouds, use subsampling or approximation
        return self._subsample_point_cloud(point_cloud, max_points)
    
    def _compute_distance_matrix_efficient(self, 
                                         point_cloud: np.ndarray, 
                                         metric: str) -> np.ndarray:
        """
        Efficiently compute distance matrix using vectorized operations
        
        Parameters:
        -----------
        point_cloud : np.ndarray
            Point cloud data of shape [n_points, n_features]
        metric : str
            Distance metric
        
        Returns:
        --------
        distance_matrix : np.ndarray
            Distance matrix of shape [n_points, n_points]
        """
        n_points, n_features = point_cloud.shape
        
        if metric == 'euclidean':
            # Vectorized Euclidean distance computation
            # ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a·b
            sq_norms = np.sum(point_cloud**2, axis=1)
            dot_product = np.dot(point_cloud, point_cloud.T)
            
            # Broadcasting to compute all pairwise distances
            distances_sq = sq_norms[:, np.newaxis] + sq_norms[np.newaxis, :] - 2 * dot_product
            
            # Handle numerical errors (negative values close to zero)
            distances_sq = np.maximum(distances_sq, 0)
            
            return np.sqrt(distances_sq)
        
        elif metric == 'manhattan':
            # L1 distance using broadcasting
            diff = point_cloud[:, np.newaxis, :] - point_cloud[np.newaxis, :, :]
            return np.sum(np.abs(diff), axis=2)
        
        elif metric == 'cosine':
            # Cosine distance = 1 - cosine similarity
            norms = np.linalg.norm(point_cloud, axis=1)
            normalized = point_cloud / norms[:, np.newaxis]
            cosine_sim = np.dot(normalized, normalized.T)
            return 1 - cosine_sim
        
        else:
            # Fallback to scipy for other metrics
            from scipy.spatial.distance import pdist, squareform
            condensed_distances = pdist(point_cloud, metric=metric)
            return squareform(condensed_distances)
    
    def _subsample_point_cloud(self, 
                              point_cloud: np.ndarray, 
                              max_points: int,
                              method: str = 'random') -> np.ndarray:
        """
        Subsample point cloud to reduce computational complexity
        
        Parameters:
        -----------
        point_cloud : np.ndarray
            Original point cloud
        max_points : int
            Maximum number of points to keep
        method : str
            Subsampling method ('random', 'furthest', 'kmeans')
        
        Returns:
        --------
        subsampled : np.ndarray
            Subsampled point cloud
        """
        n_points = point_cloud.shape[0]
        
        if n_points <= max_points:
            return point_cloud
        
        if method == 'random':
            # Random subsampling
            indices = np.random.choice(n_points, max_points, replace=False)
            return point_cloud[indices]
        
        elif method == 'furthest':
            # Furthest point subsampling for better coverage
            return self._furthest_point_sampling(point_cloud, max_points)
        
        elif method == 'kmeans':
            # K-means clustering for representative points
            return self._kmeans_subsampling(point_cloud, max_points)
        
        else:
            raise ValueError(f"Unknown subsampling method: {method}")
    
    def _furthest_point_sampling(self, 
                                point_cloud: np.ndarray, 
                                n_samples: int) -> np.ndarray:
        """
        Furthest point sampling for better point cloud coverage
        
        Parameters:
        -----------
        point_cloud : np.ndarray
            Original point cloud
        n_samples : int
            Number of samples to select
        
        Returns:
        --------
        sampled_points : np.ndarray
            Sampled point cloud
        """
        n_points = point_cloud.shape[0]
        selected_indices = []
        
        # Start with random point
        current_idx = np.random.randint(n_points)
        selected_indices.append(current_idx)
        
        for _ in range(n_samples - 1):
            # Compute distances to all selected points
            selected_points = point_cloud[selected_indices]
            
            # Find point furthest from all selected points
            min_distances = np.full(n_points, np.inf)
            
            for selected_point in selected_points:
                distances = np.linalg.norm(point_cloud - selected_point, axis=1)
                min_distances = np.minimum(min_distances, distances)
            
            # Select point with maximum minimum distance
            furthest_idx = np.argmax(min_distances)
            selected_indices.append(furthest_idx)
        
        return point_cloud[selected_indices]
    
    def _kmeans_subsampling(self, 
                           point_cloud: np.ndarray, 
                           n_clusters: int) -> np.ndarray:
        """
        K-means based subsampling using cluster centers
        
        Parameters:
        -----------
        point_cloud : np.ndarray
            Original point cloud
        n_clusters : int
            Number of clusters (samples)
        
        Returns:
        --------
        cluster_centers : np.ndarray
            Cluster centers as representative points
        """
        try:
            from sklearn.cluster import KMeans
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            kmeans.fit(point_cloud)
            return kmeans.cluster_centers_
        
        except ImportError:
            warnings.warn("scikit-learn not available for k-means subsampling, using random sampling")
            return self._subsample_point_cloud(point_cloud, n_clusters, method='random')
    
    def get_computation_stats(self) -> Dict[str, Any]:
        """Get computation statistics"""
        return self.computation_stats.copy()
    
    def reset_stats(self):
        """Reset computation statistics"""
        self.computation_stats = {
            'total_computations': 0,
            'total_time': 0.0,
            'average_time': 0.0,
            'backend_used': self.backend_name
        }
    
    def list_available_backends(self) -> List[str]:
        """List all available backends"""
        available = []
        
        if RIPSER_AVAILABLE:
            available.append('ripser')
        if GUDHI_AVAILABLE:
            available.append('gudhi')
        if GIOTTO_AVAILABLE:
            available.append('giotto')
        
        return available
    
    def switch_backend(self, backend_name: str, backend_params: Optional[Dict[str, Any]] = None):
        """
        Switch to a different backend
        
        Parameters:
        -----------
        backend_name : str
            Name of the new backend
        backend_params : Dict[str, Any], optional
            Backend-specific parameters
        """
        if backend_params is not None:
            self.backend_params.update(backend_params)
        
        self.backend = self._initialize_backend(backend_name)
        self.backend_name = self.backend.get_name()
        
        # Update stats
        self.computation_stats['backend_used'] = self.backend_name
    
    def forward(self, point_cloud: torch.Tensor) -> torch.Tensor:
        """
        PyTorch forward pass for integration with neural networks
        
        Parameters:
        -----------
        point_cloud : torch.Tensor
            Input point cloud
        
        Returns:
        --------
        diagram_tensor : torch.Tensor
            Persistence diagram as tensor
        """
        # Compute persistence diagram
        diagram = self.compute_diagrams(point_cloud)
        
        # Convert to tensor
        return diagram.to_tensor(device=point_cloud.device)


# Utility functions for convenience
def compute_persistence_diagrams(point_cloud: Union[torch.Tensor, np.ndarray],
                               backend: str = 'auto',
                               max_dimension: int = 2,
                               metric: str = 'euclidean',
                               **kwargs) -> PersistenceDiagram:
    """
    Convenient function to compute persistence diagrams
    
    Parameters:
    -----------
    point_cloud : Union[torch.Tensor, np.ndarray]
        Point cloud data
    backend : str
        Backend to use ('auto', 'ripser', 'gudhi', 'giotto')
    max_dimension : int
        Maximum homological dimension
    metric : str
        Distance metric
    **kwargs : dict
        Additional backend parameters
    
    Returns:
    --------
    diagram : PersistenceDiagram
        Computed persistence diagrams
    """
    computer = PersistentHomologyComputer(
        backend=backend,
        max_dimension=max_dimension,
        metric=metric
    )
    
    return computer.compute_diagrams(point_cloud, **kwargs)


def benchmark_backends(point_cloud: Union[torch.Tensor, np.ndarray],
                      max_dimension: int = 2,
                      metric: str = 'euclidean',
                      n_runs: int = 3) -> Dict[str, Dict[str, float]]:
    """
    Benchmark performance across available backends
    
    Parameters:
    -----------
    point_cloud : Union[torch.Tensor, np.ndarray]
        Test point cloud
    max_dimension : int
        Maximum homological dimension
    metric : str
        Distance metric
    n_runs : int
        Number of runs for timing
    
    Returns:
    --------
    benchmark_results : Dict[str, Dict[str, float]]
        Performance comparison across backends
    """
    available_backends = []
    if RIPSER_AVAILABLE:
        available_backends.append('ripser')
    if GUDHI_AVAILABLE:
        available_backends.append('gudhi')
    if GIOTTO_AVAILABLE:
        available_backends.append('giotto')
    
    if not available_backends:
        raise RuntimeError("No backends available for benchmarking")
    
    results = {}
    
    for backend_name in available_backends:
        print(f"Benchmarking {backend_name}...")
        
        try:
            computer = PersistentHomologyComputer(
                backend=backend_name,
                max_dimension=max_dimension,
                metric=metric
            )
            
            # Warmup run
            _ = computer.compute_diagrams(point_cloud)
            
            # Timing runs
            times = []
            for run in range(n_runs):
                start_time = time.time()
                diagram = computer.compute_diagrams(point_cloud)
                end_time = time.time()
                times.append(end_time - start_time)
            
            stats = computer.get_computation_stats()
            diagram_stats = diagram.diagram_statistics()
            
            results[backend_name] = {
                'mean_time': np.mean(times),
                'std_time': np.std(times),
                'min_time': np.min(times),
                'max_time': np.max(times),
                'total_features': diagram_stats['total_features'],
                'success': True
            }
            
        except Exception as e:
            results[backend_name] = {
                'error': str(e),
                'success': False
            }
    
    return results


class PersistenceDiagramVisualizer:
    """
    Utility class for visualizing persistence diagrams
    """
    
    @staticmethod
    def plot_diagram(diagram: PersistenceDiagram, 
                    title: str = "Persistence Diagram",
                    figsize: Tuple[int, int] = (10, 8),
                    save_path: Optional[str] = None):
        """
        Plot persistence diagrams
        
        Parameters:
        -----------
        diagram : PersistenceDiagram
            Persistence diagram to plot
        title : str
            Plot title
        figsize : Tuple[int, int]
            Figure size
        save_path : str, optional
            Path to save the plot
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches
        except ImportError:
            warnings.warn("Matplotlib not available for plotting")
            return
        
        n_dims = len(diagram.diagrams)
        n_cols = min(3, n_dims)
        n_rows = (n_dims + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_dims == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
        
        for i, (dim, diag) in enumerate(zip(diagram.dimensions, diagram.diagrams)):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col] if n_rows > 1 else axes[col]
            
            if len(diag) == 0:
                ax.text(0.5, 0.5, f'No features\nin dimension {dim}', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'H_{dim}')
                continue
            
            # Plot points
            births = diag[:, 0]
            deaths = diag[:, 1]
            
            # Handle infinite deaths
            finite_mask = np.isfinite(deaths)
            finite_births = births[finite_mask]
            finite_deaths = deaths[finite_mask]
            infinite_births = births[~finite_mask]
            
            # Plot finite features
            if len(finite_births) > 0:
                color = colors[dim % len(colors)]
                ax.scatter(finite_births, finite_deaths, 
                          c=color, alpha=0.7, s=50, label=f'H_{dim}')
            
            # Plot infinite features on the boundary
            if len(infinite_births) > 0:
                max_death = np.max(finite_deaths) if len(finite_deaths) > 0 else 1.0
                ax.scatter(infinite_births, [max_death * 1.1] * len(infinite_births),
                          c=colors[dim % len(colors)], marker='^', s=80, 
                          label=f'H_{dim} (infinite)')
            
            # Plot diagonal line
            if len(finite_births) > 0 or len(infinite_births) > 0:
                all_births = np.concatenate([finite_births, infinite_births])
                max_val = max(np.max(all_births), np.max(finite_deaths) if len(finite_deaths) > 0 else 0)
                min_val = min(np.min(all_births), np.min(finite_deaths) if len(finite_deaths) > 0 else 0)
                
                ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
                ax.set_xlim(min_val * 0.9, max_val * 1.1)
                ax.set_ylim(min_val * 0.9, max_val * 1.2)
            
            ax.set_xlabel('Birth')
            ax.set_ylabel('Death')
            ax.set_title(f'H_{dim} ({len(diag)} features)')
            ax.grid(True, alpha=0.3)
        
        # Hide empty subplots
        for i in range(n_dims, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            if n_rows > 1:
                axes[row, col].set_visible(False)
            else:
                axes[col].set_visible(False)
        
        plt.suptitle(title)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    @staticmethod
    def plot_persistence_barcode(diagram: PersistenceDiagram,
                               title: str = "Persistence Barcode",
                               figsize: Tuple[int, int] = (12, 8),
                               save_path: Optional[str] = None):
        """
        Plot persistence barcode
        
        Parameters:
        -----------
        diagram : PersistenceDiagram
            Persistence diagram to plot
        title : str
            Plot title
        figsize : Tuple[int, int]
            Figure size
        save_path : str, optional
            Path to save the plot
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            warnings.warn("Matplotlib not available for plotting")
            return
        
        fig, ax = plt.subplots(figsize=figsize)
        
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
        y_pos = 0
        
        for dim, diag in zip(diagram.dimensions, diagram.diagrams):
            if len(diag) == 0:
                continue
                
            color = colors[dim % len(colors)]
            
            for birth, death in diag:
                if np.isfinite(death):
                    ax.plot([birth, death], [y_pos, y_pos], 
                           color=color, linewidth=2, alpha=0.7)
                else:
                    # Infinite persistence - plot to edge
                    max_finite = np.max(diag[np.isfinite(diag[:, 1]), 1]) if np.any(np.isfinite(diag[:, 1])) else birth + 1
                    ax.plot([birth, max_finite * 1.2], [y_pos, y_pos], 
                           color=color, linewidth=2, alpha=0.7)
                    # Add arrow to indicate infinite persistence
                    ax.annotate('', xy=(max_finite * 1.2, y_pos), xytext=(max_finite * 1.1, y_pos),
                               arrowprops=dict(arrowstyle='->', color=color, lw=2))
                
                y_pos += 1
        
        ax.set_xlabel('Filtration Parameter')
        ax.set_ylabel('Features')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        # Add dimension labels
        y_pos = 0
        for dim, diag in zip(diagram.dimensions, diagram.diagrams):
            if len(diag) > 0:
                ax.text(-0.1, y_pos + len(diag)/2, f'H_{dim}', 
                       transform=ax.get_yaxis_transform(), 
                       ha='right', va='center', fontsize=12, fontweight='bold')
                y_pos += len(diag)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


# Export main classes and functions
__all__ = [
    'PersistenceBackend',
    'RipserBackend', 
    'GUDHIBackend',
    'GiottoBackend',
    'PersistenceDiagram',
    'PersistentHomologyComputer',
    'PersistenceDiagramVisualizer',
    'compute_persistence_diagrams',
    'benchmark_backends'
] 