import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional, Tuple, Union, Dict, Any, Callable
import warnings
from scipy import integrate
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt

# Try to import optional dependencies
try:
    from sklearn.metrics import pairwise_distances
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("scikit-learn not available. Some features may be limited.")

try:
    import gudhi.wasserstein
    GUDHI_WASSERSTEIN_AVAILABLE = True
except ImportError:
    GUDHI_WASSERSTEIN_AVAILABLE = False
    warnings.warn("GUDHI Wasserstein not available. Using custom implementation.")


class PersistenceLandscape:
    """
    Persistence Landscape implementation for topological data analysis
    
    Persistence landscapes provide a stable, vectorizable representation of persistence diagrams
    that can be used for machine learning and statistical analysis.
    
    Mathematical Foundation:
    For a persistence diagram D = {(b_i, d_i)}, the k-th persistence landscape function is:
    λ_k(t) = k-th largest value of {min(t - b_i, d_i - t)_+ : (b_i, d_i) ∈ D}
    
    where x_+ = max(x, 0) is the positive part.
    """
    
    def __init__(self, 
                 persistence_diagram: np.ndarray,
                 resolution: int = 500,
                 x_range: Optional[Tuple[float, float]] = None,
                 max_landscapes: int = 5):
        """
        Initialize persistence landscape from a persistence diagram
        
        Parameters:
        -----------
        persistence_diagram : np.ndarray
            Persistence diagram as array of shape [n_points, 2] with (birth, death) pairs
        resolution : int
            Number of points for discretization
        x_range : Tuple[float, float], optional
            Range for landscape computation. If None, computed from data
        max_landscapes : int
            Maximum number of landscape functions to compute
        """
        self.persistence_diagram = persistence_diagram.copy()
        self.resolution = resolution
        self.max_landscapes = max_landscapes
        
        # Remove infinite persistence points for landscape computation
        finite_mask = np.isfinite(persistence_diagram[:, 1])
        self.finite_diagram = persistence_diagram[finite_mask]
        
        # Determine x-range for landscape computation
        if x_range is not None:
            self.x_min, self.x_max = x_range
        else:
            if len(self.finite_diagram) > 0:
                self.x_min = np.min(self.finite_diagram[:, 0]) * 0.9
                self.x_max = np.max(self.finite_diagram[:, 1]) * 1.1
            else:
                self.x_min, self.x_max = 0.0, 1.0
        
        # Create discretization grid
        self.x_grid = np.linspace(self.x_min, self.x_max, resolution)
        
        # Compute landscape functions
        self.landscapes = self._compute_landscapes()
        
        # Store landscape statistics
        self.landscape_stats = self._compute_landscape_statistics()
    
    def _compute_landscapes(self) -> np.ndarray:
        """
        Compute persistence landscape functions
        
        Returns:
        --------
        landscapes : np.ndarray
            Array of shape [max_landscapes, resolution] containing landscape functions
        """
        if len(self.finite_diagram) == 0:
            return np.zeros((self.max_landscapes, self.resolution))
        
        # For each point in the grid, compute tent function values for all persistence points
        tent_values = np.zeros((len(self.finite_diagram), self.resolution))
        
        for i, (birth, death) in enumerate(self.finite_diagram):
            # Tent function: min(t - birth, death - t)_+
            tent_values[i] = np.maximum(
                np.minimum(self.x_grid - birth, death - self.x_grid), 
                0
            )
        
        # For each grid point, sort tent values and take k-th largest
        landscapes = np.zeros((self.max_landscapes, self.resolution))
        
        for j in range(self.resolution):
            sorted_values = np.sort(tent_values[:, j])[::-1]  # Sort in descending order
            
            # Fill landscape functions with k-th largest values
            for k in range(self.max_landscapes):
                if k < len(sorted_values):
                    landscapes[k, j] = sorted_values[k]
                else:
                    landscapes[k, j] = 0.0
        
        return landscapes
    
    def _compute_landscape_statistics(self) -> Dict[str, Any]:
        """
        Compute statistical properties of the landscapes
        
        Returns:
        --------
        stats : Dict[str, Any]
            Dictionary containing landscape statistics
        """
        stats = {
            'n_persistence_points': len(self.finite_diagram),
            'x_range': (self.x_min, self.x_max),
            'resolution': self.resolution,
            'max_landscapes': self.max_landscapes,
            'landscape_norms': {},
            'landscape_integrals': {},
            'landscape_maxima': {},
            'landscape_supports': {}
        }
        
        dx = (self.x_max - self.x_min) / (self.resolution - 1)
        
        for k in range(self.max_landscapes):
            landscape_k = self.landscapes[k]
            
            # L1 norm (integral)
            l1_norm = np.trapz(landscape_k, dx=dx)
            stats['landscape_norms'][f'L1_{k}'] = l1_norm
            stats['landscape_integrals'][k] = l1_norm
            
            # L2 norm
            l2_norm = np.sqrt(np.trapz(landscape_k**2, dx=dx))
            stats['landscape_norms'][f'L2_{k}'] = l2_norm
            
            # L-infinity norm (maximum)
            linf_norm = np.max(landscape_k)
            stats['landscape_norms'][f'Linf_{k}'] = linf_norm
            stats['landscape_maxima'][k] = linf_norm
            
            # Support (where landscape is non-zero)
            support_mask = landscape_k > 1e-10
            if np.any(support_mask):
                support_indices = np.where(support_mask)[0]
                support_start = self.x_grid[support_indices[0]]
                support_end = self.x_grid[support_indices[-1]]
                stats['landscape_supports'][k] = (support_start, support_end)
            else:
                stats['landscape_supports'][k] = (0.0, 0.0)
        
        return stats
    
    def get_landscape(self, k: int) -> np.ndarray:
        """
        Get the k-th landscape function
        
        Parameters:
        -----------
        k : int
            Landscape index (0-based)
        
        Returns:
        --------
        landscape : np.ndarray
            The k-th landscape function values
        """
        if k >= self.max_landscapes:
            raise ValueError(f"Landscape index {k} exceeds maximum {self.max_landscapes}")
        
        return self.landscapes[k]
    
    def integrate_landscape(self, k: int, order: int = 1) -> float:
        """
        Compute the integral of the k-th landscape function
        
        Parameters:
        -----------
        k : int
            Landscape index
        order : int
            Order of integration (1 for L1 norm, 2 for L2 norm, etc.)
        
        Returns:
        --------
        integral : float
            Integral value
        """
        if k >= self.max_landscapes:
            raise ValueError(f"Landscape index {k} exceeds maximum {self.max_landscapes}")
        
        landscape_k = self.landscapes[k]
        dx = (self.x_max - self.x_min) / (self.resolution - 1)
        
        if order == 1:
            return np.trapz(landscape_k, dx=dx)
        else:
            return np.trapz(landscape_k**order, dx=dx)
    
    def compute_moments(self, k: int, orders: List[int] = [1, 2, 3]) -> Dict[int, float]:
        """
        Compute moments of the k-th landscape function
        
        Parameters:
        -----------
        k : int
            Landscape index
        orders : List[int]
            List of moment orders to compute
        
        Returns:
        --------
        moments : Dict[int, float]
            Dictionary mapping order to moment value
        """
        if k >= self.max_landscapes:
            raise ValueError(f"Landscape index {k} exceeds maximum {self.max_landscapes}")
        
        landscape_k = self.landscapes[k]
        dx = (self.x_max - self.x_min) / (self.resolution - 1)
        
        # Normalize landscape to create a probability distribution
        total_mass = np.trapz(landscape_k, dx=dx)
        if total_mass <= 0:
            return {order: 0.0 for order in orders}
        
        normalized_landscape = landscape_k / total_mass
        
        moments = {}
        for order in orders:
            if order == 1:
                # First moment (mean)
                moment = np.trapz(self.x_grid * normalized_landscape, dx=dx)
            else:
                # Higher moments about the mean
                mean = np.trapz(self.x_grid * normalized_landscape, dx=dx)
                moment = np.trapz((self.x_grid - mean)**order * normalized_landscape, dx=dx)
            
            moments[order] = moment
        
        return moments
    
    def to_vector(self, 
                  include_landscapes: Optional[List[int]] = None,
                  include_integrals: bool = True,
                  include_maxima: bool = True) -> np.ndarray:
        """
        Convert landscape to feature vector for machine learning
        
        Parameters:
        -----------
        include_landscapes : List[int], optional
            Which landscape functions to include. If None, includes all
        include_integrals : bool
            Whether to include landscape integrals as features
        include_maxima : bool
            Whether to include landscape maxima as features
        
        Returns:
        --------
        feature_vector : np.ndarray
            Flattened feature vector
        """
        features = []
        
        # Include landscape function values
        if include_landscapes is None:
            include_landscapes = list(range(self.max_landscapes))
        
        for k in include_landscapes:
            if k < self.max_landscapes:
                features.append(self.landscapes[k])
        
        # Include integral features
        if include_integrals:
            integrals = [self.landscape_stats['landscape_integrals'].get(k, 0.0) 
                        for k in include_landscapes]
            features.append(np.array(integrals))
        
        # Include maxima features
        if include_maxima:
            maxima = [self.landscape_stats['landscape_maxima'].get(k, 0.0) 
                     for k in include_landscapes]
            features.append(np.array(maxima))
        
        # Concatenate all features
        if features:
            return np.concatenate(features)
        else:
            return np.array([])
    
    def to_tensor(self, device: str = 'cpu') -> torch.Tensor:
        """
        Convert landscapes to PyTorch tensor
        
        Parameters:
        -----------
        device : str
            Device for tensor ('cpu' or 'cuda')
        
        Returns:
        --------
        tensor : torch.Tensor
            Landscape functions as tensor of shape [max_landscapes, resolution]
        """
        return torch.tensor(self.landscapes, dtype=torch.float32, device=device)
    
    def __len__(self) -> int:
        """Return the number of landscape functions"""
        return self.max_landscapes
    
    def __getitem__(self, k: int) -> np.ndarray:
        """Get the k-th landscape function"""
        return self.get_landscape(k)
    
    def wasserstein_distance(self, 
                           other: 'PersistenceLandscape', 
                           order: int = 1,
                           landscape_indices: Optional[List[int]] = None) -> float:
        """
        Compute Wasserstein distance between landscape functions
        
        Parameters:
        -----------
        other : PersistenceLandscape
            Other landscape to compare with
        order : int
            Order of Wasserstein distance (1 or 2)
        landscape_indices : List[int], optional
            Which landscape functions to include in distance computation
        
        Returns:
        --------
        distance : float
            Wasserstein distance between landscapes
        """
        if landscape_indices is None:
            landscape_indices = list(range(min(self.max_landscapes, other.max_landscapes)))
        
        # Ensure both landscapes have the same resolution and range
        if (self.resolution != other.resolution or 
            abs(self.x_min - other.x_min) > 1e-10 or 
            abs(self.x_max - other.x_max) > 1e-10):
            # Resample other landscape to match this one
            other_resampled = other.resample(self.x_min, self.x_max, self.resolution)
        else:
            other_resampled = other
        
        total_distance = 0.0
        
        for k in landscape_indices:
            if k < self.max_landscapes and k < other_resampled.max_landscapes:
                landscape1 = self.landscapes[k]
                landscape2 = other_resampled.landscapes[k]
                
                # Compute L^p distance between landscape functions
                if order == 1:
                    diff = np.abs(landscape1 - landscape2)
                elif order == 2:
                    diff = (landscape1 - landscape2) ** 2
                else:
                    diff = np.abs(landscape1 - landscape2) ** order
                
                # Integrate the difference
                dx = (self.x_max - self.x_min) / (self.resolution - 1)
                distance_k = np.trapz(diff, dx=dx)
                
                if order == 2:
                    distance_k = np.sqrt(distance_k)
                elif order != 1:
                    distance_k = distance_k ** (1.0 / order)
                
                total_distance += distance_k
        
        return total_distance
    
    def bottleneck_distance(self, other: 'PersistenceLandscape') -> float:
        """
        Compute bottleneck distance between the underlying persistence diagrams
        
        Parameters:
        -----------
        other : PersistenceLandscape
            Other landscape to compare with
        
        Returns:
        --------
        distance : float
            Bottleneck distance
        """
        return compute_bottleneck_distance(self.finite_diagram, other.finite_diagram)
    
    def landscape_distance(self, 
                          other: 'PersistenceLandscape',
                          metric: str = 'L2',
                          landscape_indices: Optional[List[int]] = None) -> float:
        """
        Compute distance between landscapes using various metrics
        
        Parameters:
        -----------
        other : PersistenceLandscape
            Other landscape to compare with
        metric : str
            Distance metric ('L1', 'L2', 'Linf', 'wasserstein')
        landscape_indices : List[int], optional
            Which landscape functions to include
        
        Returns:
        --------
        distance : float
            Distance between landscapes
        """
        if landscape_indices is None:
            landscape_indices = list(range(min(self.max_landscapes, other.max_landscapes)))
        
        if metric.lower() == 'wasserstein':
            return self.wasserstein_distance(other, order=1, landscape_indices=landscape_indices)
        
        # Ensure compatible grids
        if (self.resolution != other.resolution or 
            abs(self.x_min - other.x_min) > 1e-10 or 
            abs(self.x_max - other.x_max) > 1e-10):
            other_resampled = other.resample(self.x_min, self.x_max, self.resolution)
        else:
            other_resampled = other
        
        total_distance = 0.0
        
        for k in landscape_indices:
            if k < self.max_landscapes and k < other_resampled.max_landscapes:
                landscape1 = self.landscapes[k]
                landscape2 = other_resampled.landscapes[k]
                
                if metric.upper() == 'L1':
                    distance_k = np.mean(np.abs(landscape1 - landscape2))
                elif metric.upper() == 'L2':
                    distance_k = np.sqrt(np.mean((landscape1 - landscape2) ** 2))
                elif metric.upper() == 'LINF':
                    distance_k = np.max(np.abs(landscape1 - landscape2))
                else:
                    raise ValueError(f"Unknown metric: {metric}")
                
                total_distance += distance_k
        
        return total_distance
    
    def resample(self, 
                 x_min: float, 
                 x_max: float, 
                 resolution: int) -> 'PersistenceLandscape':
        """
        Resample landscape to new grid
        
        Parameters:
        -----------
        x_min : float
            New minimum x value
        x_max : float
            New maximum x value
        resolution : int
            New resolution
        
        Returns:
        --------
        resampled_landscape : PersistenceLandscape
            Resampled landscape
        """
        # Create new landscape with same diagram but different grid
        return PersistenceLandscape(
            self.persistence_diagram,
            resolution=resolution,
            x_range=(x_min, x_max),
            max_landscapes=self.max_landscapes
        )


def compute_bottleneck_distance(diagram1: np.ndarray, diagram2: np.ndarray) -> float:
    """
    Compute bottleneck distance between two persistence diagrams
    
    Parameters:
    -----------
    diagram1 : np.ndarray
        First persistence diagram
    diagram2 : np.ndarray
        Second persistence diagram
    
    Returns:
    --------
    distance : float
        Bottleneck distance
    """
    # Handle identical diagrams first
    if np.array_equal(diagram1, diagram2):
        return 0.0
    
    # Use our custom implementation as it's more reliable
    return _bottleneck_distance_custom(diagram1, diagram2)


def _bottleneck_distance_custom(diagram1: np.ndarray, diagram2: np.ndarray) -> float:
    """
    Custom implementation of bottleneck distance using Hungarian algorithm
    
    Parameters:
    -----------
    diagram1 : np.ndarray
        First persistence diagram
    diagram2 : np.ndarray
        Second persistence diagram
    
    Returns:
    --------
    distance : float
        Bottleneck distance
    """
    if len(diagram1) == 0 and len(diagram2) == 0:
        return 0.0
    
    if len(diagram1) == 0:
        return np.max([max(0, (d - b) / 2) for b, d in diagram2])
    
    if len(diagram2) == 0:
        return np.max([max(0, (d - b) / 2) for b, d in diagram1])
    
    # Add diagonal points for unmatched features
    n1, n2 = len(diagram1), len(diagram2)
    
    # Create cost matrix - size should be max(n1, n2) x max(n1, n2) for proper matching
    max_n = max(n1, n2)
    cost_matrix = np.full((max_n, max_n), np.inf)
    
    # Cost of matching points between diagrams
    for i in range(n1):
        for j in range(n2):
            cost_matrix[i, j] = np.max(np.abs(diagram1[i] - diagram2[j]))
    
    # Cost of matching points to diagonal
    for i in range(n1):
        b, d = diagram1[i]
        diagonal_cost = max(0, (d - b) / 2)
        # Fill remaining columns with diagonal cost
        for j in range(n2, max_n):
            cost_matrix[i, j] = diagonal_cost
    
    for j in range(n2):
        b, d = diagram2[j]
        diagonal_cost = max(0, (d - b) / 2)
        # Fill remaining rows with diagonal cost
        for i in range(n1, max_n):
            cost_matrix[i, j] = diagonal_cost
    
    # Fill remaining diagonal elements with 0 (matching diagonal to diagonal)
    for k in range(max(n1, n2), max_n):
        cost_matrix[k, k] = 0.0
    
    # Solve assignment problem
    row_indices, col_indices = linear_sum_assignment(cost_matrix)
    
    # Return maximum cost in optimal assignment
    return np.max([cost_matrix[i, j] for i, j in zip(row_indices, col_indices)])


def compute_wasserstein_distance(diagram1: np.ndarray, 
                                diagram2: np.ndarray, 
                                order: int = 1) -> float:
    """
    Compute Wasserstein distance between two persistence diagrams
    
    Parameters:
    -----------
    diagram1 : np.ndarray
        First persistence diagram
    diagram2 : np.ndarray
        Second persistence diagram
    order : int
        Order of Wasserstein distance (1 or 2)
    
    Returns:
    --------
    distance : float
        Wasserstein distance
    """
    if GUDHI_WASSERSTEIN_AVAILABLE:
        try:
            return gudhi.wasserstein.wasserstein_distance(
                diagram1, diagram2, order=order
            )
        except Exception:
            pass
    
    # Fallback to custom implementation
    return _wasserstein_distance_custom(diagram1, diagram2, order)


def _wasserstein_distance_custom(diagram1: np.ndarray, 
                                diagram2: np.ndarray, 
                                order: int = 1) -> float:
    """
    Custom implementation of Wasserstein distance
    
    Parameters:
    -----------
    diagram1 : np.ndarray
        First persistence diagram
    diagram2 : np.ndarray
        Second persistence diagram
    order : int
        Order of Wasserstein distance
    
    Returns:
    --------
    distance : float
        Wasserstein distance
    """
    if len(diagram1) == 0 and len(diagram2) == 0:
        return 0.0
    
    # Add diagonal points
    n1, n2 = len(diagram1), len(diagram2)
    
    # Create cost matrix
    cost_matrix = np.full((n1 + n2, n1 + n2), 0.0)
    
    # Cost of matching points between diagrams
    for i in range(n1):
        for j in range(n2):
            if order == 1:
                cost_matrix[i, j] = np.sum(np.abs(diagram1[i] - diagram2[j]))
            else:
                cost_matrix[i, j] = np.sum(np.abs(diagram1[i] - diagram2[j]) ** order)
    
    # Cost of matching points to diagonal
    for i in range(n1):
        b, d = diagram1[i]
        diagonal_cost = max(0, (d - b) / 2)
        if order != 1:
            diagonal_cost = diagonal_cost ** order
        cost_matrix[i, n2 + i] = diagonal_cost
    
    for j in range(n2):
        b, d = diagram2[j]
        diagonal_cost = max(0, (d - b) / 2)
        if order != 1:
            diagonal_cost = diagonal_cost ** order
        cost_matrix[n1 + j, j] = diagonal_cost
    
    # Solve assignment problem
    row_indices, col_indices = linear_sum_assignment(cost_matrix)
    
    # Compute total cost
    total_cost = np.sum([cost_matrix[i, j] for i, j in zip(row_indices, col_indices)])
    
    if order == 1:
        return total_cost
    else:
        return total_cost ** (1.0 / order)


class TopologicalFeatureExtractor:
    """
    Extract statistical features from persistence landscapes for machine learning
    """
    
    def __init__(self, 
                 max_landscapes: int = 5,
                 resolution: int = 500,
                 include_persistence_stats: bool = True,
                 include_landscape_stats: bool = True,
                 include_stability_features: bool = True):
        """
        Initialize feature extractor
        
        Parameters:
        -----------
        max_landscapes : int
            Maximum number of landscape functions to use
        resolution : int
            Resolution for landscape computation
        include_persistence_stats : bool
            Whether to include persistence diagram statistics
        include_landscape_stats : bool
            Whether to include landscape function statistics
        include_stability_features : bool
            Whether to include stability-based features
        """
        self.max_landscapes = max_landscapes
        self.resolution = resolution
        self.include_persistence_stats = include_persistence_stats
        self.include_landscape_stats = include_landscape_stats
        self.include_stability_features = include_stability_features
        
        # Feature names for interpretability
        self.feature_names = []
        self._build_feature_names()
    
    def _build_feature_names(self):
        """Build list of feature names for interpretability"""
        self.feature_names = []
        
        if self.include_persistence_stats:
            # Persistence diagram statistics
            self.feature_names.extend([
                'n_persistence_points',
                'max_persistence',
                'mean_persistence', 
                'std_persistence',
                'persistence_entropy',
                'persistence_range',
                'birth_range',
                'death_range'
            ])
        
        if self.include_landscape_stats:
            # Landscape statistics for each landscape function
            for k in range(self.max_landscapes):
                self.feature_names.extend([
                    f'landscape_{k}_integral',
                    f'landscape_{k}_maximum',
                    f'landscape_{k}_mean',
                    f'landscape_{k}_std',
                    f'landscape_{k}_support_length',
                    f'landscape_{k}_moment_1',
                    f'landscape_{k}_moment_2',
                    f'landscape_{k}_moment_3'
                ])
        
        if self.include_stability_features:
            # Stability-based features
            self.feature_names.extend([
                'stable_rank_0.01',
                'stable_rank_0.05', 
                'stable_rank_0.1',
                'persistence_stability_score',
                'feature_density'
            ])
    
    def extract_features(self, persistence_diagrams: List[np.ndarray]) -> np.ndarray:
        """
        Extract features from multiple persistence diagrams
        
        Parameters:
        -----------
        persistence_diagrams : List[np.ndarray]
            List of persistence diagrams for different dimensions
        
        Returns:
        --------
        features : np.ndarray
            Extracted feature vector
        """
        features = []
        
        # Combine all diagrams (with dimension labels)
        combined_diagram = []
        for dim, diagram in enumerate(persistence_diagrams):
            if len(diagram) > 0:
                # Add dimension as third column
                dim_column = np.full((len(diagram), 1), dim)
                diagram_with_dim = np.concatenate([diagram, dim_column], axis=1)
                combined_diagram.append(diagram_with_dim)
        
        if combined_diagram:
            combined_diagram = np.concatenate(combined_diagram, axis=0)
        else:
            combined_diagram = np.empty((0, 3))
        
        # Extract persistence diagram statistics
        if self.include_persistence_stats:
            persistence_features = self._extract_persistence_statistics(combined_diagram)
            features.extend(persistence_features)
        
        # Extract landscape statistics for each dimension
        if self.include_landscape_stats:
            for dim, diagram in enumerate(persistence_diagrams):
                if len(diagram) > 0:
                    landscape = PersistenceLandscape(
                        diagram, 
                        resolution=self.resolution,
                        max_landscapes=self.max_landscapes
                    )
                    landscape_features = self._extract_landscape_statistics(landscape)
                    features.extend(landscape_features)
                else:
                    # Add zero features for empty diagrams
                    features.extend([0.0] * (8 * self.max_landscapes))
        
        # Extract stability features
        if self.include_stability_features:
            stability_features = self._extract_stability_features(combined_diagram)
            features.extend(stability_features)
        
        return np.array(features)
    
    def _extract_persistence_statistics(self, diagram: np.ndarray) -> List[float]:
        """Extract statistical features from persistence diagram"""
        if len(diagram) == 0:
            return [0.0] * 8
        
        # Remove infinite persistence points
        finite_mask = np.isfinite(diagram[:, 1])
        finite_diagram = diagram[finite_mask]
        
        if len(finite_diagram) == 0:
            return [0.0] * 8
        
        # Compute persistence values
        persistence = finite_diagram[:, 1] - finite_diagram[:, 0]
        births = finite_diagram[:, 0]
        deaths = finite_diagram[:, 1]
        
        features = [
            len(finite_diagram),                    # n_persistence_points
            np.max(persistence),                    # max_persistence
            np.mean(persistence),                   # mean_persistence
            np.std(persistence),                    # std_persistence
            self._compute_persistence_entropy(persistence),  # persistence_entropy
            np.max(persistence) - np.min(persistence),       # persistence_range
            np.max(births) - np.min(births),        # birth_range
            np.max(deaths) - np.min(deaths)         # death_range
        ]
        
        return features
    
    def _extract_landscape_statistics(self, landscape: PersistenceLandscape) -> List[float]:
        """Extract statistical features from persistence landscape"""
        features = []
        
        for k in range(self.max_landscapes):
            if k < len(landscape):
                landscape_k = landscape.get_landscape(k)
                
                # Basic statistics
                integral = landscape.integrate_landscape(k, order=1)
                maximum = np.max(landscape_k)
                mean = np.mean(landscape_k)
                std = np.std(landscape_k)
                
                # Support length
                support_mask = landscape_k > 1e-10
                support_length = np.sum(support_mask) / len(landscape_k)
                
                # Moments
                moments = landscape.compute_moments(k, orders=[1, 2, 3])
                
                features.extend([
                    integral,
                    maximum,
                    mean,
                    std,
                    support_length,
                    moments.get(1, 0.0),
                    moments.get(2, 0.0),
                    moments.get(3, 0.0)
                ])
            else:
                # Add zero features for non-existent landscapes
                features.extend([0.0] * 8)
        
        return features
    
    def _extract_stability_features(self, diagram: np.ndarray) -> List[float]:
        """Extract stability-based features"""
        if len(diagram) == 0:
            return [0.0] * 5
        
        # Remove infinite persistence points
        finite_mask = np.isfinite(diagram[:, 1])
        finite_diagram = diagram[finite_mask]
        
        if len(finite_diagram) == 0:
            return [0.0] * 5
        
        persistence = finite_diagram[:, 1] - finite_diagram[:, 0]
        
        # Stable rank at different thresholds
        stable_ranks = []
        for threshold in [0.01, 0.05, 0.1]:
            stable_count = np.sum(persistence >= threshold)
            stable_ranks.append(stable_count)
        
        # Persistence stability score (weighted by persistence)
        if len(persistence) > 0:
            weights = persistence / np.sum(persistence)
            stability_score = np.sum(weights * persistence)
        else:
            stability_score = 0.0
        
        # Feature density (features per unit area)
        if len(finite_diagram) > 0:
            birth_range = np.max(finite_diagram[:, 0]) - np.min(finite_diagram[:, 0])
            death_range = np.max(finite_diagram[:, 1]) - np.min(finite_diagram[:, 1])
            area = max(birth_range * death_range, 1e-10)
            feature_density = len(finite_diagram) / area
        else:
            feature_density = 0.0
        
        features = stable_ranks + [stability_score, feature_density]
        return features
    
    def _compute_persistence_entropy(self, persistence: np.ndarray) -> float:
        """Compute entropy of persistence distribution"""
        if len(persistence) == 0:
            return 0.0
        
        # Normalize to create probability distribution
        total_persistence = np.sum(persistence)
        if total_persistence <= 0:
            return 0.0
        
        probabilities = persistence / total_persistence
        
        # Compute entropy
        entropy = -np.sum(probabilities * np.log(probabilities + 1e-10))
        return entropy
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names"""
        return self.feature_names.copy()
    
    def get_feature_importance(self, 
                              features: np.ndarray,
                              method: str = 'variance') -> Dict[str, float]:
        """
        Compute feature importance scores
        
        Parameters:
        -----------
        features : np.ndarray
            Feature matrix of shape [n_samples, n_features]
        method : str
            Method for computing importance ('variance', 'range', 'stability')
        
        Returns:
        --------
        importance : Dict[str, float]
            Dictionary mapping feature names to importance scores
        """
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        n_features = features.shape[1]
        feature_names = self.get_feature_names()
        
        if len(feature_names) != n_features:
            feature_names = [f'feature_{i}' for i in range(n_features)]
        
        importance_scores = {}
        
        if method == 'variance':
            # Use variance as importance measure
            variances = np.var(features, axis=0)
            for i, name in enumerate(feature_names):
                importance_scores[name] = variances[i]
        
        elif method == 'range':
            # Use range as importance measure
            ranges = np.max(features, axis=0) - np.min(features, axis=0)
            for i, name in enumerate(feature_names):
                importance_scores[name] = ranges[i]
        
        elif method == 'stability':
            # Use coefficient of variation as stability measure
            means = np.mean(features, axis=0)
            stds = np.std(features, axis=0)
            cv = np.divide(stds, means + 1e-10)  # Coefficient of variation
            for i, name in enumerate(feature_names):
                importance_scores[name] = 1.0 / (cv[i] + 1e-10)  # Inverse CV for stability
        
        else:
            raise ValueError(f"Unknown importance method: {method}")
        
        return importance_scores


def extract_topological_features(persistence_diagrams: List[np.ndarray],
                                max_landscapes: int = 5,
                                resolution: int = 500) -> np.ndarray:
    """
    Convenience function to extract topological features from persistence diagrams
    
    Parameters:
    -----------
    persistence_diagrams : List[np.ndarray]
        List of persistence diagrams for different dimensions
    max_landscapes : int
        Maximum number of landscape functions
    resolution : int
        Resolution for landscape computation
    
    Returns:
    --------
    features : np.ndarray
        Extracted feature vector
    """
    extractor = TopologicalFeatureExtractor(
        max_landscapes=max_landscapes,
        resolution=resolution
    )
    
    return extractor.extract_features(persistence_diagrams)


def rank_topological_features(feature_matrix: np.ndarray,
                             feature_names: Optional[List[str]] = None,
                             method: str = 'variance',
                             top_k: Optional[int] = None) -> List[Tuple[str, float]]:
    """
    Rank topological features by importance
    
    Parameters:
    -----------
    feature_matrix : np.ndarray
        Matrix of features [n_samples, n_features]
    feature_names : List[str], optional
        Names of features
    method : str
        Ranking method ('variance', 'range', 'stability')
    top_k : int, optional
        Return only top k features
    
    Returns:
    --------
    ranked_features : List[Tuple[str, float]]
        List of (feature_name, importance_score) tuples, sorted by importance
    """
    extractor = TopologicalFeatureExtractor()
    
    if feature_names is None:
        feature_names = [f'feature_{i}' for i in range(feature_matrix.shape[1])]
    
    # Temporarily set feature names
    extractor.feature_names = feature_names
    
    importance_scores = extractor.get_feature_importance(feature_matrix, method=method)
    
    # Sort by importance (descending)
    ranked_features = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)
    
    if top_k is not None:
        ranked_features = ranked_features[:top_k]
    
    return ranked_features


class PersistenceLandscapeVisualizer:
    """
    Visualization tools for persistence landscapes and related analysis
    """
    
    @staticmethod
    def plot_landscape(landscape: PersistenceLandscape,
                      landscape_indices: Optional[List[int]] = None,
                      title: str = "Persistence Landscape",
                      figsize: Tuple[int, int] = (12, 8),
                      save_path: Optional[str] = None,
                      show_diagram: bool = True):
        """
        Plot persistence landscape functions
        
        Parameters:
        -----------
        landscape : PersistenceLandscape
            Landscape to plot
        landscape_indices : List[int], optional
            Which landscape functions to plot
        title : str
            Plot title
        figsize : Tuple[int, int]
            Figure size
        save_path : str, optional
            Path to save the plot
        show_diagram : bool
            Whether to show the original persistence diagram
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            warnings.warn("Matplotlib not available for plotting")
            return
        
        if landscape_indices is None:
            landscape_indices = list(range(min(5, landscape.max_landscapes)))
        
        if show_diagram:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        else:
            fig, ax2 = plt.subplots(1, 1, figsize=figsize)
            ax1 = None
        
        # Plot persistence diagram if requested
        if show_diagram and ax1 is not None:
            diagram = landscape.finite_diagram
            if len(diagram) > 0:
                ax1.scatter(diagram[:, 0], diagram[:, 1], alpha=0.7, s=50)
                
                # Plot diagonal
                min_val = np.min(diagram)
                max_val = np.max(diagram)
                ax1.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
                
                ax1.set_xlabel('Birth')
                ax1.set_ylabel('Death')
                ax1.set_title('Persistence Diagram')
                ax1.grid(True, alpha=0.3)
            else:
                ax1.text(0.5, 0.5, 'No persistence features', 
                        ha='center', va='center', transform=ax1.transAxes)
                ax1.set_title('Persistence Diagram (Empty)')
        
        # Plot landscape functions
        colors = plt.cm.viridis(np.linspace(0, 1, len(landscape_indices)))
        
        for i, k in enumerate(landscape_indices):
            if k < landscape.max_landscapes:
                landscape_k = landscape.get_landscape(k)
                ax2.plot(landscape.x_grid, landscape_k, 
                        color=colors[i], linewidth=2, alpha=0.8, 
                        label=f'λ_{k}')
        
        ax2.set_xlabel('Filtration Parameter')
        ax2.set_ylabel('Landscape Value')
        ax2.set_title('Persistence Landscape Functions')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(title)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    @staticmethod
    def plot_landscape_comparison(landscapes: List[PersistenceLandscape],
                                 labels: Optional[List[str]] = None,
                                 landscape_index: int = 0,
                                 title: str = "Landscape Comparison",
                                 figsize: Tuple[int, int] = (12, 6),
                                 save_path: Optional[str] = None):
        """
        Compare multiple persistence landscapes
        
        Parameters:
        -----------
        landscapes : List[PersistenceLandscape]
            List of landscapes to compare
        labels : List[str], optional
            Labels for each landscape
        landscape_index : int
            Which landscape function to compare
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
        
        if labels is None:
            labels = [f'Landscape {i}' for i in range(len(landscapes))]
        
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(landscapes)))
        
        for i, (landscape, label) in enumerate(zip(landscapes, labels)):
            if landscape_index < landscape.max_landscapes:
                landscape_k = landscape.get_landscape(landscape_index)
                ax.plot(landscape.x_grid, landscape_k, 
                       color=colors[i], linewidth=2, alpha=0.8, 
                       label=label)
        
        ax.set_xlabel('Filtration Parameter')
        ax.set_ylabel(f'λ_{landscape_index} Value')
        ax.set_title(f'{title} - Landscape Function {landscape_index}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    @staticmethod
    def plot_landscape_statistics(landscape: PersistenceLandscape,
                                 title: str = "Landscape Statistics",
                                 figsize: Tuple[int, int] = (15, 10),
                                 save_path: Optional[str] = None):
        """
        Plot comprehensive statistics of persistence landscape
        
        Parameters:
        -----------
        landscape : PersistenceLandscape
            Landscape to analyze
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
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        axes = axes.flatten()
        
        # 1. Landscape functions
        ax = axes[0]
        colors = plt.cm.viridis(np.linspace(0, 1, landscape.max_landscapes))
        for k in range(min(5, landscape.max_landscapes)):
            landscape_k = landscape.get_landscape(k)
            ax.plot(landscape.x_grid, landscape_k, 
                   color=colors[k], linewidth=2, alpha=0.8, 
                   label=f'λ_{k}')
        ax.set_xlabel('Filtration Parameter')
        ax.set_ylabel('Landscape Value')
        ax.set_title('Landscape Functions')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Landscape integrals
        ax = axes[1]
        integrals = [landscape.landscape_stats['landscape_integrals'].get(k, 0.0) 
                    for k in range(landscape.max_landscapes)]
        ax.bar(range(landscape.max_landscapes), integrals, alpha=0.7)
        ax.set_xlabel('Landscape Index')
        ax.set_ylabel('Integral (L1 Norm)')
        ax.set_title('Landscape Integrals')
        ax.grid(True, alpha=0.3)
        
        # 3. Landscape maxima
        ax = axes[2]
        maxima = [landscape.landscape_stats['landscape_maxima'].get(k, 0.0) 
                 for k in range(landscape.max_landscapes)]
        ax.bar(range(landscape.max_landscapes), maxima, alpha=0.7, color='orange')
        ax.set_xlabel('Landscape Index')
        ax.set_ylabel('Maximum Value')
        ax.set_title('Landscape Maxima')
        ax.grid(True, alpha=0.3)
        
        # 4. Support lengths
        ax = axes[3]
        support_lengths = []
        for k in range(landscape.max_landscapes):
            support = landscape.landscape_stats['landscape_supports'].get(k, (0, 0))
            length = support[1] - support[0] if support[1] > support[0] else 0
            support_lengths.append(length)
        ax.bar(range(landscape.max_landscapes), support_lengths, alpha=0.7, color='green')
        ax.set_xlabel('Landscape Index')
        ax.set_ylabel('Support Length')
        ax.set_title('Landscape Support Lengths')
        ax.grid(True, alpha=0.3)
        
        # 5. Persistence diagram
        ax = axes[4]
        diagram = landscape.finite_diagram
        if len(diagram) > 0:
            ax.scatter(diagram[:, 0], diagram[:, 1], alpha=0.7, s=50)
            min_val = np.min(diagram)
            max_val = np.max(diagram)
            ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
            ax.set_xlabel('Birth')
            ax.set_ylabel('Death')
            ax.set_title(f'Persistence Diagram ({len(diagram)} points)')
        else:
            ax.text(0.5, 0.5, 'No persistence features', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Persistence Diagram (Empty)')
        ax.grid(True, alpha=0.3)
        
        # 6. Landscape norms comparison
        ax = axes[5]
        norm_types = ['L1', 'L2', 'Linf']
        norm_values = []
        for k in range(min(3, landscape.max_landscapes)):  # Show first 3 landscapes
            l1_norm = landscape.landscape_stats['landscape_norms'].get(f'L1_{k}', 0)
            l2_norm = landscape.landscape_stats['landscape_norms'].get(f'L2_{k}', 0)
            linf_norm = landscape.landscape_stats['landscape_norms'].get(f'Linf_{k}', 0)
            norm_values.append([l1_norm, l2_norm, linf_norm])
        
        if norm_values:
            norm_values = np.array(norm_values)
            x = np.arange(len(norm_types))
            width = 0.25
            
            for k in range(len(norm_values)):
                ax.bar(x + k * width, norm_values[k], width, 
                      alpha=0.7, label=f'λ_{k}')
            
            ax.set_xlabel('Norm Type')
            ax.set_ylabel('Norm Value')
            ax.set_title('Landscape Norms Comparison')
            ax.set_xticks(x + width)
            ax.set_xticklabels(norm_types)
            ax.legend()
        else:
            ax.text(0.5, 0.5, 'No landscape data', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Landscape Norms (No Data)')
        ax.grid(True, alpha=0.3)
        
        plt.suptitle(title)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    @staticmethod
    def plot_feature_importance(importance_scores: Dict[str, float],
                               title: str = "Feature Importance",
                               top_k: int = 20,
                               figsize: Tuple[int, int] = (12, 8),
                               save_path: Optional[str] = None):
        """
        Plot feature importance scores
        
        Parameters:
        -----------
        importance_scores : Dict[str, float]
            Dictionary of feature importance scores
        title : str
            Plot title
        top_k : int
            Number of top features to show
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
        
        # Sort features by importance
        sorted_features = sorted(importance_scores.items(), 
                               key=lambda x: x[1], reverse=True)
        
        # Take top k features
        top_features = sorted_features[:top_k]
        
        feature_names = [item[0] for item in top_features]
        importance_values = [item[1] for item in top_features]
        
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        
        # Create horizontal bar plot
        y_pos = np.arange(len(feature_names))
        bars = ax.barh(y_pos, importance_values, alpha=0.7)
        
        # Color bars by importance
        colors = plt.cm.viridis(np.linspace(0, 1, len(bars)))
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(feature_names)
        ax.invert_yaxis()  # Top feature at the top
        ax.set_xlabel('Importance Score')
        ax.set_title(f'{title} (Top {len(top_features)} Features)')
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    @staticmethod
    def plot_distance_matrix(landscapes: List[PersistenceLandscape],
                            labels: Optional[List[str]] = None,
                            metric: str = 'L2',
                            title: str = "Landscape Distance Matrix",
                            figsize: Tuple[int, int] = (10, 8),
                            save_path: Optional[str] = None):
        """
        Plot distance matrix between multiple landscapes
        
        Parameters:
        -----------
        landscapes : List[PersistenceLandscape]
            List of landscapes to compare
        labels : List[str], optional
            Labels for each landscape
        metric : str
            Distance metric to use
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
        
        if labels is None:
            labels = [f'L{i}' for i in range(len(landscapes))]
        
        n = len(landscapes)
        distance_matrix = np.zeros((n, n))
        
        # Compute pairwise distances
        for i in range(n):
            for j in range(n):
                if i != j:
                    distance_matrix[i, j] = landscapes[i].landscape_distance(
                        landscapes[j], metric=metric
                    )
        
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        
        # Plot heatmap
        im = ax.imshow(distance_matrix, cmap='viridis', aspect='auto')
        
        # Add colorbar
        plt.colorbar(im, ax=ax, label=f'{metric} Distance')
        
        # Set ticks and labels
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        
        # Add text annotations
        for i in range(n):
            for j in range(n):
                text = ax.text(j, i, f'{distance_matrix[i, j]:.3f}',
                             ha="center", va="center", color="white" if distance_matrix[i, j] > np.max(distance_matrix)/2 else "black")
        
        ax.set_title(f'{title} ({metric} metric)')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


# Export all classes and functions
__all__ = [
    'PersistenceLandscape',
    'TopologicalFeatureExtractor', 
    'PersistenceLandscapeVisualizer',
    'compute_bottleneck_distance',
    'compute_wasserstein_distance',
    'extract_topological_features',
    'rank_topological_features'
] 