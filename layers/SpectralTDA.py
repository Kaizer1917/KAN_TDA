"""
Spectral TDA Implementation for Frequency Domain Topological Analysis

This module implements topological data analysis (TDA) in the frequency domain,
providing tools for analyzing spectrograms, periodograms, and other frequency
representations using persistent homology and related techniques.

Key Components:
- SpectralTDAProcessor: Main class for frequency domain TDA
- SpectrogramTDA: TDA analysis of time-frequency representations
- PeriodogramTDA: TDA analysis of power spectral densities
- SpectralFeatureExtractor: Feature extraction from spectral TDA results

Mathematical Foundation:
The spectral TDA approach applies topological analysis to frequency domain
representations of time series data. This captures periodic structures,
harmonic relationships, and spectral patterns that may not be visible
in time domain TDA.

Author: TDA-KAN Integration Team
Date: Current Session
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import warnings
from dataclasses import dataclass
import logging
from scipy import signal
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt
from collections import defaultdict
import time

# Import TDA components
try:
    from .PersistentHomology import PersistentHomologyComputer
    from ..utils.persistence_landscapes import PersistenceLandscape, TopologicalFeatureExtractor
except ImportError:
    # Fallback for testing
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from layers.PersistentHomology import PersistentHomologyComputer
    from utils.persistence_landscapes import PersistenceLandscape, TopologicalFeatureExtractor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SpectralTDAConfig:
    """Configuration for spectral TDA analysis."""
    
    # Analysis type
    analysis_type: str = 'spectrogram'  # 'spectrogram', 'periodogram', 'both'
    
    # Spectrogram parameters
    window_size: int = 64
    hop_length: int = 32
    n_fft: int = 128
    window_type: str = 'hann'
    
    # Periodogram parameters
    nperseg: int = 64
    noverlap: int = 32
    detrend: str = 'constant'
    
    # Wavelet parameters
    wavelet_type: str = 'morlet'  # 'morlet', 'mexican_hat', 'paul'
    n_scales: int = 50
    
    # Frequency bands for analysis
    frequency_bands: List[Tuple[float, float]] = None
    
    # TDA parameters
    max_homology_dim: int = 2
    persistence_threshold: float = 0.01
    landscape_resolution: int = 100
    
    # Preprocessing parameters
    taper_ratio: float = 0.1
    log_transform: bool = True
    normalize_spectrum: bool = True
    
    # Feature extraction
    extract_landscapes: bool = True
    extract_persistence_stats: bool = True
    n_landscapes: int = 3
    
    # Performance options
    use_gpu: bool = True
    cache_results: bool = True
    enable_caching: bool = True  # Alias for cache_results
    parallel_processing: bool = True
    
    def __post_init__(self):
        """Post-initialization validation and setup."""
        # Set default frequency bands if not provided
        if self.frequency_bands is None:
            self.frequency_bands = [(0, 0.1), (0.1, 0.3), (0.3, 0.5)]
        
        # Validate analysis type
        valid_analysis_types = ['spectrogram', 'periodogram', 'both']
        if self.analysis_type not in valid_analysis_types:
            raise ValueError(f"analysis_type must be one of {valid_analysis_types}, got {self.analysis_type}")
        
        # Validate wavelet type
        valid_wavelet_types = ['morlet', 'mexican_hat', 'paul']
        if self.wavelet_type not in valid_wavelet_types:
            raise ValueError(f"wavelet_type must be one of {valid_wavelet_types}, got {self.wavelet_type}")
        
        # Sync caching options
        if hasattr(self, 'enable_caching'):
            self.cache_results = self.enable_caching


class SpectralTDAProcessor(nn.Module):
    """
    Main processor for spectral domain TDA analysis.
    
    This class provides comprehensive TDA analysis of frequency domain
    representations including spectrograms, periodograms, and custom
    spectral transforms.
    """
    
    def __init__(self, config: Optional[SpectralTDAConfig] = None):
        super().__init__()
        self.config = config or SpectralTDAConfig()
        
        # Initialize TDA components
        self.homology_computer = PersistentHomologyComputer(
            backend='ripser',
            max_dimension=self.config.max_homology_dim,
            metric='euclidean'
        )
        
        self.feature_extractor = TopologicalFeatureExtractor()
        
        # Cache for computed results
        self.cache = {} if self.config.cache_results else None
        
        # Performance tracking
        self.computation_stats = {
            'total_computations': 0,
            'cache_hits': 0,
            'average_computation_time': 0.0,
            'memory_usage': []
        }
        
        logger.info(f"SpectralTDAProcessor initialized with config: {self.config}")
    
    def forward(self, x: torch.Tensor, analysis_type: str = 'spectrogram') -> Dict[str, Any]:
        """
        Main forward pass for spectral TDA analysis.
        
        Args:
            x: Input time series tensor [batch_size, seq_len] or [batch_size, seq_len, features]
            analysis_type: Type of spectral analysis ('spectrogram', 'periodogram', 'both')
            
        Returns:
            Dictionary containing spectral TDA results
        """
        start_time = time.time()
        
        # Input validation
        if x.dim() < 2:
            x = x.unsqueeze(0)
        if x.dim() == 2:
            batch_size, seq_len = x.shape
            n_features = 1
        else:
            batch_size, seq_len, n_features = x.shape
            
        # Check cache
        cache_key = self._generate_cache_key(x, analysis_type)
        if self.cache is not None and cache_key in self.cache:
            self.computation_stats['cache_hits'] += 1
            return self.cache[cache_key]
        
        results = {}
        
        # Process each feature dimension
        for feature_idx in range(n_features):
            if n_features == 1:
                feature_data = x
            else:
                feature_data = x[:, :, feature_idx]
            
            feature_results = {}
            
            if analysis_type in ['spectrogram', 'both']:
                feature_results['spectrogram'] = self._analyze_spectrogram(feature_data)
            
            if analysis_type in ['periodogram', 'both']:
                feature_results['periodogram'] = self._analyze_periodogram(feature_data)
            
            results[f'feature_{feature_idx}'] = feature_results
        
        # Aggregate results across features if multiple
        if n_features > 1:
            results['aggregated'] = self._aggregate_multivariate_results(results)
        
        # Cache results
        if self.cache is not None:
            self.cache[cache_key] = results
        
        # Update statistics
        computation_time = time.time() - start_time
        self.computation_stats['total_computations'] += 1
        self.computation_stats['average_computation_time'] = (
            (self.computation_stats['average_computation_time'] * 
             (self.computation_stats['total_computations'] - 1) + computation_time) /
            self.computation_stats['total_computations']
        )
        
        return results
    
    def _analyze_spectrogram(self, x: torch.Tensor) -> Dict[str, Any]:
        """Analyze spectrogram using TDA."""
        batch_size = x.shape[0]
        batch_results = []
        
        for i in range(batch_size):
            # Compute spectrogram
            spectrogram = self._compute_spectrogram(x[i].cpu().numpy())
            
            # Apply preprocessing
            processed_spec = self._preprocess_spectrogram(spectrogram)
            
            # Convert to point cloud for TDA
            point_cloud = self._spectrogram_to_point_cloud(processed_spec)
            
            # Compute persistent homology
            diagrams = self.homology_computer.compute_diagrams(point_cloud)
            
            # Extract features
            features = self._extract_spectral_features(diagrams.diagrams, 'spectrogram')
            
            batch_results.append({
                'spectrogram': processed_spec,
                'point_cloud': point_cloud,
                'persistence_diagrams': diagrams.diagrams,
                'features': features
            })
        
        return self._aggregate_batch_results(batch_results, 'spectrogram')
    
    def _analyze_periodogram(self, x: torch.Tensor) -> Dict[str, Any]:
        """Analyze periodogram using TDA."""
        batch_size = x.shape[0]
        batch_results = []
        
        for i in range(batch_size):
            # Compute periodogram
            frequencies, psd = self._compute_periodogram(x[i].cpu().numpy())
            
            # Apply preprocessing
            processed_psd = self._preprocess_periodogram(psd)
            
            # Convert to point cloud for TDA
            point_cloud = self._periodogram_to_point_cloud(frequencies, processed_psd)
            
            # Compute persistent homology
            diagrams = self.homology_computer.compute_diagrams(point_cloud)
            
            # Extract features
            features = self._extract_spectral_features(diagrams.diagrams, 'periodogram')
            
            batch_results.append({
                'frequencies': frequencies,
                'psd': processed_psd,
                'point_cloud': point_cloud,
                'persistence_diagrams': diagrams.diagrams,
                'features': features
            })
        
        return self._aggregate_batch_results(batch_results, 'periodogram')
    
    def _compute_spectrogram(self, x: np.ndarray) -> np.ndarray:
        """Compute spectrogram using STFT."""
        try:
            # Adjust parameters for small inputs
            seq_len = len(x)
            window_size = min(self.config.window_size, seq_len)
            hop_length = min(self.config.hop_length, window_size // 2)
            n_fft = min(self.config.n_fft, window_size)
            
            # Ensure noverlap is valid
            noverlap = min(window_size - hop_length, window_size - 1)
            
            f, t, Sxx = signal.spectrogram(
                x,
                nperseg=window_size,
                noverlap=noverlap,
                nfft=n_fft,
                window=self.config.window_type,
                detrend=self.config.detrend
            )
            return Sxx
        except Exception as e:
            logger.warning(f"Spectrogram computation failed: {e}. Using fallback method.")
            # Fallback: simple STFT with adjusted parameters
            seq_len = len(x)
            window_size = min(32, seq_len)  # Smaller fallback window
            if window_size < 4:
                # For very small inputs, return a minimal spectrogram
                return np.abs(np.fft.fft(x).reshape(-1, 1))**2
            
            # Reshape into windows
            n_windows = max(1, seq_len // window_size)
            if n_windows == 1:
                return np.abs(np.fft.fft(x).reshape(-1, 1))**2
            
            windowed = x[:n_windows * window_size].reshape(n_windows, window_size)
            return np.abs(np.fft.fft(windowed, axis=1))**2
    
    def _compute_periodogram(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute periodogram (power spectral density)."""
        try:
            f, Pxx = signal.periodogram(
                x,
                nperseg=self.config.nperseg,
                noverlap=self.config.noverlap,
                nfft=self.config.n_fft,
                window=self.config.window_type,
                detrend=self.config.detrend
            )
            return f, Pxx
        except Exception as e:
            logger.warning(f"Periodogram computation failed: {e}. Using fallback method.")
            # Fallback: simple FFT-based periodogram
            X = np.fft.fft(x)
            Pxx = np.abs(X)**2 / len(x)
            f = np.fft.fftfreq(len(x))
            return f[:len(f)//2], Pxx[:len(Pxx)//2]
    
    def _preprocess_spectrogram(self, spectrogram: np.ndarray) -> np.ndarray:
        """Preprocess spectrogram for TDA analysis."""
        # Apply tapering to reduce edge effects
        if self.config.taper_ratio > 0:
            taper_size = int(self.config.taper_ratio * min(spectrogram.shape))
            taper = signal.windows.tukey(taper_size, alpha=0.5)
            # Apply taper to edges
            spectrogram = self._apply_taper(spectrogram, taper)
        
        # Log transform for better dynamic range
        if self.config.log_transform:
            spectrogram = np.log1p(spectrogram)
        
        # Normalize
        if self.config.normalize_spectrum:
            spectrogram = (spectrogram - np.mean(spectrogram)) / (np.std(spectrogram) + 1e-8)
        
        return spectrogram
    
    def _preprocess_periodogram(self, psd: np.ndarray) -> np.ndarray:
        """Preprocess periodogram for TDA analysis."""
        # Log transform
        if self.config.log_transform:
            psd = np.log1p(psd)
        
        # Normalize
        if self.config.normalize_spectrum:
            psd = (psd - np.mean(psd)) / (np.std(psd) + 1e-8)
        
        return psd
    
    def _apply_taper(self, data: np.ndarray, taper: np.ndarray) -> np.ndarray:
        """Apply tapering window to reduce edge effects."""
        tapered = data.copy()
        taper_len = len(taper)
        
        # Apply to first dimension only if it's large enough
        if data.shape[0] >= taper_len and taper_len > 0:
            half_taper = taper_len // 2
            if half_taper > 0:
                # Apply taper to beginning
                if half_taper <= data.shape[0]:
                    tapered[:half_taper] *= taper[:half_taper].reshape(-1, 1)
                
                # Apply taper to end
                if half_taper <= data.shape[0]:
                    end_start = max(0, data.shape[0] - half_taper)
                    end_taper_start = max(0, taper_len - half_taper)
                    tapered[end_start:] *= taper[end_taper_start:].reshape(-1, 1)
        
        return tapered
    
    def _spectrogram_to_point_cloud(self, spectrogram: np.ndarray) -> np.ndarray:
        """Convert spectrogram to point cloud for TDA analysis."""
        freq_bins, time_bins = spectrogram.shape
        
        # Create coordinate grid
        freq_coords, time_coords = np.meshgrid(
            np.arange(freq_bins), np.arange(time_bins), indexing='ij'
        )
        
        # Flatten and combine with magnitude
        points = np.column_stack([
            freq_coords.flatten(),
            time_coords.flatten(),
            spectrogram.flatten()
        ])
        
        # Filter out low-magnitude points to reduce noise
        magnitude_threshold = np.percentile(spectrogram.flatten(), 50)
        points = points[points[:, 2] > magnitude_threshold]
        
        return points
    
    def _periodogram_to_point_cloud(self, frequencies: np.ndarray, psd: np.ndarray) -> np.ndarray:
        """Convert periodogram to point cloud for TDA analysis."""
        # Create 2D embedding: (frequency, power)
        points = np.column_stack([frequencies, psd])
        
        # Add derived features for richer topology
        # Frequency ratios (harmonic relationships)
        freq_ratios = frequencies[1:] / frequencies[:-1]
        freq_ratios = np.concatenate([[1.0], freq_ratios])  # Pad to match length
        
        # Power gradients (spectral slope)
        power_gradients = np.gradient(psd)
        
        # Combine into higher-dimensional point cloud
        points = np.column_stack([
            frequencies,
            psd,
            freq_ratios,
            power_gradients
        ])
        
        return points
    
    def _extract_spectral_features(self, diagrams: List[np.ndarray], analysis_type: str) -> Dict[str, Any]:
        """Extract features from persistence diagrams of spectral data."""
        features = {}
        
        # Basic persistence statistics
        if self.config.extract_persistence_stats:
            # Combine all diagrams for statistics
            combined_diagram = np.vstack([d for d in diagrams if len(d) > 0]) if any(len(d) > 0 for d in diagrams) else np.empty((0, 2))
            features['persistence_stats'] = self.feature_extractor._extract_persistence_statistics(combined_diagram)
        
        # Persistence landscapes
        if self.config.extract_landscapes:
            landscapes = []
            for dim, diagram in enumerate(diagrams):
                if len(diagram) > 0:
                    landscape = PersistenceLandscape(
                        persistence_diagram=diagram,
                        resolution=self.config.landscape_resolution
                    )
                    landscapes.append(landscape)
            
            features['landscapes'] = landscapes
            
            # Extract landscape features
            if landscapes:
                landscape_features = []
                for landscape in landscapes:
                    features_for_landscape = self.feature_extractor._extract_landscape_statistics(landscape)
                    landscape_features.extend(features_for_landscape)
                features['landscape_features'] = landscape_features
        
        # Spectral-specific features
        features['spectral_features'] = self._extract_spectral_specific_features(diagrams, analysis_type)
        
        return features
    
    def _extract_spectral_specific_features(self, diagrams: List[np.ndarray], analysis_type: str) -> Dict[str, float]:
        """Extract features specific to spectral TDA analysis."""
        features = {}
        
        if len(diagrams) > 1 and len(diagrams[1]) > 0:  # 1D homology (loops)
            # Spectral periodicity strength (from 1D homology)
            persistences = diagrams[1][:, 1] - diagrams[1][:, 0]
            features['spectral_periodicity_strength'] = float(np.max(persistences)) if len(persistences) > 0 else 0.0
            features['spectral_periodicity_count'] = len(persistences)
            features['spectral_periodicity_mean'] = float(np.mean(persistences)) if len(persistences) > 0 else 0.0
        
        if len(diagrams) > 0 and len(diagrams[0]) > 0:  # 0D homology (components)
            # Spectral complexity (from 0D homology)
            persistences = diagrams[0][:, 1] - diagrams[0][:, 0]
            features['spectral_complexity'] = len(persistences)
            features['spectral_stability'] = float(np.mean(persistences)) if len(persistences) > 0 else 0.0
        
        # Analysis type specific features
        if analysis_type == 'spectrogram':
            features['time_frequency_coupling'] = self._compute_time_frequency_coupling(diagrams)
        elif analysis_type == 'periodogram':
            features['harmonic_structure'] = self._compute_harmonic_structure(diagrams)
        
        return features
    
    def _compute_time_frequency_coupling(self, diagrams: List[np.ndarray]) -> float:
        """Compute time-frequency coupling strength from spectrogram TDA."""
        if len(diagrams) > 1 and len(diagrams[1]) > 0:
            # Use 1D homology persistence as proxy for coupling strength
            persistences = diagrams[1][:, 1] - diagrams[1][:, 0]
            return float(np.std(persistences)) if len(persistences) > 1 else 0.0
        return 0.0
    
    def _compute_harmonic_structure(self, diagrams: List[np.ndarray]) -> float:
        """Compute harmonic structure strength from periodogram TDA."""
        if len(diagrams) > 0 and len(diagrams[0]) > 0:
            # Use 0D homology birth times as proxy for harmonic structure
            birth_times = diagrams[0][:, 0]
            # Harmonic structure indicated by regular spacing in birth times
            if len(birth_times) > 2:
                spacings = np.diff(np.sort(birth_times))
                return float(1.0 / (np.std(spacings) + 1e-8))  # Inverse of spacing variability
        return 0.0
    
    def _aggregate_batch_results(self, batch_results: List[Dict], analysis_type: str) -> Dict[str, Any]:
        """Aggregate results across batch dimension."""
        if not batch_results:
            return {}
        
        aggregated = {
            'batch_size': len(batch_results),
            'analysis_type': analysis_type
        }
        
        # Aggregate features
        all_features = [result['features'] for result in batch_results]
        aggregated['features'] = self._aggregate_features(all_features)
        
        # Store individual results for detailed analysis
        aggregated['individual_results'] = batch_results
        
        return aggregated
    
    def _aggregate_features(self, feature_list: List[Dict]) -> Dict[str, Any]:
        """Aggregate features across batch."""
        if not feature_list:
            return {}
        
        aggregated = {}
        
        # Aggregate numerical features
        for key in feature_list[0].keys():
            if key in ['persistence_stats', 'spectral_features']:
                values = []
                for features in feature_list:
                    if key in features and isinstance(features[key], dict):
                        for subkey, value in features[key].items():
                            if isinstance(value, (int, float)):
                                values.append(value)
                
                if values:
                    aggregated[f'{key}_mean'] = np.mean(values)
                    aggregated[f'{key}_std'] = np.std(values)
                    aggregated[f'{key}_min'] = np.min(values)
                    aggregated[f'{key}_max'] = np.max(values)
        
        return aggregated
    
    def _aggregate_multivariate_results(self, results: Dict) -> Dict[str, Any]:
        """Aggregate results across multiple features."""
        aggregated = {}
        
        # Extract feature keys (excluding 'aggregated' if it exists)
        feature_keys = [k for k in results.keys() if k.startswith('feature_')]
        
        if not feature_keys:
            return aggregated
        
        # Aggregate across features
        for analysis_type in ['spectrogram', 'periodogram']:
            type_results = []
            for feature_key in feature_keys:
                if analysis_type in results[feature_key]:
                    type_results.append(results[feature_key][analysis_type])
            
            if type_results:
                aggregated[analysis_type] = self._aggregate_analysis_type_results(type_results)
        
        return aggregated
    
    def _aggregate_analysis_type_results(self, results: List[Dict]) -> Dict[str, Any]:
        """Aggregate results for a specific analysis type across features."""
        if not results:
            return {}
        
        aggregated = {
            'n_features': len(results)
        }
        
        # Aggregate features
        all_features = [result.get('features', {}) for result in results]
        aggregated['features'] = self._aggregate_features(all_features)
        
        return aggregated
    
    def _generate_cache_key(self, x: torch.Tensor, analysis_type: str) -> str:
        """Generate cache key for input tensor and analysis type."""
        # Use tensor hash and analysis type
        tensor_hash = hash(x.cpu().numpy().tobytes())
        return f"{tensor_hash}_{analysis_type}_{x.shape}"
    
    def get_computation_stats(self) -> Dict[str, Any]:
        """Get computation statistics."""
        stats = self.computation_stats.copy()
        if self.cache is not None:
            stats['cache_size'] = len(self.cache)
            stats['cache_hit_rate'] = (
                stats['cache_hits'] / max(stats['total_computations'], 1)
            )
        return stats
    
    def clear_cache(self):
        """Clear computation cache."""
        if self.cache is not None:
            self.cache.clear()
            logger.info("Cache cleared")
    
    def visualize_spectral_tda(self, results: Dict, save_path: Optional[str] = None) -> plt.Figure:
        """Visualize spectral TDA results."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Spectral TDA Analysis Results', fontsize=16)
        
        # Plot 1: Spectrogram (if available)
        if 'spectrogram' in results.get('feature_0', {}):
            spec_data = results['feature_0']['spectrogram']['individual_results'][0]['spectrogram']
            axes[0, 0].imshow(spec_data, aspect='auto', origin='lower', cmap='viridis')
            axes[0, 0].set_title('Preprocessed Spectrogram')
            axes[0, 0].set_xlabel('Time')
            axes[0, 0].set_ylabel('Frequency')
        
        # Plot 2: Periodogram (if available)
        if 'periodogram' in results.get('feature_0', {}):
            period_data = results['feature_0']['periodogram']['individual_results'][0]
            axes[0, 1].plot(period_data['frequencies'], period_data['psd'])
            axes[0, 1].set_title('Preprocessed Periodogram')
            axes[0, 1].set_xlabel('Frequency')
            axes[0, 1].set_ylabel('Power Spectral Density')
            axes[0, 1].set_yscale('log')
        
        # Plot 3: Persistence Diagram
        if 'spectrogram' in results.get('feature_0', {}):
            diagrams = results['feature_0']['spectrogram']['individual_results'][0]['persistence_diagrams']
            if len(diagrams) > 1 and len(diagrams[1]) > 0:
                diagram = diagrams[1]  # 1D homology
                axes[1, 0].scatter(diagram[:, 0], diagram[:, 1], alpha=0.7)
                axes[1, 0].plot([0, diagram.max()], [0, diagram.max()], 'k--', alpha=0.5)
                axes[1, 0].set_title('Persistence Diagram (1D Homology)')
                axes[1, 0].set_xlabel('Birth')
                axes[1, 0].set_ylabel('Death')
        
        # Plot 4: Feature Summary
        if 'features' in results.get('feature_0', {}).get('spectrogram', {}):
            features = results['feature_0']['spectrogram']['features']['spectral_features']
            feature_names = list(features.keys())
            feature_values = list(features.values())
            
            axes[1, 1].bar(range(len(feature_names)), feature_values)
            axes[1, 1].set_title('Spectral TDA Features')
            axes[1, 1].set_xticks(range(len(feature_names)))
            axes[1, 1].set_xticklabels(feature_names, rotation=45, ha='right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Visualization saved to {save_path}")
        
        return fig

    def extract_features(self, x: torch.Tensor, analysis_type: str = None) -> torch.Tensor:
        """
        Extract features as tensor for ML integration.
        
        Args:
            x: Input time series tensor [batch_size, seq_len] or [batch_size, seq_len, features]
            analysis_type: Type of spectral analysis (defaults to config.analysis_type)
            
        Returns:
            Feature tensor [batch_size, n_features]
        """
        if analysis_type is None:
            analysis_type = self.config.analysis_type
        
        # Get full results from forward pass
        results = self.forward(x, analysis_type)
        
        # Extract numerical features into tensor format
        features_list = []
        batch_size = x.shape[0] if x.dim() >= 2 else 1
        
        # Process each feature dimension
        for feature_key in results.keys():
            if feature_key.startswith('feature_') or feature_key == 'aggregated':
                feature_data = results[feature_key]
                
                # Extract features from each analysis type
                for analysis_name in ['spectrogram', 'periodogram']:
                    if analysis_name in feature_data:
                        analysis_result = feature_data[analysis_name]
                        
                        # Extract aggregated features
                        if 'features' in analysis_result:
                            agg_features = analysis_result['features']
                            
                            # Convert feature dictionaries to tensors
                            feature_values = []
                            
                            # Extract numerical values from nested dictionaries
                            for key, value in agg_features.items():
                                if isinstance(value, dict):
                                    for subkey, subvalue in value.items():
                                        if isinstance(subvalue, (int, float)):
                                            feature_values.append(float(subvalue))
                                elif isinstance(value, (int, float)):
                                    feature_values.append(float(value))
                            
                            if feature_values:
                                features_list.extend(feature_values)
        
        # If no features extracted, create default features
        if not features_list:
            # Create basic spectral features as fallback
            if x.dim() == 2:
                x_2d = x
            else:
                x_2d = x.view(batch_size, -1)
            
            # Simple spectral features using FFT
            fft_features = []
            for i in range(batch_size):
                signal = x_2d[i].cpu().numpy()
                fft_vals = np.abs(np.fft.fft(signal))[:len(signal)//2]
                
                # Basic spectral statistics
                spectral_features = [
                    float(np.mean(fft_vals)),
                    float(np.std(fft_vals)),
                    float(np.max(fft_vals)),
                    float(np.sum(fft_vals)),
                    float(np.argmax(fft_vals)),  # Dominant frequency
                    float(np.percentile(fft_vals, 90)),
                    float(np.percentile(fft_vals, 10)),
                    float(len(fft_vals[fft_vals > np.mean(fft_vals)]))  # Above-average count
                ]
                fft_features.append(spectral_features)
            
            return torch.tensor(fft_features, dtype=torch.float32)
        
        # Convert to tensor and replicate for batch
        feature_tensor = torch.tensor(features_list, dtype=torch.float32)
        
        # Ensure proper batch dimension
        if feature_tensor.dim() == 1:
            feature_tensor = feature_tensor.unsqueeze(0).expand(batch_size, -1)
        
        return feature_tensor


class WaveletTDA(nn.Module):
    """
    Wavelet-based TDA for time-frequency analysis.
    
    This class performs TDA analysis on continuous wavelet transform (CWT)
    scalograms, providing multi-resolution time-frequency topological features.
    """
    
    def __init__(self, config: Optional[SpectralTDAConfig] = None):
        super().__init__()
        self.config = config or SpectralTDAConfig()
        
        # Initialize TDA components
        self.homology_computer = PersistentHomologyComputer(
            backend='ripser',
            max_dimension=self.config.max_homology_dim,
            metric='euclidean'
        )
        
        self.feature_extractor = TopologicalFeatureExtractor()
        
        # Wavelet parameters
        self.wavelet_types = ['morlet', 'mexican_hat', 'paul']
        self.scales = np.logspace(0, 2, 50)  # 50 scales from 1 to 100
        
        # Cache for computed results
        self.cache = {} if self.config.cache_results else None
        
        logger.info(f"WaveletTDA initialized with {len(self.scales)} scales")
    
    def forward(self, x: torch.Tensor, wavelet_type: str = 'morlet') -> Dict[str, Any]:
        """
        Perform wavelet TDA analysis.
        
        Args:
            x: Input time series tensor [batch_size, seq_len] or [batch_size, seq_len, features]
            wavelet_type: Type of wavelet ('morlet', 'mexican_hat', 'paul')
            
        Returns:
            Dictionary containing wavelet TDA results
        """
        start_time = time.time()
        
        # Input validation and reshaping
        if x.dim() < 2:
            x = x.unsqueeze(0)
        
        if x.dim() == 3:
            # For 3D input, process first feature or flatten
            if x.shape[2] == 1:
                x = x.squeeze(-1)  # [batch_size, seq_len]
            else:
                x = x[:, :, 0]  # Use first feature
        
        batch_size, seq_len = x.shape
        
        # Check cache
        cache_key = self._generate_wavelet_cache_key(x, wavelet_type)
        if self.cache is not None and cache_key in self.cache:
            return self.cache[cache_key]
        
        results = {}
        batch_results = []
        
        for i in range(batch_size):
            # Compute continuous wavelet transform
            scalogram = self._compute_cwt(x[i].cpu().numpy(), wavelet_type)
            
            # Preprocess scalogram
            processed_scalogram = self._preprocess_scalogram(scalogram)
            
            # Multi-scale TDA analysis
            scale_results = self._analyze_scalogram_multiscale(processed_scalogram)
            
            # Cross-scale feature extraction
            cross_scale_features = self._extract_cross_scale_features(scale_results)
            
            batch_results.append({
                'scalogram': processed_scalogram,
                'scale_results': scale_results,
                'cross_scale_features': cross_scale_features,
                'wavelet_type': wavelet_type
            })
        
        results = self._aggregate_wavelet_batch_results(batch_results)
        
        # Cache results
        if self.cache is not None:
            self.cache[cache_key] = results
        
        return results
    
    def _compute_cwt(self, x: np.ndarray, wavelet_type: str) -> np.ndarray:
        """Compute continuous wavelet transform."""
        try:
            import pywt
            
            # Select wavelet
            if wavelet_type == 'morlet':
                wavelet = 'cmor1.5-1.0'  # Complex Morlet
            elif wavelet_type == 'mexican_hat':
                wavelet = 'mexh'
            elif wavelet_type == 'paul':
                wavelet = 'cgau8'  # Complex Gaussian as approximation
            else:
                wavelet = 'cmor1.5-1.0'  # Default to Morlet
            
            # Compute CWT
            coefficients, frequencies = pywt.cwt(x, self.scales, wavelet)
            
            # Return magnitude for real-valued analysis
            return np.abs(coefficients)
            
        except ImportError:
            logger.warning("PyWavelets not available. Using fallback Gabor transform.")
            return self._fallback_gabor_transform(x)
        except Exception as e:
            logger.warning(f"CWT computation failed: {e}. Using fallback.")
            return self._fallback_gabor_transform(x)
    
    def _fallback_gabor_transform(self, x: np.ndarray) -> np.ndarray:
        """Fallback Gabor transform when PyWavelets is not available."""
        # Simple Gabor transform implementation
        n_scales = len(self.scales)
        n_time = len(x)
        scalogram = np.zeros((n_scales, n_time))
        
        for i, scale in enumerate(self.scales):
            # Create Gabor window
            sigma = scale / 4.0
            window_size = int(6 * sigma)
            if window_size % 2 == 0:
                window_size += 1
            
            # Gabor kernel
            t = np.arange(window_size) - window_size // 2
            gabor_real = np.exp(-t**2 / (2 * sigma**2)) * np.cos(2 * np.pi * t / scale)
            gabor_imag = np.exp(-t**2 / (2 * sigma**2)) * np.sin(2 * np.pi * t / scale)
            
            # Convolve with signal
            real_part = np.convolve(x, gabor_real, mode='same')
            imag_part = np.convolve(x, gabor_imag, mode='same')
            
            # Magnitude
            scalogram[i, :] = np.sqrt(real_part**2 + imag_part**2)
        
        return scalogram
    
    def _preprocess_scalogram(self, scalogram: np.ndarray) -> np.ndarray:
        """Preprocess scalogram for TDA analysis."""
        # Log transform for better dynamic range
        if self.config.log_transform:
            scalogram = np.log1p(scalogram)
        
        # Normalize across scales and time
        if self.config.normalize_spectrum:
            scalogram = (scalogram - np.mean(scalogram)) / (np.std(scalogram) + 1e-8)
        
        # Apply tapering to reduce edge effects
        if self.config.taper_ratio > 0:
            taper_size = int(self.config.taper_ratio * scalogram.shape[1])
            if taper_size > 0:
                taper = signal.windows.tukey(taper_size, alpha=0.5)
                # Apply taper to time dimension
                scalogram = self._apply_time_taper(scalogram, taper)
        
        return scalogram
    
    def _apply_time_taper(self, scalogram: np.ndarray, taper: np.ndarray) -> np.ndarray:
        """Apply tapering window to time dimension."""
        tapered = scalogram.copy()
        taper_len = len(taper)
        
        if scalogram.shape[1] >= taper_len:
            # Apply taper to beginning and end of time series
            for i in range(scalogram.shape[0]):
                tapered[i, :taper_len//2] *= taper[:taper_len//2]
                tapered[i, -taper_len//2:] *= taper[-taper_len//2:]
        
        return tapered
    
    def _analyze_scalogram_multiscale(self, scalogram: np.ndarray) -> Dict[str, Any]:
        """Perform multi-scale TDA analysis of scalogram."""
        n_scales, n_time = scalogram.shape
        
        # Define scale bands for analysis
        scale_bands = [
            (0, n_scales//4),           # High frequency (low scales)
            (n_scales//4, n_scales//2), # Mid-high frequency
            (n_scales//2, 3*n_scales//4), # Mid-low frequency
            (3*n_scales//4, n_scales)   # Low frequency (high scales)
        ]
        
        band_results = {}
        
        for band_idx, (start_scale, end_scale) in enumerate(scale_bands):
            band_name = f'band_{band_idx}'
            
            # Extract scale band
            band_data = scalogram[start_scale:end_scale, :]
            
            # Convert to point cloud
            point_cloud = self._scalogram_band_to_point_cloud(band_data, start_scale)
            
            # Compute persistent homology
            diagrams = self.homology_computer.compute_diagrams(point_cloud)
            
            # Extract features
            features = self._extract_wavelet_features(diagrams.diagrams, band_name)
            
            band_results[band_name] = {
                'scale_range': (start_scale, end_scale),
                'point_cloud': point_cloud,
                'persistence_diagrams': diagrams.diagrams,
                'features': features
            }
        
        return band_results
    
    def _scalogram_band_to_point_cloud(self, band_data: np.ndarray, scale_offset: int) -> np.ndarray:
        """Convert scalogram band to point cloud."""
        n_scales, n_time = band_data.shape
        
        # Create coordinate grids
        scale_coords, time_coords = np.meshgrid(
            np.arange(n_scales) + scale_offset,
            np.arange(n_time),
            indexing='ij'
        )
        
        # Flatten and combine with magnitude
        points = np.column_stack([
            scale_coords.flatten(),
            time_coords.flatten(),
            band_data.flatten()
        ])
        
        # Filter out low-magnitude points
        magnitude_threshold = np.percentile(band_data.flatten(), 60)
        points = points[points[:, 2] > magnitude_threshold]
        
        # Add derived features for richer topology
        if len(points) > 0:
            # Scale-time ratios
            scale_time_ratios = points[:, 0] / (points[:, 1] + 1)
            
            # Local magnitude gradients
            magnitude_gradients = np.gradient(points[:, 2])
            
            # Combine features
            points = np.column_stack([
                points,
                scale_time_ratios,
                magnitude_gradients
            ])
        
        return points
    
    def _extract_wavelet_features(self, diagrams: List[np.ndarray], band_name: str) -> Dict[str, Any]:
        """Extract features from wavelet TDA analysis."""
        features = {}
        
        # Basic persistence statistics
        if self.config.extract_persistence_stats:
            # Combine all diagrams for statistics
            combined_diagram = np.vstack([d for d in diagrams if len(d) > 0]) if any(len(d) > 0 for d in diagrams) else np.empty((0, 2))
            features['persistence_stats'] = self.feature_extractor._extract_persistence_statistics(combined_diagram)
        
        # Persistence landscapes
        if self.config.extract_landscapes:
            landscapes = []
            for dim, diagram in enumerate(diagrams):
                if len(diagram) > 0:
                    landscape = PersistenceLandscape(
                        persistence_diagram=diagram,
                        resolution=self.config.landscape_resolution
                    )
                    landscapes.append(landscape)
            
            features['landscapes'] = landscapes
            
            # Extract landscape features
            if landscapes:
                landscape_features = []
                for landscape in landscapes:
                    features_for_landscape = self.feature_extractor._extract_landscape_statistics(landscape)
                    landscape_features.extend(features_for_landscape)
                features['landscape_features'] = landscape_features
        
        # Wavelet-specific features
        features['wavelet_features'] = self._extract_wavelet_specific_features(diagrams, band_name)
        
        return features
    
    def _extract_wavelet_specific_features(self, diagrams: List[np.ndarray], band_name: str) -> Dict[str, float]:
        """Extract features specific to wavelet TDA analysis."""
        features = {}
        
        # Time-frequency localization (from 0D homology)
        if len(diagrams) > 0 and len(diagrams[0]) > 0:
            birth_times = diagrams[0][:, 0]
            death_times = diagrams[0][:, 1]
            persistences = death_times - birth_times
            
            features[f'{band_name}_localization_strength'] = float(np.mean(persistences)) if len(persistences) > 0 else 0.0
            features[f'{band_name}_localization_count'] = len(persistences)
            features[f'{band_name}_localization_variability'] = float(np.std(persistences)) if len(persistences) > 1 else 0.0
        
        # Oscillatory patterns (from 1D homology)
        if len(diagrams) > 1 and len(diagrams[1]) > 0:
            persistences = diagrams[1][:, 1] - diagrams[1][:, 0]
            features[f'{band_name}_oscillation_strength'] = float(np.max(persistences)) if len(persistences) > 0 else 0.0
            features[f'{band_name}_oscillation_count'] = len(persistences)
            features[f'{band_name}_oscillation_regularity'] = float(1.0 / (np.std(persistences) + 1e-8)) if len(persistences) > 1 else 0.0
        
        # Scale-time coupling
        features[f'{band_name}_scale_time_coupling'] = self._compute_scale_time_coupling(diagrams)
        
        return features
    
    def _compute_scale_time_coupling(self, diagrams: List[np.ndarray]) -> float:
        """Compute scale-time coupling strength."""
        if len(diagrams) > 1 and len(diagrams[1]) > 0:
            # Use 1D homology birth-death relationships
            births = diagrams[1][:, 0]
            deaths = diagrams[1][:, 1]
            
            if len(births) > 1:
                # Coupling indicated by correlation between birth and death times
                correlation = np.corrcoef(births, deaths)[0, 1]
                return float(abs(correlation)) if not np.isnan(correlation) else 0.0
        
        return 0.0
    
    def _extract_cross_scale_features(self, scale_results: Dict[str, Any]) -> Dict[str, float]:
        """Extract features that capture relationships across scale bands."""
        cross_features = {}
        
        # Collect persistence statistics from all bands
        band_persistences = {}
        for band_name, band_result in scale_results.items():
            if 'features' in band_result and 'wavelet_features' in band_result['features']:
                band_features = band_result['features']['wavelet_features']
                band_persistences[band_name] = band_features
        
        if len(band_persistences) >= 2:
            # Cross-band correlation
            band_names = list(band_persistences.keys())
            correlations = []
            
            for i in range(len(band_names)):
                for j in range(i+1, len(band_names)):
                    band1_features = band_persistences[band_names[i]]
                    band2_features = band_persistences[band_names[j]]
                    
                    # Find common feature types
                    common_features = set(band1_features.keys()) & set(band2_features.keys())
                    
                    if common_features:
                        values1 = [band1_features[feat] for feat in common_features]
                        values2 = [band2_features[feat] for feat in common_features]
                        
                        if len(values1) > 1:
                            corr = np.corrcoef(values1, values2)[0, 1]
                            if not np.isnan(corr):
                                correlations.append(abs(corr))
            
            if correlations:
                cross_features['cross_band_correlation_mean'] = float(np.mean(correlations))
                cross_features['cross_band_correlation_std'] = float(np.std(correlations))
                cross_features['cross_band_coupling_strength'] = float(np.max(correlations))
        
        # Scale hierarchy features
        cross_features.update(self._extract_scale_hierarchy_features(scale_results))
        
        return cross_features
    
    def _extract_scale_hierarchy_features(self, scale_results: Dict[str, Any]) -> Dict[str, float]:
        """Extract features related to hierarchical scale structure."""
        hierarchy_features = {}
        
        # Collect oscillation strengths across bands
        oscillation_strengths = []
        localization_strengths = []
        
        for band_name, band_result in scale_results.items():
            if 'features' in band_result and 'wavelet_features' in band_result['features']:
                wavelet_features = band_result['features']['wavelet_features']
                
                # Extract oscillation strength
                osc_key = f'{band_name}_oscillation_strength'
                if osc_key in wavelet_features:
                    oscillation_strengths.append(wavelet_features[osc_key])
                
                # Extract localization strength
                loc_key = f'{band_name}_localization_strength'
                if loc_key in wavelet_features:
                    localization_strengths.append(wavelet_features[loc_key])
        
        # Hierarchy features
        if oscillation_strengths:
            hierarchy_features['oscillation_hierarchy_slope'] = self._compute_hierarchy_slope(oscillation_strengths)
            hierarchy_features['oscillation_hierarchy_consistency'] = float(1.0 / (np.std(oscillation_strengths) + 1e-8))
        
        if localization_strengths:
            hierarchy_features['localization_hierarchy_slope'] = self._compute_hierarchy_slope(localization_strengths)
            hierarchy_features['localization_hierarchy_consistency'] = float(1.0 / (np.std(localization_strengths) + 1e-8))
        
        # Multi-scale energy distribution
        if oscillation_strengths and localization_strengths:
            total_energy = sum(oscillation_strengths) + sum(localization_strengths)
            if total_energy > 0:
                hierarchy_features['energy_concentration'] = float(max(oscillation_strengths + localization_strengths) / total_energy)
        
        return hierarchy_features
    
    def _compute_hierarchy_slope(self, values: List[float]) -> float:
        """Compute slope of hierarchy (trend across scales)."""
        if len(values) < 2:
            return 0.0
        
        # Linear regression slope
        x = np.arange(len(values))
        y = np.array(values)
        
        if np.std(x) > 0 and np.std(y) > 0:
            slope = np.corrcoef(x, y)[0, 1] * (np.std(y) / np.std(x))
            return float(slope) if not np.isnan(slope) else 0.0
        
        return 0.0
    
    def _aggregate_wavelet_batch_results(self, batch_results: List[Dict]) -> Dict[str, Any]:
        """Aggregate wavelet TDA results across batch."""
        if not batch_results:
            return {}
        
        aggregated = {
            'batch_size': len(batch_results),
            'analysis_type': 'wavelet_tda',
            'wavelet_type': batch_results[0]['wavelet_type']
        }
        
        # Aggregate scale band results
        band_names = list(batch_results[0]['scale_results'].keys())
        aggregated['scale_bands'] = {}
        
        for band_name in band_names:
            band_features = []
            for result in batch_results:
                if band_name in result['scale_results']:
                    band_result = result['scale_results'][band_name]
                    if 'features' in band_result:
                        band_features.append(band_result['features'])
            
            if band_features:
                aggregated['scale_bands'][band_name] = self._aggregate_features(band_features)
        
        # Aggregate cross-scale features
        cross_scale_features = [result['cross_scale_features'] for result in batch_results]
        aggregated['cross_scale_features'] = self._aggregate_cross_scale_features(cross_scale_features)
        
        # Store individual results for detailed analysis
        aggregated['individual_results'] = batch_results
        
        return aggregated
    
    def _aggregate_cross_scale_features(self, feature_list: List[Dict]) -> Dict[str, float]:
        """Aggregate cross-scale features across batch."""
        if not feature_list:
            return {}
        
        aggregated = {}
        
        # Get all feature keys
        all_keys = set()
        for features in feature_list:
            all_keys.update(features.keys())
        
        # Aggregate each feature
        for key in all_keys:
            values = [features.get(key, 0.0) for features in feature_list]
            values = [v for v in values if isinstance(v, (int, float)) and not np.isnan(v)]
            
            if values:
                aggregated[f'{key}_mean'] = float(np.mean(values))
                aggregated[f'{key}_std'] = float(np.std(values))
                aggregated[f'{key}_min'] = float(np.min(values))
                aggregated[f'{key}_max'] = float(np.max(values))
        
        return aggregated
    
    def _generate_wavelet_cache_key(self, x: torch.Tensor, wavelet_type: str) -> str:
        """Generate cache key for wavelet analysis."""
        tensor_hash = hash(x.cpu().numpy().tobytes())
        return f"wavelet_{tensor_hash}_{wavelet_type}_{x.shape}"
    
    def visualize_wavelet_tda(self, results: Dict, save_path: Optional[str] = None) -> plt.Figure:
        """Visualize wavelet TDA results."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'Wavelet TDA Analysis Results ({results.get("wavelet_type", "unknown")})', fontsize=16)
        
        if 'individual_results' in results and len(results['individual_results']) > 0:
            first_result = results['individual_results'][0]
            
            # Plot 1: Scalogram
            if 'scalogram' in first_result:
                scalogram = first_result['scalogram']
                im = axes[0, 0].imshow(scalogram, aspect='auto', origin='lower', cmap='viridis')
                axes[0, 0].set_title('Preprocessed Scalogram')
                axes[0, 0].set_xlabel('Time')
                axes[0, 0].set_ylabel('Scale')
                plt.colorbar(im, ax=axes[0, 0])
            
            # Plot 2-5: Scale band persistence diagrams
            scale_results = first_result.get('scale_results', {})
            band_names = list(scale_results.keys())[:4]  # First 4 bands
            
            for i, band_name in enumerate(band_names):
                row, col = divmod(i+1, 3)
                if row < 2 and col < 3:
                    band_result = scale_results[band_name]
                    if 'persistence_diagrams' in band_result:
                        diagrams = band_result['persistence_diagrams']
                        if len(diagrams) > 1 and len(diagrams[1]) > 0:
                            diagram = diagrams[1]  # 1D homology
                            axes[row, col].scatter(diagram[:, 0], diagram[:, 1], alpha=0.7)
                            axes[row, col].plot([0, diagram.max()], [0, diagram.max()], 'k--', alpha=0.5)
                            axes[row, col].set_title(f'Persistence Diagram - {band_name}')
                            axes[row, col].set_xlabel('Birth')
                            axes[row, col].set_ylabel('Death')
            
            # Plot 6: Cross-scale features
            if 'cross_scale_features' in first_result:
                cross_features = first_result['cross_scale_features']
                if cross_features:
                    feature_names = list(cross_features.keys())[:10]  # First 10 features
                    feature_values = [cross_features[name] for name in feature_names]
                    
                    axes[1, 2].bar(range(len(feature_names)), feature_values)
                    axes[1, 2].set_title('Cross-Scale Features')
                    axes[1, 2].set_xticks(range(len(feature_names)))
                    axes[1, 2].set_xticklabels(feature_names, rotation=45, ha='right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Wavelet TDA visualization saved to {save_path}")
        
        return fig

    def extract_features(self, x: torch.Tensor, wavelet_type: str = None) -> torch.Tensor:
        """
        Extract wavelet TDA features as tensor for ML integration.
        
        Args:
            x: Input time series tensor [batch_size, seq_len] or [batch_size, seq_len, features]
            wavelet_type: Type of wavelet (defaults to config.wavelet_type)
            
        Returns:
            Feature tensor [batch_size, n_features]
        """
        if wavelet_type is None:
            wavelet_type = self.config.wavelet_type
        
        # Get full results from forward pass
        results = self.forward(x, wavelet_type)
        
        # Extract numerical features into tensor format
        batch_size = x.shape[0] if x.dim() >= 2 else 1
        all_features = []
        
        # Extract features from scale bands
        if 'scale_bands' in results:
            scale_bands = results['scale_bands']
            
            for band_name, band_data in scale_bands.items():
                # Extract aggregated features for this band
                for key, value in band_data.items():
                    if key.endswith('_mean') or key.endswith('_std') or key.endswith('_min') or key.endswith('_max'):
                        if isinstance(value, (int, float)) and not np.isnan(value):
                            all_features.append(float(value))
        
        # Extract cross-scale features
        if 'cross_scale_features' in results:
            cross_features = results['cross_scale_features']
            
            for key, value in cross_features.items():
                if isinstance(value, (int, float)) and not np.isnan(value):
                    all_features.append(float(value))
        
        # If no features extracted, create default wavelet features
        if not all_features:
            # Create basic wavelet-like features as fallback
            if x.dim() == 2:
                x_2d = x
            else:
                x_2d = x.view(batch_size, -1)
            
            # Simple multi-scale features
            wavelet_features = []
            for i in range(batch_size):
                signal = x_2d[i].cpu().numpy()
                
                # Multi-scale variance features (approximating wavelet scales)
                scale_features = []
                for scale in [2, 4, 8, 16]:
                    if len(signal) >= scale:
                        # Downsample and compute variance
                        downsampled = signal[::scale]
                        scale_features.extend([
                            float(np.var(downsampled)),
                            float(np.mean(downsampled)),
                            float(np.std(downsampled)),
                            float(np.max(downsampled) - np.min(downsampled)),
                            float(len(downsampled))
                        ])
                    else:
                        scale_features.extend([0.0, 0.0, 0.0, 0.0, 0.0])
                
                # Cross-scale features
                if len(scale_features) >= 8:
                    cross_scale = [
                        float(np.corrcoef(scale_features[:4], scale_features[4:8])[0, 1]) if not np.isnan(np.corrcoef(scale_features[:4], scale_features[4:8])[0, 1]) else 0.0,
                        float(np.mean(scale_features)),
                        float(np.std(scale_features)),
                        float(np.max(scale_features)),
                        float(np.min(scale_features)),
                        float(np.sum(scale_features)),
                        float(len([f for f in scale_features if f > 0])),
                        float(np.median(scale_features))
                    ]
                    scale_features.extend(cross_scale)
                
                wavelet_features.append(scale_features)
            
            return torch.tensor(wavelet_features, dtype=torch.float32)
        
        # Convert to tensor and replicate for batch
        feature_tensor = torch.tensor(all_features, dtype=torch.float32)
        
        # Ensure proper batch dimension
        if feature_tensor.dim() == 1:
            feature_tensor = feature_tensor.unsqueeze(0).expand(batch_size, -1)
        
        return feature_tensor


class SpectralTemporalTDAIntegrator:
    """
    Integrates Spectral TDA with existing temporal TDA pipeline
    Provides unified feature extraction combining frequency and time domain analysis
    """
    
    def __init__(self, 
                 spectral_config: Optional[SpectralTDAConfig] = None,
                 temporal_config: Optional[Any] = None):
        """
        Initialize integrated spectral-temporal TDA processor
        
        Args:
            spectral_config: Configuration for spectral TDA analysis
            temporal_config: Configuration for temporal TDA analysis (TDAConfig)
        """
        self.spectral_config = spectral_config or SpectralTDAConfig()
        self.temporal_config = temporal_config
        
        # Initialize components
        self.spectral_processor = SpectralTDAProcessor(self.spectral_config)
        self.wavelet_processor = WaveletTDA(self.spectral_config)
        
        # Initialize temporal TDA if config provided
        self.temporal_extractor = None
        if temporal_config is not None:
            try:
                from .TDAFeatureExtractor import TDAFeatureExtractor
                self.temporal_extractor = TDAFeatureExtractor(temporal_config)
            except ImportError:
                warnings.warn("TDAFeatureExtractor not available, temporal features disabled")
        
        # Feature fusion weights
        self.fusion_weights = {
            'spectral': 0.4,
            'wavelet': 0.3,
            'temporal': 0.3
        }
        
        # Performance tracking
        self.integration_stats = defaultdict(list)
        
    def extract_unified_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract unified spectral-temporal TDA features
        
        Args:
            x: Input time series [batch_size, seq_len, features]
            
        Returns:
            Dictionary containing all extracted features
        """
        start_time = time.time()
        
        features = {}
        
        # Ensure input is 2D for spectral/wavelet processing
        if x.dim() == 3:
            # For multivariate input, process first feature or flatten
            if x.shape[2] == 1:
                x_2d = x.squeeze(-1)  # [batch_size, seq_len]
            else:
                x_2d = x[:, :, 0]  # Use first feature
        else:
            x_2d = x
        
        # Extract spectral features
        try:
            spectral_features = self.spectral_processor.extract_features(x_2d)
            features['spectral'] = spectral_features
        except Exception as e:
            warnings.warn(f"Spectral feature extraction failed: {e}")
            features['spectral'] = torch.zeros(x.shape[0], 32)
        
        # Extract wavelet features
        try:
            wavelet_features = self.wavelet_processor.extract_features(x_2d)
            features['wavelet'] = wavelet_features
        except Exception as e:
            warnings.warn(f"Wavelet feature extraction failed: {e}")
            features['wavelet'] = torch.zeros(x.shape[0], 40)
        
        # Extract temporal features if available
        if self.temporal_extractor is not None:
            try:
                temporal_features = self.temporal_extractor(x)
                features['temporal'] = temporal_features
            except Exception as e:
                warnings.warn(f"Temporal feature extraction failed: {e}")
                features['temporal'] = torch.zeros(x.shape[0], 50)
        
        # Track performance
        duration = time.time() - start_time
        self.integration_stats['extraction_time'].append(duration)
        self.integration_stats['feature_counts'].append({
            k: v.shape[1] if len(v.shape) > 1 else v.numel() 
            for k, v in features.items()
        })
        
        return features
    
    def fuse_features(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Fuse spectral and temporal features into unified representation
        
        Args:
            features: Dictionary of feature tensors
            
        Returns:
            Fused feature tensor
        """
        fused_features = []
        
        for feature_type, feature_tensor in features.items():
            if feature_type in self.fusion_weights:
                weight = self.fusion_weights[feature_type]
                weighted_features = feature_tensor * weight
                fused_features.append(weighted_features)
        
        if not fused_features:
            # Fallback to concatenation if no weights match
            fused_features = list(features.values())
        
        # Concatenate all features
        if len(fused_features) == 1:
            return fused_features[0]
        
        # Ensure all features have same batch size
        batch_size = fused_features[0].shape[0]
        normalized_features = []
        
        for feat in fused_features:
            if len(feat.shape) == 1:
                feat = feat.unsqueeze(0).expand(batch_size, -1)
            elif feat.shape[0] != batch_size:
                # Repeat or truncate to match batch size
                if feat.shape[0] == 1:
                    feat = feat.expand(batch_size, -1)
                else:
                    feat = feat[:batch_size]
            normalized_features.append(feat)
        
        return torch.cat(normalized_features, dim=1)
    
    def process_batch(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process batch with unified spectral-temporal TDA analysis
        
        Args:
            x: Input batch [batch_size, seq_len, features]
            
        Returns:
            Unified TDA features [batch_size, total_features]
        """
        features = self.extract_unified_features(x)
        return self.fuse_features(features)
    
    def get_feature_dimensions(self) -> Dict[str, int]:
        """Get dimensions of each feature type"""
        dims = {}
        
        # Get actual frequency bands from config
        freq_bands = getattr(self.spectral_config, 'frequency_bands', [(0, 0.1), (0.1, 0.3), (0.3, 0.5)])
        n_bands = len(freq_bands)
        
        # Spectral features (estimated based on implementation)
        dims['spectral'] = (
            n_bands * 4 +  # Band features
            6 +  # Global spectral features
            n_bands * 3  # TDA features per band
        )
        
        # Wavelet features (estimated based on scale bands)
        n_scale_bands = 4  # From implementation: 4 scale bands
        dims['wavelet'] = (
            n_scale_bands * 5 +  # Scale band features
            8 +  # Cross-scale features
            7   # Hierarchical features
        )
        
        # Temporal features (if available)
        if self.temporal_extractor is not None:
            # Estimate based on typical TDA feature extraction
            dims['temporal'] = 50  # Typical TDA feature count
        
        return dims
    
    def get_integration_stats(self) -> Dict[str, Any]:
        """Get integration performance statistics"""
        stats = dict(self.integration_stats)
        
        if 'extraction_time' in stats and stats['extraction_time']:
            stats['avg_extraction_time'] = np.mean(stats['extraction_time'])
            stats['total_extractions'] = len(stats['extraction_time'])
        
        return stats
    
    def optimize_fusion_weights(self, x: torch.Tensor, y: torch.Tensor = None) -> Dict[str, float]:
        """
        Optimize fusion weights based on feature importance
        
        Args:
            x: Input data for analysis
            y: Optional target data for supervised optimization
            
        Returns:
            Optimized fusion weights
        """
        features = self.extract_unified_features(x)
        
        # Simple variance-based optimization if no targets
        if y is None:
            variances = {}
            for feature_type, feature_tensor in features.items():
                if feature_tensor.numel() > 0:
                    variances[feature_type] = torch.var(feature_tensor).item()
            
            # Normalize variances to weights
            total_var = sum(variances.values())
            if total_var > 0:
                self.fusion_weights = {
                    k: v / total_var for k, v in variances.items()
                }
        
        return self.fusion_weights


def create_spectral_tda_pipeline(
    spectral_config: Optional[SpectralTDAConfig] = None,
    temporal_config: Optional[Any] = None,
    enable_integration: bool = True
) -> Union[SpectralTDAProcessor, SpectralTemporalTDAIntegrator]:
    """
    Factory function to create spectral TDA pipeline
    
    Args:
        spectral_config: Configuration for spectral analysis
        temporal_config: Configuration for temporal TDA analysis
        enable_integration: Whether to create integrated pipeline
        
    Returns:
        Spectral TDA processor or integrated processor
    """
    if enable_integration and temporal_config is not None:
        return SpectralTemporalTDAIntegrator(spectral_config, temporal_config)
    else:
        return SpectralTDAProcessor(spectral_config or SpectralTDAConfig())


def validate_spectral_patterns(processor: SpectralTDAProcessor, 
                             pattern_type: str = "sine_wave") -> Dict[str, Any]:
    """
    Validate spectral TDA processor on known patterns
    
    Args:
        processor: SpectralTDA processor to validate
        pattern_type: Type of test pattern to generate
        
    Returns:
        Validation results
    """
    # Generate test patterns
    if pattern_type == "sine_wave":
        t = np.linspace(0, 4*np.pi, 200)
        # Multi-frequency sine wave
        signal = (np.sin(t) + 0.5*np.sin(3*t) + 0.25*np.sin(5*t) + 
                 0.1*np.random.randn(len(t)))
        
    elif pattern_type == "chirp":
        t = np.linspace(0, 1, 200)
        # Frequency sweep
        signal = np.sin(2*np.pi*(5*t + 10*t**2)) + 0.1*np.random.randn(len(t))
        
    elif pattern_type == "noise":
        # White noise
        signal = np.random.randn(200)
        
    else:
        raise ValueError(f"Unknown pattern type: {pattern_type}")
    
    # Convert to tensor
    x = torch.tensor(signal, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
    
    # Extract features
    start_time = time.time()
    features = processor.extract_features(x)
    extraction_time = time.time() - start_time
    
    # Analyze results
    results = {
        'pattern_type': pattern_type,
        'signal_length': len(signal),
        'feature_count': features.shape[1] if len(features.shape) > 1 else features.numel(),
        'extraction_time': extraction_time,
        'feature_stats': {
            'mean': torch.mean(features).item(),
            'std': torch.std(features).item(),
            'min': torch.min(features).item(),
            'max': torch.max(features).item(),
            'non_zero_ratio': (features != 0).float().mean().item()
        }
    }
    
    # Pattern-specific validation
    if pattern_type == "sine_wave":
        # Should detect multiple frequency components
        results['expected_frequencies'] = [1/(2*np.pi), 3/(2*np.pi), 5/(2*np.pi)]
        results['validation'] = 'Multi-frequency pattern should show rich spectral structure'
        
    elif pattern_type == "chirp":
        # Should show time-varying frequency content
        results['expected_behavior'] = 'Frequency sweep pattern'
        results['validation'] = 'Should show varying spectral content over time'
        
    elif pattern_type == "noise":
        # Should show broad spectrum
        results['expected_behavior'] = 'Broad spectrum noise'
        results['validation'] = 'Should show distributed spectral energy'
    
    return results


# Export main classes and functions
__all__ = [
    'SpectralTDAConfig',
    'SpectralTDAProcessor', 
    'WaveletTDA',
    'SpectralTemporalTDAIntegrator',
    'create_spectral_tda_pipeline',
    'validate_spectral_patterns'
] 