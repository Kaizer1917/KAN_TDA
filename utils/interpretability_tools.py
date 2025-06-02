"""
Interpretability and Debugging Tools for TDA-KAN_TDA Integration

This module provides comprehensive tools for interpreting and debugging the
TDA-KAN_TDA integration system, including feature importance analysis,
attention visualization, and topological feature interpretation.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from dataclasses import dataclass
import pandas as pd
from pathlib import Path
import json
import warnings
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class InterpretabilityConfig:
    """Configuration for interpretability analysis."""
    
    # Feature importance settings
    importance_method: str = "gradient"  # gradient, permutation, attention
    num_permutation_samples: int = 100
    importance_threshold: float = 0.01
    
    # Visualization settings
    figure_size: Tuple[int, int] = (12, 8)
    color_palette: str = "viridis"
    save_plots: bool = True
    plot_format: str = "png"
    dpi: int = 300
    
    # Analysis settings
    top_k_features: int = 20
    correlation_threshold: float = 0.7
    enable_statistical_tests: bool = True
    
    # Output settings
    output_dir: str = "interpretability_results"
    save_detailed_reports: bool = True
    include_raw_data: bool = False


class TDAFeatureImportanceAnalyzer:
    """Analyzes importance of TDA features in the integrated model."""
    
    def __init__(self, config: InterpretabilityConfig):
        self.config = config
        self.importance_cache = {}
        self.feature_names = []
        
    def analyze_feature_importance(self, 
                                 model: nn.Module,
                                 x: torch.Tensor,
                                 tda_features: Dict,
                                 target: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        Analyze importance of TDA features in model predictions.
        
        Args:
            model: The TDA-KAN_TDA model
            x: Input time series data
            tda_features: TDA features dictionary
            target: Optional target values for supervised importance
            
        Returns:
            Dictionary containing importance analysis results
        """
        results = {}
        
        # Extract feature names and values
        feature_dict = self._extract_feature_dict(tda_features)
        self.feature_names = list(feature_dict.keys())
        
        # Compute importance using different methods
        if self.config.importance_method == "gradient":
            results['gradient_importance'] = self._compute_gradient_importance(
                model, x, feature_dict, target
            )
        elif self.config.importance_method == "permutation":
            results['permutation_importance'] = self._compute_permutation_importance(
                model, x, feature_dict, target
            )
        elif self.config.importance_method == "attention":
            results['attention_importance'] = self._compute_attention_importance(
                model, x, feature_dict
            )
        else:
            # Compute all methods
            results['gradient_importance'] = self._compute_gradient_importance(
                model, x, feature_dict, target
            )
            results['permutation_importance'] = self._compute_permutation_importance(
                model, x, feature_dict, target
            )
            results['attention_importance'] = self._compute_attention_importance(
                model, x, feature_dict
            )
        
        # Analyze feature correlations
        results['feature_correlations'] = self._analyze_feature_correlations(feature_dict)
        
        # Compute feature statistics
        results['feature_statistics'] = self._compute_feature_statistics(feature_dict)
        
        # Generate interpretability report
        results['interpretation_report'] = self._generate_interpretation_report(results)
        
        return results
    
    def _extract_feature_dict(self, tda_features: Dict) -> Dict[str, torch.Tensor]:
        """Extract flattened feature dictionary from TDA features."""
        feature_dict = {}
        
        # Extract persistence features
        if 'persistence_features' in tda_features:
            pers_features = tda_features['persistence_features']
            if isinstance(pers_features, torch.Tensor):
                for i in range(pers_features.size(-1)):
                    feature_dict[f'persistence_feature_{i}'] = pers_features[..., i]
        
        # Extract landscape features
        if 'landscape_features' in tda_features:
            land_features = tda_features['landscape_features']
            if isinstance(land_features, torch.Tensor):
                for i in range(land_features.size(-1)):
                    feature_dict[f'landscape_feature_{i}'] = land_features[..., i]
        
        # Extract statistical features
        if 'statistical_features' in tda_features:
            stat_features = tda_features['statistical_features']
            if isinstance(stat_features, dict):
                for name, value in stat_features.items():
                    if isinstance(value, torch.Tensor):
                        feature_dict[f'stat_{name}'] = value
        
        # Extract embedding features
        if 'embedding_features' in tda_features:
            emb_features = tda_features['embedding_features']
            if isinstance(emb_features, torch.Tensor):
                for i in range(emb_features.size(-1)):
                    feature_dict[f'embedding_feature_{i}'] = emb_features[..., i]
        
        return feature_dict
    
    def _compute_gradient_importance(self, 
                                   model: nn.Module,
                                   x: torch.Tensor,
                                   feature_dict: Dict[str, torch.Tensor],
                                   target: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """Compute feature importance using gradient-based methods."""
        model.eval()
        importance_scores = {}
        
        # Create feature tensor
        feature_tensor = torch.stack(list(feature_dict.values()), dim=-1)
        feature_tensor.requires_grad_(True)
        
        # Forward pass
        try:
            output = model(x, tda_features={'combined_features': feature_tensor})
            
            # Compute loss
            if target is not None:
                loss = nn.MSELoss()(output, target)
            else:
                loss = output.mean()  # Use mean output as proxy
            
            # Backward pass
            loss.backward()
            
            # Extract gradients
            if feature_tensor.grad is not None:
                gradients = feature_tensor.grad.abs().mean(dim=(0, 1))  # Average over batch and time
                
                for i, (name, _) in enumerate(feature_dict.items()):
                    importance_scores[name] = gradients[i].item()
            
        except Exception as e:
            logger.warning(f"Gradient importance computation failed: {e}")
            # Fallback to zero importance
            for name in feature_dict.keys():
                importance_scores[name] = 0.0
        
        return importance_scores
    
    def _compute_permutation_importance(self,
                                      model: nn.Module,
                                      x: torch.Tensor,
                                      feature_dict: Dict[str, torch.Tensor],
                                      target: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """Compute feature importance using permutation method."""
        model.eval()
        importance_scores = {}
        
        # Get baseline performance
        with torch.no_grad():
            baseline_output = model(x, tda_features=self._dict_to_tda_features(feature_dict))
            if target is not None:
                baseline_loss = nn.MSELoss()(baseline_output, target).item()
            else:
                baseline_loss = baseline_output.var().item()  # Use variance as proxy
        
        # Compute importance for each feature
        for feature_name in feature_dict.keys():
            losses = []
            
            for _ in range(self.config.num_permutation_samples):
                # Create permuted feature dict
                permuted_dict = feature_dict.copy()
                
                # Permute the specific feature
                original_feature = permuted_dict[feature_name].clone()
                permuted_indices = torch.randperm(original_feature.size(0))
                permuted_dict[feature_name] = original_feature[permuted_indices]
                
                # Compute loss with permuted feature
                with torch.no_grad():
                    try:
                        permuted_output = model(x, tda_features=self._dict_to_tda_features(permuted_dict))
                        if target is not None:
                            permuted_loss = nn.MSELoss()(permuted_output, target).item()
                        else:
                            permuted_loss = permuted_output.var().item()
                        losses.append(permuted_loss)
                    except Exception:
                        losses.append(baseline_loss)  # Fallback
            
            # Compute importance as average loss increase
            avg_permuted_loss = np.mean(losses)
            importance_scores[feature_name] = max(0, avg_permuted_loss - baseline_loss)
        
        return importance_scores
    
    def _compute_attention_importance(self,
                                    model: nn.Module,
                                    x: torch.Tensor,
                                    feature_dict: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Compute feature importance using attention weights."""
        model.eval()
        importance_scores = {}
        
        # Hook to capture attention weights
        attention_weights = {}
        
        def attention_hook(name):
            def hook(module, input, output):
                if hasattr(output, 'attention_weights'):
                    attention_weights[name] = output.attention_weights
                elif isinstance(output, tuple) and len(output) > 1:
                    # Assume second element is attention weights
                    attention_weights[name] = output[1]
            return hook
        
        # Register hooks
        hooks = []
        for name, module in model.named_modules():
            if 'attention' in name.lower() or 'fusion' in name.lower():
                hook = module.register_forward_hook(attention_hook(name))
                hooks.append(hook)
        
        # Forward pass
        try:
            with torch.no_grad():
                _ = model(x, tda_features=self._dict_to_tda_features(feature_dict))
            
            # Extract attention-based importance
            if attention_weights:
                # Average attention weights across all attention modules
                all_weights = []
                for weights in attention_weights.values():
                    if isinstance(weights, torch.Tensor):
                        all_weights.append(weights.mean(dim=0))  # Average over batch
                
                if all_weights:
                    avg_attention = torch.stack(all_weights).mean(dim=0)
                    
                    # Map attention weights to features
                    num_features = len(feature_dict)
                    if avg_attention.numel() >= num_features:
                        attention_scores = avg_attention.flatten()[:num_features]
                        
                        for i, name in enumerate(feature_dict.keys()):
                            importance_scores[name] = attention_scores[i].item()
        
        except Exception as e:
            logger.warning(f"Attention importance computation failed: {e}")
        
        finally:
            # Remove hooks
            for hook in hooks:
                hook.remove()
        
        # Fallback to uniform importance if no attention weights found
        if not importance_scores:
            uniform_importance = 1.0 / len(feature_dict)
            for name in feature_dict.keys():
                importance_scores[name] = uniform_importance
        
        return importance_scores
    
    def _dict_to_tda_features(self, feature_dict: Dict[str, torch.Tensor]) -> Dict:
        """Convert feature dictionary back to TDA features format."""
        # Simple conversion - stack all features
        feature_tensor = torch.stack(list(feature_dict.values()), dim=-1)
        return {'combined_features': feature_tensor}
    
    def _analyze_feature_correlations(self, feature_dict: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Analyze correlations between TDA features."""
        # Convert to numpy for correlation analysis
        feature_matrix = []
        feature_names = []
        
        for name, tensor in feature_dict.items():
            if tensor.numel() > 1:
                feature_matrix.append(tensor.flatten().detach().cpu().numpy())
                feature_names.append(name)
        
        if len(feature_matrix) < 2:
            return {'correlation_matrix': None, 'high_correlations': []}
        
        feature_matrix = np.array(feature_matrix).T
        
        # Compute correlation matrix
        correlation_matrix = np.corrcoef(feature_matrix.T)
        
        # Find high correlations
        high_correlations = []
        for i in range(len(feature_names)):
            for j in range(i + 1, len(feature_names)):
                corr = correlation_matrix[i, j]
                if abs(corr) > self.config.correlation_threshold:
                    high_correlations.append({
                        'feature1': feature_names[i],
                        'feature2': feature_names[j],
                        'correlation': corr
                    })
        
        return {
            'correlation_matrix': correlation_matrix,
            'feature_names': feature_names,
            'high_correlations': high_correlations
        }
    
    def _compute_feature_statistics(self, feature_dict: Dict[str, torch.Tensor]) -> Dict[str, Dict]:
        """Compute statistical properties of TDA features."""
        statistics = {}
        
        for name, tensor in feature_dict.items():
            tensor_np = tensor.detach().cpu().numpy()
            
            statistics[name] = {
                'mean': float(np.mean(tensor_np)),
                'std': float(np.std(tensor_np)),
                'min': float(np.min(tensor_np)),
                'max': float(np.max(tensor_np)),
                'median': float(np.median(tensor_np)),
                'skewness': float(self._safe_skewness(tensor_np)),
                'kurtosis': float(self._safe_kurtosis(tensor_np)),
                'non_zero_ratio': float(np.count_nonzero(tensor_np) / tensor_np.size)
            }
        
        return statistics
    
    def _safe_skewness(self, x: np.ndarray) -> float:
        """Safely compute skewness."""
        try:
            from scipy import stats
            return stats.skew(x.flatten())
        except:
            return 0.0
    
    def _safe_kurtosis(self, x: np.ndarray) -> float:
        """Safely compute kurtosis."""
        try:
            from scipy import stats
            return stats.kurtosis(x.flatten())
        except:
            return 0.0
    
    def _generate_interpretation_report(self, results: Dict[str, Any]) -> str:
        """Generate human-readable interpretation report."""
        report = []
        report.append("=== TDA Feature Importance Analysis Report ===\n")
        
        # Feature importance summary
        for method_name, importance_dict in results.items():
            if 'importance' in method_name and isinstance(importance_dict, dict):
                report.append(f"\n{method_name.replace('_', ' ').title()}:")
                
                # Sort features by importance
                sorted_features = sorted(importance_dict.items(), 
                                       key=lambda x: x[1], reverse=True)
                
                for i, (feature, importance) in enumerate(sorted_features[:self.config.top_k_features]):
                    report.append(f"  {i+1:2d}. {feature}: {importance:.4f}")
        
        # Correlation analysis
        if 'feature_correlations' in results:
            corr_data = results['feature_correlations']
            if corr_data['high_correlations']:
                report.append(f"\nHigh Feature Correlations (|r| > {self.config.correlation_threshold}):")
                for corr in corr_data['high_correlations'][:10]:  # Top 10
                    report.append(f"  {corr['feature1']} ↔ {corr['feature2']}: {corr['correlation']:.3f}")
        
        # Feature statistics summary
        if 'feature_statistics' in results:
            stats = results['feature_statistics']
            report.append(f"\nFeature Statistics Summary:")
            report.append(f"  Total features: {len(stats)}")
            
            # Find most variable features
            variability = [(name, data['std']) for name, data in stats.items()]
            variability.sort(key=lambda x: x[1], reverse=True)
            
            report.append(f"  Most variable features:")
            for name, std in variability[:5]:
                report.append(f"    {name}: std={std:.4f}")
        
        return "\n".join(report)


class AttentionVisualizer:
    """Visualizes attention patterns in the TDA-KAN_TDA model."""
    
    def __init__(self, config: InterpretabilityConfig):
        self.config = config
        
    def visualize_attention_patterns(self,
                                   model: nn.Module,
                                   x: torch.Tensor,
                                   tda_features: Dict,
                                   save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Visualize attention patterns in the model.
        
        Args:
            model: The TDA-KAN_TDA model
            x: Input time series data
            tda_features: TDA features
            save_path: Optional path to save visualizations
            
        Returns:
            Dictionary containing attention analysis results
        """
        model.eval()
        attention_data = {}
        
        # Hook to capture attention weights
        def attention_hook(name):
            def hook(module, input, output):
                if hasattr(output, 'attention_weights'):
                    attention_data[name] = output.attention_weights.detach().cpu()
                elif isinstance(output, tuple) and len(output) > 1:
                    attention_data[name] = output[1].detach().cpu()
            return hook
        
        # Register hooks
        hooks = []
        for name, module in model.named_modules():
            if 'attention' in name.lower() or 'fusion' in name.lower():
                hook = module.register_forward_hook(attention_hook(name))
                hooks.append(hook)
        
        try:
            # Forward pass
            with torch.no_grad():
                output = model(x, tda_features=tda_features)
            
            # Create visualizations
            if attention_data:
                self._create_attention_heatmaps(attention_data, save_path)
                self._create_attention_flow_diagram(attention_data, save_path)
                
        finally:
            # Remove hooks
            for hook in hooks:
                hook.remove()
        
        return {
            'attention_weights': attention_data,
            'visualization_paths': save_path
        }
    
    def _create_attention_heatmaps(self, attention_data: Dict, save_path: Optional[str]):
        """Create attention heatmap visualizations."""
        for name, weights in attention_data.items():
            if weights.dim() >= 2:
                plt.figure(figsize=self.config.figure_size)
                
                # Average over batch dimension if present
                if weights.dim() > 2:
                    weights_2d = weights.mean(dim=0)
                else:
                    weights_2d = weights
                
                # Create heatmap
                sns.heatmap(weights_2d.numpy(), 
                           cmap=self.config.color_palette,
                           cbar=True,
                           square=True)
                
                plt.title(f'Attention Weights: {name}')
                plt.xlabel('Key Dimension')
                plt.ylabel('Query Dimension')
                
                if save_path:
                    plt.savefig(f"{save_path}/attention_heatmap_{name}.{self.config.plot_format}",
                               dpi=self.config.dpi, bbox_inches='tight')
                
                if self.config.save_plots:
                    plt.show()
                else:
                    plt.close()
    
    def _create_attention_flow_diagram(self, attention_data: Dict, save_path: Optional[str]):
        """Create attention flow diagram."""
        if not attention_data:
            return
        
        fig, axes = plt.subplots(1, len(attention_data), 
                                figsize=(self.config.figure_size[0] * len(attention_data), 
                                        self.config.figure_size[1]))
        
        if len(attention_data) == 1:
            axes = [axes]
        
        for i, (name, weights) in enumerate(attention_data.items()):
            ax = axes[i]
            
            # Compute attention flow (sum over dimensions)
            if weights.dim() >= 2:
                flow = weights.sum(dim=-1).mean(dim=0) if weights.dim() > 2 else weights.sum(dim=-1)
                
                ax.plot(flow.numpy(), marker='o', linewidth=2, markersize=4)
                ax.set_title(f'Attention Flow: {name}')
                ax.set_xlabel('Position')
                ax.set_ylabel('Attention Weight')
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(f"{save_path}/attention_flow.{self.config.plot_format}",
                       dpi=self.config.dpi, bbox_inches='tight')
        
        if self.config.save_plots:
            plt.show()
        else:
            plt.close()


class TopologicalFeatureInterpreter:
    """Interprets topological features and their contributions."""
    
    def __init__(self, config: InterpretabilityConfig):
        self.config = config
        
    def interpret_topological_features(self, 
                                     tda_features: Dict,
                                     feature_importance: Dict[str, float],
                                     save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Interpret topological features and their meanings.
        
        Args:
            tda_features: TDA features dictionary
            feature_importance: Feature importance scores
            save_path: Optional path to save interpretations
            
        Returns:
            Dictionary containing interpretation results
        """
        interpretation = {}
        
        # Interpret persistence diagrams
        if 'persistence_diagrams' in tda_features:
            interpretation['persistence_interpretation'] = self._interpret_persistence_diagrams(
                tda_features['persistence_diagrams']
            )
        
        # Interpret landscape features
        if 'landscape_features' in tda_features:
            interpretation['landscape_interpretation'] = self._interpret_landscape_features(
                tda_features['landscape_features'], feature_importance
            )
        
        # Interpret embedding features
        if 'embedding_features' in tda_features:
            interpretation['embedding_interpretation'] = self._interpret_embedding_features(
                tda_features['embedding_features']
            )
        
        # Create visualizations
        if save_path:
            self._create_topological_visualizations(tda_features, interpretation, save_path)
        
        # Generate interpretation report
        interpretation['report'] = self._generate_topological_report(interpretation)
        
        return interpretation
    
    def _interpret_persistence_diagrams(self, diagrams: List) -> Dict[str, Any]:
        """Interpret persistence diagrams."""
        interpretation = {
            'total_diagrams': len(diagrams),
            'homology_analysis': {},
            'persistence_statistics': {}
        }
        
        for dim, diagram in enumerate(diagrams):
            if diagram is not None and len(diagram) > 0:
                # Convert to numpy if tensor
                if isinstance(diagram, torch.Tensor):
                    diagram_np = diagram.detach().cpu().numpy()
                else:
                    diagram_np = diagram
                
                # Compute persistence values
                persistence = diagram_np[:, 1] - diagram_np[:, 0]  # death - birth
                
                interpretation['homology_analysis'][f'H{dim}'] = {
                    'num_features': len(diagram_np),
                    'max_persistence': float(np.max(persistence)),
                    'mean_persistence': float(np.mean(persistence)),
                    'total_persistence': float(np.sum(persistence))
                }
                
                # Interpret meaning based on dimension
                if dim == 0:
                    interpretation['homology_analysis'][f'H{dim}']['meaning'] = \
                        "Connected components - represents trend changes and regime shifts"
                elif dim == 1:
                    interpretation['homology_analysis'][f'H{dim}']['meaning'] = \
                        "Loops/cycles - represents periodic patterns and oscillations"
                elif dim == 2:
                    interpretation['homology_analysis'][f'H{dim}']['meaning'] = \
                        "Voids - represents complex multi-dimensional patterns"
        
        return interpretation
    
    def _interpret_landscape_features(self, 
                                    landscapes: torch.Tensor,
                                    importance: Dict[str, float]) -> Dict[str, Any]:
        """Interpret persistence landscape features."""
        interpretation = {}
        
        if landscapes is not None:
            landscapes_np = landscapes.detach().cpu().numpy()
            
            # Basic statistics
            interpretation['shape'] = landscapes_np.shape
            interpretation['statistics'] = {
                'mean': float(np.mean(landscapes_np)),
                'std': float(np.std(landscapes_np)),
                'max': float(np.max(landscapes_np)),
                'min': float(np.min(landscapes_np))
            }
            
            # Identify most important landscape features
            landscape_importance = {k: v for k, v in importance.items() 
                                  if 'landscape' in k.lower()}
            
            if landscape_importance:
                sorted_importance = sorted(landscape_importance.items(), 
                                         key=lambda x: x[1], reverse=True)
                interpretation['most_important_features'] = sorted_importance[:10]
            
            # Interpret landscape meaning
            interpretation['meaning'] = {
                'description': "Persistence landscapes provide statistical summaries of topological features",
                'interpretation': "Higher values indicate stronger topological signals",
                'usage': "Used for machine learning integration of topological features"
            }
        
        return interpretation
    
    def _interpret_embedding_features(self, embeddings: torch.Tensor) -> Dict[str, Any]:
        """Interpret Takens embedding features."""
        interpretation = {}
        
        if embeddings is not None:
            embeddings_np = embeddings.detach().cpu().numpy()
            
            interpretation['embedding_statistics'] = {
                'dimensions': embeddings_np.shape,
                'mean_distance': float(np.mean(np.linalg.norm(embeddings_np, axis=-1))),
                'spread': float(np.std(embeddings_np)),
                'density': float(embeddings_np.size / np.prod(embeddings_np.shape))
            }
            
            interpretation['meaning'] = {
                'description': "Takens embedding reconstructs phase space from time series",
                'interpretation': "Captures dynamical system properties and temporal dependencies",
                'quality_indicators': "Higher spread and appropriate density indicate good reconstruction"
            }
        
        return interpretation
    
    def _create_topological_visualizations(self, 
                                         tda_features: Dict,
                                         interpretation: Dict,
                                         save_path: str):
        """Create visualizations for topological features."""
        # Create persistence diagram plots
        if 'persistence_diagrams' in tda_features:
            self._plot_persistence_diagrams(tda_features['persistence_diagrams'], save_path)
        
        # Create landscape plots
        if 'landscape_features' in tda_features:
            self._plot_landscape_features(tda_features['landscape_features'], save_path)
    
    def _plot_persistence_diagrams(self, diagrams: List, save_path: str):
        """Plot persistence diagrams."""
        fig, axes = plt.subplots(1, len(diagrams), 
                                figsize=(self.config.figure_size[0] * len(diagrams), 
                                        self.config.figure_size[1]))
        
        if len(diagrams) == 1:
            axes = [axes]
        
        for dim, (ax, diagram) in enumerate(zip(axes, diagrams)):
            if diagram is not None and len(diagram) > 0:
                if isinstance(diagram, torch.Tensor):
                    diagram_np = diagram.detach().cpu().numpy()
                else:
                    diagram_np = diagram
                
                # Plot points
                ax.scatter(diagram_np[:, 0], diagram_np[:, 1], 
                          alpha=0.7, s=50, c=f'C{dim}')
                
                # Plot diagonal
                max_val = max(diagram_np.max(), 1.0)
                ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.5)
                
                ax.set_xlabel('Birth')
                ax.set_ylabel('Death')
                ax.set_title(f'H{dim} Persistence Diagram')
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{save_path}/persistence_diagrams.{self.config.plot_format}",
                   dpi=self.config.dpi, bbox_inches='tight')
        
        if self.config.save_plots:
            plt.show()
        else:
            plt.close()
    
    def _plot_landscape_features(self, landscapes: torch.Tensor, save_path: str):
        """Plot persistence landscape features."""
        if landscapes is None:
            return
        
        landscapes_np = landscapes.detach().cpu().numpy()
        
        plt.figure(figsize=self.config.figure_size)
        
        # Plot first few landscapes
        num_landscapes = min(5, landscapes_np.shape[-1])
        for i in range(num_landscapes):
            if landscapes_np.ndim == 3:  # [batch, time, features]
                landscape_data = landscapes_np[0, :, i]  # First batch
            else:
                landscape_data = landscapes_np[:, i]
            
            plt.plot(landscape_data, label=f'Landscape {i+1}', alpha=0.7)
        
        plt.xlabel('Position')
        plt.ylabel('Landscape Value')
        plt.title('Persistence Landscapes')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.savefig(f"{save_path}/landscape_features.{self.config.plot_format}",
                   dpi=self.config.dpi, bbox_inches='tight')
        
        if self.config.save_plots:
            plt.show()
        else:
            plt.close()
    
    def _generate_topological_report(self, interpretation: Dict) -> str:
        """Generate topological interpretation report."""
        report = []
        report.append("=== Topological Feature Interpretation Report ===\n")
        
        # Persistence diagram interpretation
        if 'persistence_interpretation' in interpretation:
            pers_interp = interpretation['persistence_interpretation']
            report.append("Persistence Diagram Analysis:")
            
            for dim_name, analysis in pers_interp.get('homology_analysis', {}).items():
                report.append(f"\n  {dim_name} ({analysis.get('meaning', 'Unknown')}):")
                report.append(f"    Features detected: {analysis.get('num_features', 0)}")
                report.append(f"    Max persistence: {analysis.get('max_persistence', 0):.4f}")
                report.append(f"    Mean persistence: {analysis.get('mean_persistence', 0):.4f}")
        
        # Landscape interpretation
        if 'landscape_interpretation' in interpretation:
            land_interp = interpretation['landscape_interpretation']
            report.append(f"\nPersistence Landscape Analysis:")
            
            if 'statistics' in land_interp:
                stats = land_interp['statistics']
                report.append(f"  Statistical summary:")
                report.append(f"    Mean value: {stats.get('mean', 0):.4f}")
                report.append(f"    Standard deviation: {stats.get('std', 0):.4f}")
                report.append(f"    Value range: [{stats.get('min', 0):.4f}, {stats.get('max', 0):.4f}]")
        
        return "\n".join(report)


# Convenience functions
def analyze_model_interpretability(model: nn.Module,
                                 x: torch.Tensor,
                                 tda_features: Dict,
                                 target: Optional[torch.Tensor] = None,
                                 save_dir: str = "interpretability_results") -> Dict[str, Any]:
    """
    Comprehensive interpretability analysis of TDA-KAN_TDA model.
    
    Args:
        model: The TDA-KAN_TDA model
        x: Input time series data
        tda_features: TDA features
        target: Optional target values
        save_dir: Directory to save results
        
    Returns:
        Complete interpretability analysis results
    """
    config = InterpretabilityConfig(output_dir=save_dir)
    
    # Create output directory
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    # Feature importance analysis
    importance_analyzer = TDAFeatureImportanceAnalyzer(config)
    results['feature_importance'] = importance_analyzer.analyze_feature_importance(
        model, x, tda_features, target
    )
    
    # Attention visualization
    attention_visualizer = AttentionVisualizer(config)
    results['attention_analysis'] = attention_visualizer.visualize_attention_patterns(
        model, x, tda_features, save_dir
    )
    
    # Topological interpretation
    topo_interpreter = TopologicalFeatureInterpreter(config)
    importance_scores = results['feature_importance'].get('gradient_importance', {})
    results['topological_interpretation'] = topo_interpreter.interpret_topological_features(
        tda_features, importance_scores, save_dir
    )
    
    # Save comprehensive report
    if config.save_detailed_reports:
        report_path = Path(save_dir) / "interpretability_report.txt"
        with open(report_path, 'w') as f:
            f.write("=== TDA-KAN_TDA Interpretability Analysis ===\n\n")
            
            for section_name, section_data in results.items():
                f.write(f"\n{'='*50}\n")
                f.write(f"{section_name.upper()}\n")
                f.write(f"{'='*50}\n")
                
                if isinstance(section_data, dict) and 'report' in section_data:
                    f.write(section_data['report'])
                elif isinstance(section_data, dict) and 'interpretation_report' in section_data:
                    f.write(section_data['interpretation_report'])
                else:
                    f.write(f"Data: {type(section_data)}\n")
    
    return results


def create_interpretability_dashboard(results: Dict[str, Any], 
                                    save_path: str = "interpretability_dashboard.html"):
    """Create an interactive HTML dashboard for interpretability results."""
    # This would create an interactive dashboard
    # Implementation would depend on preferred visualization library (plotly, bokeh, etc.)
    logger.info(f"Dashboard creation not implemented yet. Results saved to {save_path}")
    return save_path 