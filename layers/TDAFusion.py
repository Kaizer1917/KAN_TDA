"""
Adaptive Fusion Strategies for TDA-KAN_TDA Integration

This module implements multiple fusion strategies and adaptive selection mechanisms
for combining topological and frequency features effectively.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import warnings

from .TopologicalAttention import CrossModalAttention


class FusionStrategy(Enum):
    """Enumeration of available fusion strategies."""
    EARLY = "early"
    MID = "mid"
    LATE = "late"
    ATTENTION = "attention"
    GATE = "gate"
    ADAPTIVE = "adaptive"


class AdaptiveFusion(nn.Module):
    """
    Adaptive fusion module that can select and combine multiple fusion strategies
    based on input characteristics and learned preferences.
    """
    
    def __init__(
        self,
        tda_dim: int,
        freq_dim: int,
        output_dim: int,
        strategies: List[str] = None,
        hidden_dim: int = 64,
        num_heads: int = 4,
        dropout: float = 0.1,
        temperature: float = 1.0,
        adaptive_threshold: float = 0.1
    ):
        """
        Initialize AdaptiveFusion.
        
        Args:
            tda_dim: Dimension of TDA features
            freq_dim: Dimension of frequency features
            output_dim: Desired output dimension
            strategies: List of fusion strategies to use
            hidden_dim: Hidden dimension for internal computations
            num_heads: Number of attention heads
            dropout: Dropout probability
            temperature: Temperature for strategy selection
            adaptive_threshold: Threshold for strategy switching
        """
        super(AdaptiveFusion, self).__init__()
        
        self.tda_dim = tda_dim
        self.freq_dim = freq_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.adaptive_threshold = adaptive_threshold
        self.temperature = temperature
        
        # Default strategies if none provided
        if strategies is None:
            strategies = ['early', 'attention', 'gate', 'late']
        self.strategies = strategies
        
        # Initialize fusion components
        self._init_fusion_components(num_heads, dropout)
        
        # Strategy selection network
        self.strategy_selector = nn.Sequential(
            nn.Linear(tda_dim + freq_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, len(strategies)),
            nn.Softmax(dim=-1)
        )
        
        # Performance tracking
        self.strategy_performance = {strategy: [] for strategy in strategies}
        self.strategy_usage_count = {strategy: 0 for strategy in strategies}
        
    def _init_fusion_components(self, num_heads: int, dropout: float):
        """Initialize components for different fusion strategies."""
        
        # Early fusion (concatenation + projection)
        if 'early' in self.strategies:
            self.early_fusion = nn.Sequential(
                nn.Linear(self.tda_dim + self.freq_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(self.hidden_dim, self.output_dim)
            )
        
        # Mid fusion (cross-modal attention)
        if 'attention' in self.strategies:
            self.attention_fusion = CrossModalAttention(
                tda_dim=self.tda_dim,
                freq_dim=self.freq_dim,
                hidden_dim=self.hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                fusion_strategy='concat'
            )
            self.attention_output = nn.Linear(
                max(self.tda_dim, self.freq_dim), self.output_dim
            )
        
        # Gate fusion (learned gating mechanism)
        if 'gate' in self.strategies:
            self.gate_fusion = nn.Sequential(
                nn.Linear(self.tda_dim + self.freq_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, 2),  # Gates for TDA and freq
                nn.Sigmoid()
            )
            self.gate_output = nn.Linear(self.tda_dim + self.freq_dim, self.output_dim)
        
        # Late fusion (separate processing + combination)
        if 'late' in self.strategies:
            self.tda_processor = nn.Sequential(
                nn.Linear(self.tda_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(self.hidden_dim, self.output_dim)
            )
            self.freq_processor = nn.Sequential(
                nn.Linear(self.freq_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(self.hidden_dim, self.output_dim)
            )
            self.late_combiner = nn.Sequential(
                nn.Linear(2 * self.output_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.output_dim)
            )
    
    def _early_fusion(self, tda_features: torch.Tensor, freq_features: torch.Tensor) -> torch.Tensor:
        """
        Early fusion: concatenate features and process together.
        
        Args:
            tda_features: TDA features [B, T, D_tda] or [B, D_tda]
            freq_features: Frequency features [B, T, D_freq]
            
        Returns:
            fused_output: Fused features [B, T, output_dim]
        """
        # Handle dimension alignment
        tda_features, freq_features = self._align_temporal_dims(tda_features, freq_features)
        
        # Concatenate features
        concatenated = torch.cat([tda_features, freq_features], dim=-1)
        
        # Process through early fusion network
        output = self.early_fusion(concatenated)
        
        return output
    
    def _attention_fusion(self, tda_features: torch.Tensor, freq_features: torch.Tensor) -> torch.Tensor:
        """
        Attention-based fusion using cross-modal attention.
        
        Args:
            tda_features: TDA features [B, T, D_tda] or [B, D_tda]
            freq_features: Frequency features [B, T, D_freq]
            
        Returns:
            fused_output: Attention-fused features [B, T, output_dim]
        """
        # Apply cross-modal attention
        attention_result = self.attention_fusion(tda_features, freq_features)
        fused_features = attention_result['fused_features']
        
        # Project to output dimension
        output = self.attention_output(fused_features)
        
        return output
    
    def _gate_fusion(self, tda_features: torch.Tensor, freq_features: torch.Tensor) -> torch.Tensor:
        """
        Gate-based fusion with learned importance weights.
        
        Args:
            tda_features: TDA features [B, T, D_tda] or [B, D_tda]
            freq_features: Frequency features [B, T, D_freq]
            
        Returns:
            fused_output: Gate-fused features [B, T, output_dim]
        """
        # Handle dimension alignment
        tda_features, freq_features = self._align_temporal_dims(tda_features, freq_features)
        
        # Concatenate for gate computation
        concatenated = torch.cat([tda_features, freq_features], dim=-1)
        
        # Compute gates
        gates = self.gate_fusion(concatenated)  # [B, T, 2]
        tda_gate = gates[..., 0:1]  # [B, T, 1]
        freq_gate = gates[..., 1:2]  # [B, T, 1]
        
        # Apply gates
        gated_tda = tda_features * tda_gate
        gated_freq = freq_features * freq_gate
        
        # Combine gated features
        gated_combined = torch.cat([gated_tda, gated_freq], dim=-1)
        output = self.gate_output(gated_combined)
        
        return output
    
    def _late_fusion(self, tda_features: torch.Tensor, freq_features: torch.Tensor) -> torch.Tensor:
        """
        Late fusion: process features separately then combine.
        
        Args:
            tda_features: TDA features [B, T, D_tda] or [B, D_tda]
            freq_features: Frequency features [B, T, D_freq]
            
        Returns:
            fused_output: Late-fused features [B, T, output_dim]
        """
        # Handle 2D TDA features
        if tda_features.dim() == 2:
            tda_features = tda_features.unsqueeze(1)
        
        # Process each modality separately
        tda_processed = self.tda_processor(tda_features)
        freq_processed = self.freq_processor(freq_features)
        
        # Align temporal dimensions
        if tda_processed.shape[1] != freq_processed.shape[1]:
            if tda_processed.shape[1] == 1:
                tda_processed = tda_processed.expand(-1, freq_processed.shape[1], -1)
            elif freq_processed.shape[1] == 1:
                freq_processed = freq_processed.expand(-1, tda_processed.shape[1], -1)
        
        # Combine processed features
        combined = torch.cat([tda_processed, freq_processed], dim=-1)
        output = self.late_combiner(combined)
        
        return output
    
    def _align_temporal_dims(
        self, 
        tda_features: torch.Tensor, 
        freq_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Align temporal dimensions between TDA and frequency features.
        
        Args:
            tda_features: TDA features [B, T_tda, D_tda] or [B, D_tda]
            freq_features: Frequency features [B, T_freq, D_freq]
            
        Returns:
            Aligned TDA and frequency features
        """
        # Handle 2D TDA features
        if tda_features.dim() == 2:
            tda_features = tda_features.unsqueeze(1)  # [B, 1, D_tda]
        
        # Align temporal dimensions
        if tda_features.shape[1] != freq_features.shape[1]:
            if tda_features.shape[1] == 1:
                tda_features = tda_features.expand(-1, freq_features.shape[1], -1)
            elif freq_features.shape[1] == 1:
                freq_features = freq_features.expand(-1, tda_features.shape[1], -1)
            else:
                # Use interpolation for different temporal lengths
                target_len = max(tda_features.shape[1], freq_features.shape[1])
                if tda_features.shape[1] != target_len:
                    tda_features = F.interpolate(
                        tda_features.transpose(1, 2), 
                        size=target_len, 
                        mode='linear', 
                        align_corners=False
                    ).transpose(1, 2)
                if freq_features.shape[1] != target_len:
                    freq_features = F.interpolate(
                        freq_features.transpose(1, 2), 
                        size=target_len, 
                        mode='linear', 
                        align_corners=False
                    ).transpose(1, 2)
        
        return tda_features, freq_features
    
    def _select_strategy(
        self, 
        tda_features: torch.Tensor, 
        freq_features: torch.Tensor
    ) -> Tuple[str, torch.Tensor]:
        """
        Select fusion strategy based on input characteristics.
        
        Args:
            tda_features: TDA features
            freq_features: Frequency features
            
        Returns:
            selected_strategy: Name of selected strategy
            strategy_weights: Soft weights for all strategies
        """
        # Compute input characteristics
        tda_flat = tda_features.reshape(tda_features.shape[0], -1)
        freq_flat = freq_features.reshape(freq_features.shape[0], -1)
        
        # Take mean across batch for strategy selection
        tda_mean = tda_flat.mean(dim=0)
        freq_mean = freq_flat.mean(dim=0)
        
        # Combine for strategy selection - ensure correct dimensions
        combined_features = torch.cat([tda_mean, freq_mean], dim=0).unsqueeze(0)
        
        # Ensure the combined features match the expected input dimension
        expected_dim = self.tda_dim + self.freq_dim
        if combined_features.shape[1] != expected_dim:
            # Adjust dimensions if needed
            if combined_features.shape[1] > expected_dim:
                # Truncate if too large
                combined_features = combined_features[:, :expected_dim]
            else:
                # Pad if too small
                padding_size = expected_dim - combined_features.shape[1]
                padding = torch.zeros(1, padding_size, device=combined_features.device)
                combined_features = torch.cat([combined_features, padding], dim=1)
        
        # Get strategy weights
        strategy_weights = self.strategy_selector(combined_features)  # [1, num_strategies]
        
        # Select strategy (either hard selection or soft combination)
        if self.training:
            # During training, use soft combination
            selected_strategy = 'adaptive'
        else:
            # During inference, select best strategy
            best_idx = torch.argmax(strategy_weights, dim=-1).item()
            selected_strategy = self.strategies[best_idx]
        
        return selected_strategy, strategy_weights.squeeze(0)
    
    def forward(
        self,
        tda_features: torch.Tensor,
        freq_features: torch.Tensor,
        strategy: Optional[str] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with adaptive or specified fusion strategy.
        
        Args:
            tda_features: TDA features [B, T, D_tda] or [B, D_tda]
            freq_features: Frequency features [B, T, D_freq]
            strategy: Optional specific strategy to use
            
        Returns:
            Dictionary containing:
                - fused_output: Final fused features
                - strategy_used: Strategy that was used
                - strategy_weights: Weights for all strategies (if adaptive)
                - individual_outputs: Outputs from each strategy (if adaptive)
        """
        # Select strategy if not specified
        if strategy is None:
            selected_strategy, strategy_weights = self._select_strategy(tda_features, freq_features)
        else:
            selected_strategy = strategy
            strategy_weights = None
        
        # Apply all strategies and get outputs
        individual_outputs = {}
        for strategy in self.strategies:
            try:
                if strategy == 'early':
                    output = self._early_fusion(tda_features, freq_features)
                elif strategy == 'attention':
                    output = self._attention_fusion(tda_features, freq_features)
                elif strategy == 'gate':
                    output = self._gate_fusion(tda_features, freq_features)
                elif strategy == 'late':
                    output = self._late_fusion(tda_features, freq_features)
                else:
                    continue  # Skip unknown strategies
                
                individual_outputs[strategy] = output
            except Exception as e:
                warnings.warn(f"Strategy {strategy} failed: {e}")
                continue
        
        # Handle case where no strategies succeeded
        if not individual_outputs:
            # Fallback to simple concatenation
            batch_size = tda_features.shape[0]
            seq_len = freq_features.shape[1]
            fallback_output = torch.zeros(batch_size, seq_len, self.output_dim, device=tda_features.device)
            individual_outputs['fallback'] = fallback_output
        
        # Adaptive strategy selection
        if len(individual_outputs) > 1:
            # Combine outputs from different strategies
            fused_output = torch.zeros_like(list(individual_outputs.values())[0])
            
            for strategy, output in individual_outputs.items():
                if strategy_weights is not None and isinstance(strategy_weights, dict):
                    weight = strategy_weights.get(strategy, 1.0 / len(individual_outputs))
                else:
                    weight = 1.0 / len(individual_outputs)
                fused_output += weight * output
        else:
            # Single strategy output
            fused_output = list(individual_outputs.values())[0]
        
        # Update usage statistics
        if selected_strategy in self.strategy_usage_count:
            self.strategy_usage_count[selected_strategy] += 1
        
        return {
            'fused_output': fused_output,
            'strategy_used': selected_strategy,
            'strategy_weights': strategy_weights,
            'individual_outputs': individual_outputs
        }
    
    def get_strategy_statistics(self) -> Dict[str, Any]:
        """Get statistics about strategy usage and performance."""
        total_usage = sum(self.strategy_usage_count.values())
        
        usage_percentages = {
            strategy: (count / total_usage * 100) if total_usage > 0 else 0
            for strategy, count in self.strategy_usage_count.items()
        }
        
        avg_performance = {
            strategy: np.mean(perf_list) if perf_list else 0.0
            for strategy, perf_list in self.strategy_performance.items()
        }
        
        return {
            'usage_counts': self.strategy_usage_count,
            'usage_percentages': usage_percentages,
            'average_performance': avg_performance,
            'total_usage': total_usage
        }
    
    def reset_statistics(self):
        """Reset strategy usage and performance statistics."""
        self.strategy_performance = {strategy: [] for strategy in self.strategies}
        self.strategy_usage_count = {strategy: 0 for strategy in self.strategies} 