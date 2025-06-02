"""
Topological Attention Mechanisms for TDA-KAN_TDA Integration

This module implements cross-modal attention mechanisms that enable effective fusion
between topological features (from TDA) and frequency features (from KAN_TDA).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict, Any


class CrossModalAttention(nn.Module):
    """
    Cross-modal attention mechanism for fusing TDA and frequency features.
    
    Enables bidirectional attention between topological features and frequency
    representations, allowing each modality to attend to relevant information
    in the other modality.
    """
    
    def __init__(
        self,
        tda_dim: int,
        freq_dim: int,
        hidden_dim: int = 64,
        num_heads: int = 4,
        dropout: float = 0.1,
        temperature: float = 1.0,
        fusion_strategy: str = 'concat'
    ):
        """
        Initialize CrossModalAttention.
        
        Args:
            tda_dim: Dimension of TDA features
            freq_dim: Dimension of frequency features  
            hidden_dim: Hidden dimension for attention computation
            num_heads: Number of attention heads
            dropout: Dropout probability
            temperature: Temperature for attention softmax
            fusion_strategy: How to combine attended features ('concat', 'add', 'gate')
        """
        super(CrossModalAttention, self).__init__()
        
        self.tda_dim = tda_dim
        self.freq_dim = freq_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.temperature = temperature
        self.fusion_strategy = fusion_strategy
        
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        # TDA to Frequency attention (TDA queries, Frequency keys/values)
        self.tda_to_freq_query = nn.Linear(tda_dim, hidden_dim)
        self.tda_to_freq_key = nn.Linear(freq_dim, hidden_dim)
        self.tda_to_freq_value = nn.Linear(freq_dim, hidden_dim)
        
        # Frequency to TDA attention (Frequency queries, TDA keys/values)
        self.freq_to_tda_query = nn.Linear(freq_dim, hidden_dim)
        self.freq_to_tda_key = nn.Linear(tda_dim, hidden_dim)
        self.freq_to_tda_value = nn.Linear(tda_dim, hidden_dim)
        
        # Output projections
        self.tda_output_proj = nn.Linear(hidden_dim, tda_dim)
        self.freq_output_proj = nn.Linear(hidden_dim, freq_dim)
        
        # Fusion layers based on strategy
        if fusion_strategy == 'concat':
            self.fusion_layer = nn.Linear(tda_dim + freq_dim, max(tda_dim, freq_dim))
        elif fusion_strategy == 'gate':
            self.tda_gate = nn.Sequential(
                nn.Linear(tda_dim + freq_dim, tda_dim),
                nn.Sigmoid()
            )
            self.freq_gate = nn.Sequential(
                nn.Linear(tda_dim + freq_dim, freq_dim),
                nn.Sigmoid()
            )
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm_tda = nn.LayerNorm(tda_dim)
        self.layer_norm_freq = nn.LayerNorm(freq_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier uniform initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def _multi_head_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute multi-head attention.
        
        Args:
            query: Query tensor [B, T_q, D]
            key: Key tensor [B, T_k, D]
            value: Value tensor [B, T_v, D]
            mask: Optional attention mask [B, T_q, T_k]
            
        Returns:
            attended_output: Attended features [B, T_q, D]
            attention_weights: Attention weights [B, H, T_q, T_k]
        """
        B, T_q, _ = query.shape
        T_k = key.shape[1]
        
        # Reshape for multi-head attention
        query = query.view(B, T_q, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, T_q, D_h]
        key = key.view(B, T_k, self.num_heads, self.head_dim).transpose(1, 2)      # [B, H, T_k, D_h]
        value = value.view(B, T_k, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, T_k, D_h]
        
        # Compute attention scores
        scores = torch.matmul(query, key.transpose(-2, -1)) / (math.sqrt(self.head_dim) * self.temperature)
        
        # Apply mask if provided
        if mask is not None:
            mask = mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
            scores.masked_fill_(mask == 0, float('-inf'))
        
        # Apply softmax
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attended = torch.matmul(attention_weights, value)  # [B, H, T_q, D_h]
        
        # Concatenate heads
        attended = attended.transpose(1, 2).contiguous().view(B, T_q, self.hidden_dim)
        
        return attended, attention_weights
    
    def forward(
        self,
        tda_features: torch.Tensor,
        freq_features: torch.Tensor,
        tda_mask: Optional[torch.Tensor] = None,
        freq_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of cross-modal attention.
        
        Args:
            tda_features: TDA features [B, T_tda, D_tda] or [B, D_tda]
            freq_features: Frequency features [B, T_freq, D_freq]
            tda_mask: Optional mask for TDA features
            freq_mask: Optional mask for frequency features
            
        Returns:
            Dictionary containing:
                - enhanced_tda: TDA features enhanced with frequency information
                - enhanced_freq: Frequency features enhanced with TDA information
                - fused_features: Combined features based on fusion strategy
                - attention_weights: Attention weight matrices for analysis
        """
        # Handle 2D TDA features (expand temporal dimension)
        if tda_features.dim() == 2:
            tda_features = tda_features.unsqueeze(1)  # [B, 1, D_tda]
        
        B, T_tda, D_tda = tda_features.shape
        B, T_freq, D_freq = freq_features.shape
        
        # Store original features for residual connections
        tda_residual = tda_features
        freq_residual = freq_features
        
        # TDA to Frequency attention (TDA attends to frequency features)
        tda_query = self.tda_to_freq_query(tda_features)  # [B, T_tda, H]
        freq_key = self.tda_to_freq_key(freq_features)    # [B, T_freq, H]
        freq_value = self.tda_to_freq_value(freq_features) # [B, T_freq, H]
        
        tda_attended, tda_to_freq_weights = self._multi_head_attention(
            tda_query, freq_key, freq_value, freq_mask
        )
        
        # Project back to TDA dimension
        tda_attended = self.tda_output_proj(tda_attended)  # [B, T_tda, D_tda]
        
        # Frequency to TDA attention (Frequency attends to TDA features)
        freq_query = self.freq_to_tda_query(freq_features)  # [B, T_freq, H]
        tda_key = self.freq_to_tda_key(tda_features)        # [B, T_tda, H]
        tda_value = self.freq_to_tda_value(tda_features)    # [B, T_tda, H]
        
        freq_attended, freq_to_tda_weights = self._multi_head_attention(
            freq_query, tda_key, tda_value, tda_mask
        )
        
        # Project back to frequency dimension
        freq_attended = self.freq_output_proj(freq_attended)  # [B, T_freq, D_freq]
        
        # Apply residual connections and layer normalization
        enhanced_tda = self.layer_norm_tda(tda_residual + tda_attended)
        enhanced_freq = self.layer_norm_freq(freq_residual + freq_attended)
        
        # Fuse features based on strategy
        fused_features = self._fuse_features(enhanced_tda, enhanced_freq)
        
        return {
            'enhanced_tda': enhanced_tda,
            'enhanced_freq': enhanced_freq,
            'fused_features': fused_features,
            'attention_weights': {
                'tda_to_freq': tda_to_freq_weights,
                'freq_to_tda': freq_to_tda_weights
            }
        }
    
    def _fuse_features(
        self,
        tda_features: torch.Tensor,
        freq_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Fuse TDA and frequency features based on fusion strategy.
        
        Args:
            tda_features: Enhanced TDA features [B, T_tda, D_tda]
            freq_features: Enhanced frequency features [B, T_freq, D_freq]
            
        Returns:
            fused_features: Combined features
        """
        # Handle temporal dimension mismatch
        if tda_features.shape[1] == 1 and freq_features.shape[1] > 1:
            # Expand TDA features to match frequency temporal dimension
            tda_features = tda_features.expand(-1, freq_features.shape[1], -1)
        elif freq_features.shape[1] == 1 and tda_features.shape[1] > 1:
            # Expand frequency features to match TDA temporal dimension
            freq_features = freq_features.expand(-1, tda_features.shape[1], -1)
        
        if self.fusion_strategy == 'concat':
            # Concatenate and project
            concatenated = torch.cat([tda_features, freq_features], dim=-1)
            fused = self.fusion_layer(concatenated)
            
        elif self.fusion_strategy == 'add':
            # Element-wise addition (requires same dimensions)
            if tda_features.shape[-1] != freq_features.shape[-1]:
                # Project to common dimension
                common_dim = max(tda_features.shape[-1], freq_features.shape[-1])
                if tda_features.shape[-1] != common_dim:
                    tda_features = F.linear(tda_features, 
                                          torch.randn(common_dim, tda_features.shape[-1], 
                                                    device=tda_features.device))
                if freq_features.shape[-1] != common_dim:
                    freq_features = F.linear(freq_features,
                                           torch.randn(common_dim, freq_features.shape[-1],
                                                     device=freq_features.device))
            fused = tda_features + freq_features
            
        elif self.fusion_strategy == 'gate':
            # Gated fusion
            concatenated = torch.cat([tda_features, freq_features], dim=-1)
            tda_gate = self.tda_gate(concatenated)
            freq_gate = self.freq_gate(concatenated)
            
            gated_tda = tda_features * tda_gate
            gated_freq = freq_features * freq_gate
            
            # Concatenate gated features
            fused = torch.cat([gated_tda, gated_freq], dim=-1)
            
        else:
            raise ValueError(f"Unknown fusion strategy: {self.fusion_strategy}")
        
        return fused
    
    def get_attention_info(self) -> Dict[str, Any]:
        """Get information about attention configuration."""
        return {
            'tda_dim': self.tda_dim,
            'freq_dim': self.freq_dim,
            'hidden_dim': self.hidden_dim,
            'num_heads': self.num_heads,
            'head_dim': self.head_dim,
            'fusion_strategy': self.fusion_strategy,
            'temperature': self.temperature
        } 