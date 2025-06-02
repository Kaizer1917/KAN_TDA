"""
Memory Optimization Utilities for TDA-KAN_TDA Fusion

This module provides utilities for optimizing memory usage during fusion operations,
including gradient checkpointing, memory-efficient attention, and dynamic batching.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import math
from typing import Optional, Tuple, Dict, Any, List
import gc
import psutil
import warnings


class MemoryOptimizedAttention(nn.Module):
    """
    Memory-optimized attention mechanism that reduces memory usage
    through chunked computation and gradient checkpointing.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        chunk_size: int = 512,
        use_checkpointing: bool = True,
        dropout: float = 0.1
    ):
        """
        Initialize memory-optimized attention.
        
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            chunk_size: Size of chunks for memory-efficient computation
            use_checkpointing: Whether to use gradient checkpointing
            dropout: Dropout probability
        """
        super(MemoryOptimizedAttention, self).__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.chunk_size = chunk_size
        self.use_checkpointing = use_checkpointing
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
    def _chunked_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute attention in chunks to reduce memory usage.
        
        Args:
            query: Query tensor [B, H, T_q, D_h]
            key: Key tensor [B, H, T_k, D_h]
            value: Value tensor [B, H, T_k, D_h]
            mask: Optional attention mask
            
        Returns:
            Attention output [B, H, T_q, D_h]
        """
        B, H, T_q, D_h = query.shape
        T_k = key.shape[2]
        
        # Initialize output tensor
        output = torch.zeros_like(query)
        
        # Process in chunks
        for i in range(0, T_q, self.chunk_size):
            end_i = min(i + self.chunk_size, T_q)
            query_chunk = query[:, :, i:end_i, :]  # [B, H, chunk_size, D_h]
            
            # Compute attention scores for this chunk
            scores = torch.matmul(query_chunk, key.transpose(-2, -1)) * self.scale
            
            # Apply mask if provided
            if mask is not None:
                mask_chunk = mask[:, :, i:end_i, :]
                scores.masked_fill_(mask_chunk == 0, float('-inf'))
            
            # Apply softmax
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            
            # Apply attention to values
            output_chunk = torch.matmul(attn_weights, value)
            output[:, :, i:end_i, :] = output_chunk
            
            # Clear intermediate tensors to free memory
            del scores, attn_weights, output_chunk
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return output
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with memory optimization.
        
        Args:
            query: Query tensor [B, T_q, D]
            key: Key tensor [B, T_k, D]
            value: Value tensor [B, T_k, D]
            mask: Optional attention mask
            
        Returns:
            Attention output [B, T_q, D]
        """
        B, T_q, D = query.shape
        T_k = key.shape[1]
        
        # Project to Q, K, V
        Q = self.q_proj(query)
        K = self.k_proj(key)
        V = self.v_proj(value)
        
        # Reshape for multi-head attention
        Q = Q.view(B, T_q, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, T_k, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, T_k, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention with or without checkpointing
        if self.use_checkpointing and self.training:
            attn_output = checkpoint(self._chunked_attention, Q, K, V, mask)
        else:
            attn_output = self._chunked_attention(Q, K, V, mask)
        
        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T_q, D)
        
        # Final projection
        output = self.out_proj(attn_output)
        
        return output


class MemoryMonitor:
    """
    Utility class for monitoring memory usage during fusion operations.
    """
    
    def __init__(self, device: torch.device):
        """
        Initialize memory monitor.
        
        Args:
            device: Device to monitor (CPU or CUDA)
        """
        self.device = device
        self.is_cuda = device.type == 'cuda'
        self.memory_history = []
        
    def get_memory_usage(self) -> Dict[str, float]:
        """
        Get current memory usage statistics.
        
        Returns:
            Dictionary with memory usage information
        """
        if self.is_cuda:
            allocated = torch.cuda.memory_allocated(self.device) / 1024**3  # GB
            reserved = torch.cuda.memory_reserved(self.device) / 1024**3   # GB
            max_allocated = torch.cuda.max_memory_allocated(self.device) / 1024**3  # GB
            
            return {
                'allocated_gb': allocated,
                'reserved_gb': reserved,
                'max_allocated_gb': max_allocated,
                'free_gb': reserved - allocated
            }
        else:
            # CPU memory monitoring
            process = psutil.Process()
            memory_info = process.memory_info()
            
            return {
                'rss_gb': memory_info.rss / 1024**3,  # Resident Set Size
                'vms_gb': memory_info.vms / 1024**3,  # Virtual Memory Size
                'percent': process.memory_percent()
            }
    
    def log_memory(self, tag: str = ""):
        """
        Log current memory usage with optional tag.
        
        Args:
            tag: Optional tag to identify the measurement point
        """
        usage = self.get_memory_usage()
        usage['tag'] = tag
        self.memory_history.append(usage)
        
        if self.is_cuda:
            print(f"[{tag}] GPU Memory - Allocated: {usage['allocated_gb']:.2f}GB, "
                  f"Reserved: {usage['reserved_gb']:.2f}GB")
        else:
            print(f"[{tag}] CPU Memory - RSS: {usage['rss_gb']:.2f}GB, "
                  f"Percent: {usage['percent']:.1f}%")
    
    def reset_peak_memory(self):
        """Reset peak memory statistics."""
        if self.is_cuda:
            torch.cuda.reset_peak_memory_stats(self.device)
    
    def clear_cache(self):
        """Clear memory cache."""
        if self.is_cuda:
            torch.cuda.empty_cache()
        gc.collect()
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """
        Get summary of memory usage throughout the session.
        
        Returns:
            Summary statistics of memory usage
        """
        if not self.memory_history:
            return {}
        
        if self.is_cuda:
            allocated_values = [entry['allocated_gb'] for entry in self.memory_history]
            reserved_values = [entry['reserved_gb'] for entry in self.memory_history]
            
            return {
                'peak_allocated_gb': max(allocated_values),
                'avg_allocated_gb': sum(allocated_values) / len(allocated_values),
                'peak_reserved_gb': max(reserved_values),
                'avg_reserved_gb': sum(reserved_values) / len(reserved_values),
                'num_measurements': len(self.memory_history)
            }
        else:
            rss_values = [entry['rss_gb'] for entry in self.memory_history]
            percent_values = [entry['percent'] for entry in self.memory_history]
            
            return {
                'peak_rss_gb': max(rss_values),
                'avg_rss_gb': sum(rss_values) / len(rss_values),
                'peak_percent': max(percent_values),
                'avg_percent': sum(percent_values) / len(percent_values),
                'num_measurements': len(self.memory_history)
            }


class DynamicBatchProcessor:
    """
    Dynamic batch processor that adjusts batch sizes based on available memory.
    """
    
    def __init__(
        self,
        initial_batch_size: int = 32,
        min_batch_size: int = 1,
        max_batch_size: int = 128,
        memory_threshold: float = 0.8,
        device: torch.device = None
    ):
        """
        Initialize dynamic batch processor.
        
        Args:
            initial_batch_size: Starting batch size
            min_batch_size: Minimum allowed batch size
            max_batch_size: Maximum allowed batch size
            memory_threshold: Memory usage threshold for batch size adjustment
            device: Device for memory monitoring
        """
        self.current_batch_size = initial_batch_size
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.memory_threshold = memory_threshold
        
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.memory_monitor = MemoryMonitor(device)
        self.adjustment_history = []
        
    def adjust_batch_size(self) -> int:
        """
        Adjust batch size based on current memory usage.
        
        Returns:
            New batch size
        """
        memory_usage = self.memory_monitor.get_memory_usage()
        
        if self.memory_monitor.is_cuda:
            # Use allocated memory percentage
            total_memory = torch.cuda.get_device_properties(self.memory_monitor.device).total_memory
            memory_percent = memory_usage['allocated_gb'] * 1024**3 / total_memory
        else:
            # Use system memory percentage
            memory_percent = memory_usage['percent'] / 100.0
        
        old_batch_size = self.current_batch_size
        
        if memory_percent > self.memory_threshold:
            # Reduce batch size
            self.current_batch_size = max(
                self.min_batch_size,
                int(self.current_batch_size * 0.8)
            )
        elif memory_percent < self.memory_threshold * 0.6:
            # Increase batch size
            self.current_batch_size = min(
                self.max_batch_size,
                int(self.current_batch_size * 1.2)
            )
        
        if self.current_batch_size != old_batch_size:
            adjustment = {
                'old_size': old_batch_size,
                'new_size': self.current_batch_size,
                'memory_percent': memory_percent,
                'reason': 'reduce' if memory_percent > self.memory_threshold else 'increase'
            }
            self.adjustment_history.append(adjustment)
            
            print(f"Batch size adjusted: {old_batch_size} → {self.current_batch_size} "
                  f"(Memory: {memory_percent:.1%})")
        
        return self.current_batch_size
    
    def process_batches(
        self,
        data: torch.Tensor,
        process_fn: callable,
        **kwargs
    ) -> List[torch.Tensor]:
        """
        Process data in dynamically sized batches.
        
        Args:
            data: Input data tensor [N, ...]
            process_fn: Function to process each batch
            **kwargs: Additional arguments for process_fn
            
        Returns:
            List of processed batch outputs
        """
        N = data.shape[0]
        results = []
        
        i = 0
        while i < N:
            # Adjust batch size based on memory
            batch_size = self.adjust_batch_size()
            
            # Get current batch
            end_idx = min(i + batch_size, N)
            batch = data[i:end_idx]
            
            try:
                # Process batch
                self.memory_monitor.log_memory(f"Before batch {i//batch_size + 1}")
                result = process_fn(batch, **kwargs)
                results.append(result)
                
                self.memory_monitor.log_memory(f"After batch {i//batch_size + 1}")
                
                i = end_idx
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    # Reduce batch size and retry
                    self.current_batch_size = max(
                        self.min_batch_size,
                        self.current_batch_size // 2
                    )
                    
                    print(f"OOM detected, reducing batch size to {self.current_batch_size}")
                    self.memory_monitor.clear_cache()
                    
                    if self.current_batch_size < self.min_batch_size:
                        raise RuntimeError("Cannot reduce batch size further")
                else:
                    raise e
        
        return results
    
    def get_adjustment_summary(self) -> Dict[str, Any]:
        """Get summary of batch size adjustments."""
        if not self.adjustment_history:
            return {'num_adjustments': 0}
        
        increases = sum(1 for adj in self.adjustment_history if adj['reason'] == 'increase')
        decreases = sum(1 for adj in self.adjustment_history if adj['reason'] == 'reduce')
        
        return {
            'num_adjustments': len(self.adjustment_history),
            'increases': increases,
            'decreases': decreases,
            'final_batch_size': self.current_batch_size,
            'adjustment_history': self.adjustment_history
        }


def optimize_fusion_memory(
    fusion_module: nn.Module,
    enable_checkpointing: bool = True,
    chunk_size: int = 512,
    mixed_precision: bool = True
) -> nn.Module:
    """
    Apply memory optimizations to a fusion module.
    
    Args:
        fusion_module: The fusion module to optimize
        enable_checkpointing: Whether to enable gradient checkpointing
        chunk_size: Chunk size for attention computation
        mixed_precision: Whether to enable mixed precision training
        
    Returns:
        Optimized fusion module
    """
    # Enable gradient checkpointing if requested
    if enable_checkpointing:
        fusion_module.gradient_checkpointing_enable()
    
    # Replace attention modules with memory-optimized versions
    for name, module in fusion_module.named_modules():
        if isinstance(module, nn.MultiheadAttention):
            # Replace with memory-optimized attention
            optimized_attn = MemoryOptimizedAttention(
                embed_dim=module.embed_dim,
                num_heads=module.num_heads,
                chunk_size=chunk_size,
                use_checkpointing=enable_checkpointing,
                dropout=module.dropout
            )
            
            # Copy weights if possible
            try:
                optimized_attn.load_state_dict(module.state_dict(), strict=False)
            except:
                warnings.warn(f"Could not transfer weights for {name}")
            
            # Replace module
            parent_module = fusion_module
            for attr in name.split('.')[:-1]:
                parent_module = getattr(parent_module, attr)
            setattr(parent_module, name.split('.')[-1], optimized_attn)
    
    # Enable mixed precision if requested
    if mixed_precision and torch.cuda.is_available():
        fusion_module = fusion_module.half()
    
    return fusion_module


def estimate_memory_requirements(
    batch_size: int,
    seq_len: int,
    tda_dim: int,
    freq_dim: int,
    num_heads: int = 4,
    hidden_dim: int = 64
) -> Dict[str, float]:
    """
    Estimate memory requirements for fusion operations.
    
    Args:
        batch_size: Batch size
        seq_len: Sequence length
        tda_dim: TDA feature dimension
        freq_dim: Frequency feature dimension
        num_heads: Number of attention heads
        hidden_dim: Hidden dimension
        
    Returns:
        Dictionary with memory estimates in GB
    """
    # Estimate tensor sizes (in elements)
    tda_features = batch_size * tda_dim
    freq_features = batch_size * seq_len * freq_dim
    
    # Attention matrices
    attention_scores = batch_size * num_heads * seq_len * seq_len
    attention_weights = attention_scores
    
    # Hidden representations
    hidden_features = batch_size * seq_len * hidden_dim
    
    # Gradients (roughly double the forward pass memory)
    gradient_multiplier = 2.0
    
    # Convert to bytes (assuming float32 = 4 bytes)
    bytes_per_element = 4
    
    total_elements = (
        tda_features + freq_features + 
        attention_scores + attention_weights + 
        hidden_features
    ) * gradient_multiplier
    
    total_gb = total_elements * bytes_per_element / (1024**3)
    
    return {
        'tda_features_gb': tda_features * bytes_per_element / (1024**3),
        'freq_features_gb': freq_features * bytes_per_element / (1024**3),
        'attention_gb': (attention_scores + attention_weights) * bytes_per_element / (1024**3),
        'hidden_gb': hidden_features * bytes_per_element / (1024**3),
        'total_forward_gb': total_gb / gradient_multiplier,
        'total_with_gradients_gb': total_gb,
        'recommended_batch_size': max(1, int(batch_size * 8.0 / total_gb))  # Target 8GB usage
    } 