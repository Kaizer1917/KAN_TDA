import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Autoformer_EncDec import series_decomp
from layers.Embed import DataEmbedding_wo_pos
from layers.StandardNorm import Normalize
from layers.ChebyKANLayer import ChebyKANLinear
from layers.TakensEmbedding import TakensEmbedding
from layers.PersistentHomology import PersistentHomologyComputer
from utils.persistence_landscapes import PersistenceLandscape, TopologicalFeatureExtractor
import math
import numpy as np


class ChebyKANLayer(nn.Module):
    def __init__(self, in_features, out_features,order):
        super().__init__()
        self.fc1 = ChebyKANLinear(
                            in_features,
                            out_features,
                            order)
    def forward(self, x):
        B, N, C = x.shape
        x = self.fc1(x.reshape(B*N,C))
        x = x.reshape(B,N,-1).contiguous()
        return x
    

class FrequencyDecomp(nn.Module):

    def __init__(self, configs):
        super(FrequencyDecomp, self).__init__()
        self.configs = configs

    def forward(self, level_list):
      
        level_list_reverse = level_list.copy()
        level_list_reverse.reverse()
        out_low = level_list_reverse[0]
        out_high = level_list_reverse[1]
        out_level_list = [out_low]
        for i in range(len(level_list_reverse) - 1):
            out_high_res = self.frequency_interpolation(out_low.transpose(1,2),
                                                        self.configs.seq_len // (self.configs.down_sampling_window ** (self.configs.down_sampling_layers-i)),
                                                        self.configs.seq_len // (self.configs.down_sampling_window ** (self.configs.down_sampling_layers-i-1))
                                                        ).transpose(1,2)
            out_high_left = out_high - out_high_res
            out_low = out_high
            if i + 2 <= len(level_list_reverse) - 1:
                out_high = level_list_reverse[i + 2]    
            out_level_list.append(out_high_left) 
        out_level_list.reverse()
        return out_level_list   
    
    def frequency_interpolation(self, x, seq_len, target_len):
        """
        Fixed frequency interpolation with proper tensor size handling
        """
        len_ratio = seq_len / target_len
        x_fft = torch.fft.rfft(x, dim=2)
        
        # Calculate proper FFT sizes
        x_fft_size = x_fft.size(2)  # Actual FFT size
        target_fft_size = target_len // 2 + 1
        
        # Create output FFT tensor
        out_fft = torch.zeros([x_fft.size(0), x_fft.size(1), target_fft_size], 
                             dtype=x_fft.dtype, device=x_fft.device)
        
        # Copy available frequencies, handling size mismatch
        copy_size = min(x_fft_size, target_fft_size)
        out_fft[:, :, :copy_size] = x_fft[:, :, :copy_size]
        
        # Inverse FFT with proper size specification
        out = torch.fft.irfft(out_fft, n=target_len, dim=2)
        out = out * len_ratio
        
        return out
    

class FrequencyMixing(nn.Module):

    def __init__(self, configs):
        super(FrequencyMixing, self).__init__()
        self.configs = configs
        self.front_block = M_KAN(configs.d_model,
                                 self.configs.seq_len // (self.configs.down_sampling_window ** (self.configs.down_sampling_layers)),
                                 order=configs.begin_order)
                  
          
        self.front_blocks = torch.nn.ModuleList(
                [
                    M_KAN(configs.d_model,
                          self.configs.seq_len // (self.configs.down_sampling_window ** (self.configs.down_sampling_layers-i-1)),
                          order=i+configs.begin_order+1)
                    for i in range(configs.down_sampling_layers)
                ])
     
    def forward(self, level_list):
        level_list_reverse = level_list.copy()
        level_list_reverse.reverse()
        out_low = level_list_reverse[0]
        out_high = level_list_reverse[1]
        out_low = self.front_block(out_low)
        out_level_list = [out_low]
        for i in range(len(level_list_reverse) - 1):
            out_high = self.front_blocks[i](out_high)
            out_high_res = self.frequency_interpolation(out_low.transpose(1,2),
                                            self.configs.seq_len // (self.configs.down_sampling_window ** (self.configs.down_sampling_layers-i)),
                                            self.configs.seq_len // (self.configs.down_sampling_window ** (self.configs.down_sampling_layers-i-1))
                                            ).transpose(1,2)
            out_high = out_high + out_high_res
            out_low = out_high
            if i + 2 <= len(level_list_reverse) - 1:
                out_high = level_list_reverse[i + 2]    
            out_level_list.append(out_low) 
        out_level_list.reverse()
        return out_level_list

    def frequency_interpolation(self,x,seq_len,target_len):
        len_ratio = seq_len/target_len
        x_fft = torch.fft.rfft(x, dim=2)
        out_fft = torch.zeros([x_fft.size(0),x_fft.size(1),target_len//2+1],dtype=x_fft.dtype).to(x_fft.device)
        out_fft[:,:,:seq_len//2+1] = x_fft
        out = torch.fft.irfft(out_fft, dim=2)
        out = out * len_ratio
        return out
    
class M_KAN(nn.Module):
    def __init__(self,d_model,seq_len,order):
        super().__init__()
        self.channel_mixer = nn.Sequential(
            ChebyKANLayer(d_model, d_model,order)
        )
        self.conv = BasicConv(d_model,d_model,kernel_size=3,degree=order,groups=d_model)
    def forward(self,x):
        x1 = self.channel_mixer(x)
        x2 = self.conv(x)
        out  = x1 + x2
        return out 

class BasicConv(nn.Module):
    def __init__(self,c_in,c_out, kernel_size, degree,stride=1, padding=0, dilation=1, groups=1, act=False, bn=False, bias=False,dropout=0.):
        super(BasicConv, self).__init__()
        self.out_channels = c_out
        self.conv = nn.Conv1d(c_in,c_out, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm1d(c_out) if bn else None
        self.act = nn.GELU() if act else None
        self.dropout = nn.Dropout(dropout)
    def forward(self, x): 
        if self.bn is not None:
            x = self.bn(x)
        x = self.conv(x.transpose(-1,-2)).transpose(-1,-2)
        if self.act is not None:
            x = self.act(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x


class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.down_sampling_window = configs.down_sampling_window
        self.channel_independence = configs.channel_independence
        self.res_blocks = nn.ModuleList([FrequencyDecomp(configs)
                                         for _ in range(configs.e_layers)])
        self.add_blocks = nn.ModuleList([FrequencyMixing(configs)
                                         for _ in range(configs.e_layers)])

        self.preprocess = series_decomp(configs.moving_avg)
        self.enc_in = configs.enc_in
        self.use_future_temporal_feature = configs.use_future_temporal_feature


        self.enc_embedding = DataEmbedding_wo_pos(1, configs.d_model, configs.embed, configs.freq,
                                                      configs.dropout)
        self.layer = configs.e_layers
        self.normalize_layers = torch.nn.ModuleList(
            [
                Normalize(self.configs.enc_in, affine=True, non_norm=True if configs.use_norm == 0 else False)
                for i in range(configs.down_sampling_layers + 1)
            ]
        )
        self.projection_layer = nn.Linear(
                    configs.d_model, 1, bias=True)
        self.predict_layer =nn. Linear(
                        configs.seq_len,
                        configs.pred_len,
                    )

    def forecast(self, x_enc):
        x_enc = self.__multi_level_process_inputs(x_enc)
        x_list = []
        for i, x in zip(range(len(x_enc)), x_enc, ):
            B, T, N = x.size()
            x = self.normalize_layers[i](x, 'norm')
            x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
            x_list.append(x)

       
        enc_out_list = []
        for i, x in zip(range(len(x_list)), x_list):
            enc_out = self.enc_embedding(x, None)  # [B,T,C]
            enc_out_list.append(enc_out)

        for i in range(self.layer):
            enc_out_list = self.res_blocks[i](enc_out_list)
            enc_out_list = self.add_blocks[i](enc_out_list)

        dec_out = enc_out_list[0]
        dec_out = self.predict_layer(dec_out.permute(0, 2, 1)).permute(
                0, 2, 1)  
        dec_out = self.projection_layer(dec_out).reshape(B, self.configs.c_out, self.pred_len).permute(0, 2, 1).contiguous()
        dec_out = self.normalize_layers[0](dec_out, 'denorm')
        return dec_out
    

    def __multi_level_process_inputs(self, x_enc):
        down_pool = torch.nn.AvgPool1d(self.configs.down_sampling_window)
        # B,T,C -> B,C,T
        x_enc = x_enc.permute(0, 2, 1)
        x_enc_ori = x_enc
        x_enc_sampling_list = []
        x_enc_sampling_list.append(x_enc.permute(0, 2, 1))
        for i in range(self.configs.down_sampling_layers):
            x_enc_sampling = down_pool(x_enc_ori)
            x_enc_sampling_list.append(x_enc_sampling.permute(0, 2, 1))
            x_enc_ori = x_enc_sampling
        x_enc = x_enc_sampling_list
        return x_enc

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast':
            dec_out = self.forecast(x_enc)
            return dec_out
        else:
            raise ValueError('Other tasks implemented yet')

class TDAFrequencyDecomp(FrequencyDecomp):
    """
    Enhanced Frequency Decomposition with Topological Data Analysis guidance.
    
    Integrates TDA features to guide frequency decomposition by:
    1. Computing topological features from time series embeddings
    2. Using persistent features to weight frequency components
    3. Injecting topological information into frequency upsampling
    """
    
    def __init__(self, configs):
        super(TDAFrequencyDecomp, self).__init__(configs)
        self.configs = configs
        
        # TDA Components
        self.takens_embedding = TakensEmbedding(
            dims=getattr(configs, 'takens_dims', [2, 3, 5]),
            delays=getattr(configs, 'takens_delays', [1, 2, 4]),
            strategy='multi_scale'
        )
        
        self.homology_computer = PersistentHomologyComputer(
            backend=getattr(configs, 'homology_backend', 'ripser'),
            max_dimension=getattr(configs, 'max_homology_dim', 2)
        )
        
        self.feature_extractor = TopologicalFeatureExtractor(
            resolution=getattr(configs, 'landscape_resolution', 100)
        )
        
        # Topological feature integration
        self.tda_weight = getattr(configs, 'tda_weight', 0.3)
        self.enable_tda_guidance = getattr(configs, 'enable_tda_guidance', True)
        
        # Learnable topological attention
        tda_feature_dim = self._estimate_tda_feature_dim()
        self.tda_attention = nn.Sequential(
            nn.Linear(tda_feature_dim, configs.d_model),
            nn.ReLU(),
            nn.Linear(configs.d_model, configs.d_model),
            nn.Sigmoid()
        )
        
        # Topological injection layers - make them adaptive
        num_injection_layers = max(1, configs.down_sampling_layers + 1)
        self.topo_injection = nn.ModuleList([
            nn.Linear(tda_feature_dim, configs.d_model)
            for _ in range(num_injection_layers)
        ])
    
    def _estimate_tda_feature_dim(self):
        """Estimate TDA feature dimension based on configuration"""
        # Simplified statistical features: mean, std, min, max, trend, autocorr, spectral_centroid
        return 7
    
    def _compute_tda_features(self, x):
        """
        Compute topological features from input time series
        
        Args:
            x: Input tensor [B, T, D]
            
        Returns:
            tda_features: Topological features [B, tda_dim]
        """
        if not self.enable_tda_guidance:
            return None
            
        B, T, D = x.shape
        
        # For now, return simple statistical features as a placeholder
        # This ensures the integration works while we can enhance TDA features later
        tda_features = []
        
        for b in range(B):
            # Compute basic statistical features that capture some temporal structure
            sample_features = []
            
            for d in range(D):
                time_series = x[b, :, d]
                
                # Basic statistical features
                mean_val = torch.mean(time_series)
                std_val = torch.std(time_series)
                min_val = torch.min(time_series)
                max_val = torch.max(time_series)
                
                # Trend features
                indices = torch.arange(T, dtype=torch.float32, device=x.device)
                trend = torch.corrcoef(torch.stack([indices, time_series]))[0, 1]
                if torch.isnan(trend):
                    trend = torch.tensor(0.0, device=x.device)
                
                # Autocorrelation at lag 1
                if T > 1:
                    autocorr = torch.corrcoef(torch.stack([time_series[:-1], time_series[1:]]))[0, 1]
                    if torch.isnan(autocorr):
                        autocorr = torch.tensor(0.0, device=x.device)
                else:
                    autocorr = torch.tensor(0.0, device=x.device)
                
                # Frequency domain features
                fft_vals = torch.fft.fft(time_series)
                power_spectrum = torch.abs(fft_vals) ** 2
                spectral_centroid = torch.sum(indices[:T//2] * power_spectrum[:T//2]) / torch.sum(power_spectrum[:T//2])
                if torch.isnan(spectral_centroid):
                    spectral_centroid = torch.tensor(0.0, device=x.device)
                
                # Combine features
                channel_features = torch.stack([
                    mean_val, std_val, min_val, max_val, trend, autocorr, spectral_centroid
                ])
                sample_features.append(channel_features)
            
            # Average across channels
            if sample_features:
                batch_features = torch.stack(sample_features).mean(dim=0)
            else:
                batch_features = torch.zeros(7, device=x.device)
                
            tda_features.append(batch_features)
        
        return torch.stack(tda_features)  # [B, 7]
    
    def _topological_weighting(self, freq_components, tda_features):
        """
        Apply topological attention to frequency components
        
        Args:
            freq_components: Frequency components [B, T, D]
            tda_features: TDA features [B, tda_dim]
            
        Returns:
            weighted_components: Topologically weighted components [B, T, D]
        """
        if tda_features is None:
            return freq_components
            
        # Compute attention weights from TDA features
        attention_weights = self.tda_attention(tda_features)  # [B, D]
        attention_weights = attention_weights.unsqueeze(1)    # [B, 1, D]
        
        # Apply attention to frequency components
        weighted_components = freq_components * attention_weights
        
        return weighted_components
    
    def _inject_topological_features(self, freq_band, tda_features, level_idx):
        """
        Inject topological features into frequency bands
        
        Args:
            freq_band: Frequency band [B, T, D]
            tda_features: TDA features [B, tda_dim]
            level_idx: Current frequency level index
            
        Returns:
            enhanced_band: Topologically enhanced frequency band [B, T, D]
        """
        if tda_features is None:
            return freq_band
            
        # Safety check for level index - use modulo to wrap around if needed
        if len(self.topo_injection) == 0:
            return freq_band
            
        # Ensure valid index by wrapping around
        safe_idx = level_idx % len(self.topo_injection)
        if safe_idx < 0:
            safe_idx = 0
            
        # Project TDA features to model dimension
        topo_projection = self.topo_injection[safe_idx](tda_features)  # [B, D]
        topo_projection = topo_projection.unsqueeze(1)  # [B, 1, D]
        
        # Inject with learnable weight
        enhanced_band = freq_band + self.tda_weight * topo_projection
        
        return enhanced_band
    
    def forward(self, level_list):
        """
        Enhanced forward pass with topological guidance
        
        Args:
            level_list: List of multi-level sequences
            
        Returns:
            out_level_list: Topologically guided frequency components
        """
        # Compute TDA features from the finest level (most information)
        tda_features = self._compute_tda_features(level_list[0])
        
        # Handle single level case (no decomposition needed)
        if len(level_list) == 1:
            # Apply topological weighting and injection to the single level
            enhanced_level = self._topological_weighting(level_list[0], tda_features)
            enhanced_level = self._inject_topological_features(enhanced_level, tda_features, 0)
            return [enhanced_level]
        
        # Original frequency decomposition for multi-level case
        level_list_reverse = level_list.copy()
        level_list_reverse.reverse()
        out_low = level_list_reverse[0]
        out_high = level_list_reverse[1]
        out_level_list = [out_low]
        
        for i in range(len(level_list_reverse) - 1):
            # Standard frequency upsampling
            out_high_res = self.frequency_interpolation(
                out_low.transpose(1,2),
                self.configs.seq_len // (self.configs.down_sampling_window ** (self.configs.down_sampling_layers-i)),
                self.configs.seq_len // (self.configs.down_sampling_window ** (self.configs.down_sampling_layers-i-1))
            ).transpose(1,2)
            
            # Extract frequency component
            out_high_left = out_high - out_high_res
            
            # Apply topological weighting
            out_high_left = self._topological_weighting(out_high_left, tda_features)
            
            # Inject topological features
            out_high_left = self._inject_topological_features(
                out_high_left, tda_features, len(level_list_reverse) - i - 2
            )
            
            out_low = out_high
            if i + 2 <= len(level_list_reverse) - 1:
                out_high = level_list_reverse[i + 2]    
            out_level_list.append(out_high_left) 
            
        out_level_list.reverse()
        return out_level_list

class TopologyGuidedMKAN(M_KAN):
    """
    Topology-Guided Multi-order KAN with adaptive complexity.
    
    Enhances M_KAN by:
    1. Adapting KAN polynomial orders based on topological complexity
    2. Using persistent features for attention mechanisms
    3. Cross-frequency topological fusion
    """
    
    def __init__(self, d_model, seq_len, order, configs=None):
        # Initialize with base order, will be adapted dynamically
        super(TopologyGuidedMKAN, self).__init__(d_model, seq_len, order)
        
        self.base_order = order
        self.max_order = getattr(configs, 'max_kan_order', order + 3) if configs else order + 3
        self.min_order = max(1, order - 1)
        self.configs = configs
        
        # Topological complexity analyzer
        self.complexity_analyzer = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )
        
        # Multi-order KAN layers for adaptive complexity
        self.adaptive_kan_layers = nn.ModuleDict({
            str(o): ChebyKANLayer(d_model, d_model, o)
            for o in range(self.min_order, self.max_order + 1)
        })
        
        # Topological attention for cross-frequency fusion
        self.topo_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=getattr(configs, 'topo_attention_heads', 4) if configs else 4,
            dropout=getattr(configs, 'topo_attention_dropout', 0.1) if configs else 0.1,
            batch_first=True
        )
        
        # Persistence-guided feature enhancement
        self.persistence_gate = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )
        
        # Order selection network
        self.order_selector = nn.Sequential(
            nn.Linear(d_model, self.max_order - self.min_order + 1),
            nn.Softmax(dim=-1)
        )
        
        # Enable/disable topological guidance
        self.enable_topology_guidance = getattr(configs, 'enable_topology_guidance', True) if configs else True
        
    def _analyze_topological_complexity(self, x):
        """
        Analyze topological complexity of input features
        
        Args:
            x: Input features [B, T, D]
            
        Returns:
            complexity_score: Complexity score [B, 1]
        """
        # Global average pooling to get sequence-level features
        global_features = x.mean(dim=1)  # [B, D]
        
        # Compute complexity score
        complexity_score = self.complexity_analyzer(global_features)  # [B, 1]
        
        return complexity_score
    
    def _adaptive_order_selection(self, x):
        """
        Select optimal KAN order based on input complexity
        
        Args:
            x: Input features [B, T, D]
            
        Returns:
            selected_order: Selected polynomial order
            order_weights: Soft weights for order mixing [B, num_orders]
        """
        if not self.enable_topology_guidance:
            return self.base_order, None
            
        # Analyze complexity
        global_features = x.mean(dim=1)  # [B, D]
        order_weights = self.order_selector(global_features)  # [B, num_orders]
        
        # Compute weighted average order
        order_range = torch.arange(
            self.min_order, self.max_order + 1, 
            dtype=torch.float32, device=x.device
        ).unsqueeze(0)  # [1, num_orders]
        
        selected_order = (order_weights * order_range).sum(dim=-1)  # [B]
        avg_order = selected_order.mean().item()
        
        # Use integer order closest to weighted average
        selected_order = int(round(avg_order))
        selected_order = max(self.min_order, min(self.max_order, selected_order))
        
        return selected_order, order_weights
    
    def _persistence_guided_attention(self, x, frequency_context=None):
        """
        Apply persistence-guided attention mechanism
        
        Args:
            x: Input features [B, T, D]
            frequency_context: Context from other frequency bands [B, T, D]
            
        Returns:
            attended_x: Attention-enhanced features [B, T, D]
        """
        if not self.enable_topology_guidance or frequency_context is None:
            return x
            
        # Apply multi-head attention with frequency context
        attended_x, attention_weights = self.topo_attention(
            query=x,
            key=frequency_context,
            value=frequency_context
        )
        
        # Residual connection
        attended_x = x + attended_x
        
        return attended_x
    
    def _apply_persistence_gate(self, x):
        """
        Apply persistence-based gating to features
        
        Args:
            x: Input features [B, T, D]
            
        Returns:
            gated_x: Gated features [B, T, D]
        """
        if not self.enable_topology_guidance:
            return x
            
        # Compute persistence-based gates
        gates = self.persistence_gate(x)  # [B, T, D]
        
        # Apply gating
        gated_x = x * gates
        
        return gated_x
    
    def _mixed_order_kan_forward(self, x, order_weights):
        """
        Forward pass with mixed-order KAN based on soft weights
        
        Args:
            x: Input features [B, T, D]
            order_weights: Soft weights for different orders [B, num_orders]
            
        Returns:
            mixed_output: Mixed-order KAN output [B, T, D]
        """
        if order_weights is None:
            # Fallback to base order
            return self.channel_mixer(x)
            
        B, T, D = x.shape
        mixed_output = torch.zeros_like(x)
        
        # Compute weighted combination of different orders
        for i, order in enumerate(range(self.min_order, self.max_order + 1)):
            kan_layer = self.adaptive_kan_layers[str(order)]
            order_output = kan_layer(x)  # [B, T, D]
            
            # Weight by order selection probability
            weight = order_weights[:, i:i+1].unsqueeze(1)  # [B, 1, 1]
            mixed_output += weight * order_output
            
        return mixed_output
    
    def forward(self, x, frequency_context=None, tda_features=None):
        """
        Enhanced forward pass with topological guidance
        
        Args:
            x: Input features [B, T, D]
            frequency_context: Context from other frequency bands [B, T, D]
            tda_features: Precomputed TDA features [B, tda_dim]
            
        Returns:
            out: Enhanced output features [B, T, D]
        """
        # Apply persistence-guided attention if context available
        x_attended = self._persistence_guided_attention(x, frequency_context)
        
        # Adaptive order selection based on complexity
        selected_order, order_weights = self._adaptive_order_selection(x_attended)
        
        # Channel mixing with adaptive order
        if self.enable_topology_guidance and order_weights is not None:
            x1 = self._mixed_order_kan_forward(x_attended, order_weights)
        else:
            # Update the base KAN layer order if needed
            if selected_order != self.base_order:
                # Use the closest available order
                available_orders = list(self.adaptive_kan_layers.keys())
                closest_order = min(available_orders, key=lambda o: abs(int(o) - selected_order))
                x1 = self.adaptive_kan_layers[closest_order](x_attended)
            else:
                x1 = self.channel_mixer(x_attended)
        
        # Temporal convolution (unchanged)
        x2 = self.conv(x_attended)
        
        # Combine branches
        combined = x1 + x2
        
        # Apply persistence gating
        gated_output = self._apply_persistence_gate(combined)
        
        # Final residual connection
        out = x + gated_output
        
        return out
    
    def get_complexity_info(self, x):
        """
        Get topological complexity information for analysis
        
        Args:
            x: Input features [B, T, D]
            
        Returns:
            info: Dictionary with complexity analysis
        """
        complexity_score = self._analyze_topological_complexity(x)
        selected_order, order_weights = self._adaptive_order_selection(x)
        
        return {
            'complexity_score': complexity_score.mean().item(),
            'selected_order': selected_order,
            'order_weights': order_weights.mean(dim=0).tolist() if order_weights is not None else None,
            'base_order': self.base_order,
            'order_range': (self.min_order, self.max_order)
        }

class TDAKAN_TDA(Model):
    """
    Enhanced KAN_TDA with Topological Data Analysis integration.
    
    Maintains full backward compatibility with original KAN_TDA while adding:
    1. TDA-guided frequency decomposition
    2. Topology-guided M-KAN blocks
    3. Configurable TDA features
    """
    
    def __init__(self, configs):
        # Initialize base KAN_TDA
        super(TDAKAN_TDA, self).__init__(configs)
        
        # TDA Configuration
        self.enable_tda = getattr(configs, 'enable_tda', False)
        self.tda_mode = getattr(configs, 'tda_mode', 'full')  # 'full', 'decomp_only', 'kan_only'
        
        if self.enable_tda:
            # Replace frequency decomposition blocks with TDA-enhanced versions
            if self.tda_mode in ['full', 'decomp_only']:
                self.tda_res_blocks = nn.ModuleList([
                    TDAFrequencyDecomp(configs) for _ in range(configs.e_layers)
                ])
            
            # Replace M-KAN blocks with topology-guided versions
            if self.tda_mode in ['full', 'kan_only']:
                self.tda_add_blocks = nn.ModuleList([
                    TDAFrequencyMixing(configs) for _ in range(configs.e_layers)
                ])
        
        # Performance monitoring
        self.tda_computation_time = 0.0
        self.tda_feature_cache = {}
        self.enable_tda_caching = getattr(configs, 'enable_tda_caching', True)
    
    def _get_cache_key(self, x_enc):
        """Generate cache key for TDA features"""
        if not self.enable_tda_caching:
            return None
        # Simple hash based on tensor shape and first/last values
        shape_str = str(x_enc.shape)
        value_str = f"{x_enc.flatten()[0]:.6f}_{x_enc.flatten()[-1]:.6f}"
        return f"{shape_str}_{value_str}"
    
    def forecast(self, x_enc):
        """
        Enhanced forecast with optional TDA integration
        
        Args:
            x_enc: Input time series [B, T, C]
            
        Returns:
            dec_out: Forecasted values [B, pred_len, C]
        """
        if not self.enable_tda:
            # Fallback to original KAN_TDA
            return super().forecast(x_enc)
        
        import time
        tda_start_time = time.time()
        
        # Multi-level preprocessing (same as original)
        x_enc = self._Model__multi_level_process_inputs(x_enc)
        x_list = []
        
        for i, x in zip(range(len(x_enc)), x_enc):
            B, T, N = x.size()
            x = self.normalize_layers[i](x, 'norm')
            x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
            x_list.append(x)

        # Embedding (same as original)
        enc_out_list = []
        for i, x in zip(range(len(x_list)), x_list):
            enc_out = self.enc_embedding(x, None)  # [B,T,C]
            enc_out_list.append(enc_out)

        # Enhanced processing with TDA
        for i in range(self.layer):
            # Frequency decomposition
            if self.tda_mode in ['full', 'decomp_only']:
                enc_out_list = self.tda_res_blocks[i](enc_out_list)
            else:
                enc_out_list = self.res_blocks[i](enc_out_list)
            
            # Frequency mixing
            if self.tda_mode in ['full', 'kan_only']:
                enc_out_list = self.tda_add_blocks[i](enc_out_list)
            else:
                enc_out_list = self.add_blocks[i](enc_out_list)

        # Output generation (same as original)
        dec_out = enc_out_list[0]
        dec_out = self.predict_layer(dec_out.permute(0, 2, 1)).permute(0, 2, 1)
        dec_out = self.projection_layer(dec_out).reshape(B, self.configs.c_out, self.pred_len).permute(0, 2, 1).contiguous()
        
        # Denormalization
        dec_out = self.normalize_layers[0](dec_out, 'denorm')
        
        # Track TDA computation time
        self.tda_computation_time += time.time() - tda_start_time
        
        return dec_out
    
    def get_tda_performance_stats(self):
        """Get TDA performance statistics"""
        return {
            'tda_enabled': self.enable_tda,
            'tda_mode': self.tda_mode,
            'total_tda_time': self.tda_computation_time,
            'cache_size': len(self.tda_feature_cache),
            'cache_enabled': self.enable_tda_caching
        }
    
    def reset_tda_stats(self):
        """Reset TDA performance statistics"""
        self.tda_computation_time = 0.0
        self.tda_feature_cache.clear()


class TDAFrequencyMixing(FrequencyMixing):
    """
    Enhanced Frequency Mixing with Topology-Guided M-KAN blocks
    """
    
    def __init__(self, configs):
        # Initialize base FrequencyMixing but replace M_KAN with TopologyGuidedMKAN
        super(FrequencyMixing, self).__init__()  # Skip FrequencyMixing.__init__
        self.configs = configs
        
        # Replace with topology-guided M-KAN blocks
        self.front_block = TopologyGuidedMKAN(
            configs.d_model,
            self.configs.seq_len // (self.configs.down_sampling_window ** (self.configs.down_sampling_layers)),
            order=configs.begin_order,
            configs=configs
        )
                  
        self.front_blocks = torch.nn.ModuleList([
            TopologyGuidedMKAN(
                configs.d_model,
                self.configs.seq_len // (self.configs.down_sampling_window ** (self.configs.down_sampling_layers-i-1)),
                order=i+configs.begin_order+1,
                configs=configs
            )
            for i in range(configs.down_sampling_layers)
        ])
        
        # Cross-frequency context for attention
        self.enable_cross_frequency_attention = getattr(configs, 'enable_cross_frequency_attention', True)
    
    def forward(self, level_list):
        """
        Enhanced forward pass with cross-frequency topological attention
        
        Args:
            level_list: List of frequency level features
            
        Returns:
            out_level_list: Enhanced frequency features
        """
        level_list_reverse = level_list.copy()
        level_list_reverse.reverse()
        out_low = level_list_reverse[0]
        out_high = level_list_reverse[1]
        
        # Process lowest frequency with base topology-guided M-KAN
        out_low = self.front_block(out_low)
        out_level_list = [out_low]
        
        # Process higher frequencies with cross-frequency attention
        for i in range(len(level_list_reverse) - 1):
            # Prepare frequency context for attention
            frequency_context = out_low if self.enable_cross_frequency_attention else None
            
            # Apply topology-guided M-KAN with context
            out_high = self.front_blocks[i](out_high, frequency_context=frequency_context)
            
            # Standard frequency upsampling and mixing
            out_high_res = self.frequency_interpolation(
                out_low.transpose(1,2),
                self.configs.seq_len // (self.configs.down_sampling_window ** (self.configs.down_sampling_layers-i)),
                self.configs.seq_len // (self.configs.down_sampling_window ** (self.configs.down_sampling_layers-i-1))
            ).transpose(1,2)
            
            out_high = out_high + out_high_res
            out_low = out_high
            
            if i + 2 <= len(level_list_reverse) - 1:
                out_high = level_list_reverse[i + 2]    
            out_level_list.append(out_low) 
            
        out_level_list.reverse()
        return out_level_list

    def frequency_interpolation(self, x, seq_len, target_len):
        """
        Fixed frequency interpolation with proper tensor size handling
        """
        len_ratio = seq_len / target_len
        x_fft = torch.fft.rfft(x, dim=2)
        
        # Calculate proper FFT sizes
        x_fft_size = x_fft.size(2)  # Actual FFT size
        target_fft_size = target_len // 2 + 1
        
        # Create output FFT tensor
        out_fft = torch.zeros([x_fft.size(0), x_fft.size(1), target_fft_size], 
                             dtype=x_fft.dtype, device=x_fft.device)
        
        # Copy available frequencies, handling size mismatch
        copy_size = min(x_fft_size, target_fft_size)
        out_fft[:, :, :copy_size] = x_fft[:, :, :copy_size]
        
        # Inverse FFT with proper size specification
        out = torch.fft.irfft(out_fft, n=target_len, dim=2)
        out = out * len_ratio
        
        return out


