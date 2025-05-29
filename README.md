# KAN_TDA: KAN-based Frequency Decomposition Learning Architecture

<div align="center">
  <h2><b> (ICLR 2025) KAN_TDA: KAN-based Frequency Decomposition Learning Architecture for Long-term Time Series Forecasting 🚀 </b></h2>
</div>

**Official implementation of "KAN_TDA: KAN-based Frequency Decomposition Learning Architecture for Long-term Time Series Forecasting"**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)

## 📖 Overview

KAN_TDA introduces a novel **KAN-based Frequency Decomposition Learning architecture** that revolutionizes long-term time series forecasting by addressing the challenge of mixed frequency components in real-world time series data. The model achieves **state-of-the-art performance** while being **extremely lightweight** (12.84K-38.12K parameters vs competitors with 75K-25M parameters).

### 🎯 Key Innovations

1. **Frequency Decomposition Approach**: Decomposes mixed frequency components into multiple single frequency components for independent modeling
2. **Multi-order KAN Integration**: First application of Kolmogorov-Arnold Networks (KAN) to time series forecasting with adaptive complexity matching
3. **Decomposition-Learning-Mixing Framework**: Cascaded process that decomposes → learns → mixes frequency components

## 🏗️ Architecture Overview


The KAN_TDA architecture consists of three main components:

### 1. **Hierarchical Sequence Preprocessing**
```
x_i = AvgPool(Padding(x_{i-1}))   for i ∈ {1,...,k}
x_i = Linear(x_i) → x_i ∈ R^{T/d^{i-1} × D}
```
- Progressive downsampling using moving averages
- Multi-level sequences: `{x_1, x_2, ..., x_k}` (high → low frequency)
- Each sequence embedded into higher dimension `D`

### 2. **Cascaded Frequency Decomposition (CFD) Blocks**
```
x̂_i = IFFT(Padding(FFT(x_{i+1})))     # Frequency Upsampling
f_i = x_i - x̂_i                        # i-th frequency component extraction
```
**Key Innovation - Lossless Frequency Upsampling:**
- Uses FFT → Zero-Padding → IFFT to preserve frequency information
- Residual extraction isolates specific frequency bands
- No information degradation during upsampling

### 3. **Multi-order KAN Representation Learning (M-KAN) Blocks**

**Dual-Branch Architecture:**

**Branch 1 - Temporal Dependencies:**
```
f_{i,1} = Conv_{D→D}(f_i, group=D)  # Depthwise Convolution
```

**Branch 2 - KAN Representation Learning:**
```
T_n(x) = cos(n × arccos(x))                    # Chebyshev polynomial
φ_o(x) = Σ_{j=1}^D Σ_{i=0}^n Θ_{o,j,i} T_i(tanh(x_j))   # Learnable function
f_{i,2} = KAN(f_i, order=b+k-i)               # Multi-order adaptation
```

**Multi-order Strategy:**
- **Low frequencies → Low-order KANs**: Simple patterns, minimal complexity
- **High frequencies → High-order KANs**: Complex patterns, higher representation capacity
- **Adaptive complexity**: `Order(frequency_level_i) = base_order + (total_levels - i)`

**Final Output:**
```
f̂_i = f_{i,1} + f_{i,2}  # Residual connection
```

### 4. **Frequency Mixing Blocks**
```
x_i = IFFT(Padding(FFT(x_{i+1}))) + f_i    # Reconstruct multi-level sequences
X_O = Linear(x_1)                          # Final prediction
```

## 🧮 Mathematical Theory

### Kolmogorov-Arnold Network (KAN) Foundation
- **Kolmogorov-Arnold Representation Theorem**: Any multivariate continuous function can be expressed as combinations of univariate functions and additions
- **KAN vs MLP**: Replaces fixed node activations with learnable edge functions
- **Implementation**: `z_{l+1,j} = Σ_{i=1}^{n_l} φ_{l,j,i}(z_{l,i})`

### Frequency Decomposition Mathematics
**Core Principle:**
```
FFT: Time Domain → Frequency Domain
Padding: Extend frequency spectrum  
IFFT: Frequency Domain → Time Domain (preserved characteristics)
```

**Frequency Band Isolation:**
- Level `i` sequence: `x_i` (contains frequencies 1 to i)
- Level `i+1` sequence: `x_{i+1}` (contains frequencies 1 to i-1)
- Frequency upsampling: `x̂_i` (reconstructed without i-th frequency)
- Isolated i-th frequency: `f_i = x_i - x̂_i`

## 📊 Performance Results


### Key Achievements:
- **State-of-the-art performance** across multiple datasets (ETTh1, ETTh2, ETTm1, ETTm2, Weather)
- **Extremely lightweight**: 12.84K-38.12K parameters vs competitors with 75K-25M parameters
- **Computational efficiency**: 7.63M-29.86M MACs vs competitors with 20M-35G MACs
- **Consistent improvements** with longer look-back windows

### Ablation Study Findings:
1. **Frequency Upsampling is irreplaceable**: Alternatives degrade performance significantly
2. **Multi-order KANs outperform**: Both fixed-order KANs and MLPs
3. **Depthwise Convolution optimal**: Better than standard convolution or self-attention

## 🚀 Getting Started

### Prerequisites
```bash
pip install -r requirements.txt
```

### Dataset Preparation
Download all datasets from [Autoformer](https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy).

### Training
All training scripts are in the `./scripts` directory. 

**Example**: To train on Weather dataset with input-96-predict-96:
```bash
sh scripts/long_term_forecast/Weather/weather_96.sh
```

### Model Configuration
Key parameters in `run.py`:
```python
# Model Architecture
d_model: int = 16              # Embedding dimension
e_layers: int = 2              # Number of decomposition-mixing blocks
down_sampling_layers: int = 0  # Number of frequency levels
begin_order: int = 1           # Base KAN polynomial order

# Time Series
seq_len: int = 96              # Input sequence length
pred_len: int = 96             # Prediction length
moving_avg: int = 25           # Moving average window
```

## 📁 Project Structure

```
KAN_TDA/
├── models/
│   └── KAN_TDA.py              # Main model architecture
├── layers/
│   ├── ChebyKANLayer.py        # Chebyshev KAN implementation
│   ├── Embed.py                # Embedding layers
│   ├── StandardNorm.py         # Normalization utilities
│   ├── TakensEmbedding.py      # TDA: Takens embedding
│   └── PersistentHomology.py   # TDA: Persistent homology
├── utils/
│   └── persistence_landscapes.py  # TDA: Persistence landscapes
├── exp/
│   ├── exp_basic.py            # Base experiment class
│   └── exp_long_term_forecasting.py  # Training/evaluation logic
├── data_provider/              # Data loading and preprocessing
├── scripts/                    # Training scripts
└── run.py                      # Main entry point
```

## 🔬 TDA Integration (Advanced Features)

This implementation includes **Topological Data Analysis (TDA)** integration for enhanced pattern recognition:

- **Takens Embedding**: Transform time series into point clouds preserving topological properties
- **Persistent Homology**: Capture birth-death of topological features (components, loops, voids)
- **Persistence Landscapes**: Convert topological features to ML-ready statistical summaries

### TDA Components:
- `TakensEmbedding`: Multi-scale delay embedding with automatic parameter optimization
- `PersistentHomology`: Multi-backend homology computation (Ripser, GUDHI, Giotto)
- `PersistenceLandscape`: Statistical feature extraction from persistence diagrams

## 🔧 Extension Points

The architecture is designed for easy extension:

1. **New KAN Variants**: Replace Chebyshev with other basis functions (Fourier, Legendre)
2. **Alternative Decomposition**: Wavelet decomposition, empirical mode decomposition
3. **Attention Mechanisms**: Cross-frequency attention, temporal attention
4. **TDA Integration**: Topology-guided frequency decomposition and KAN adaptation

