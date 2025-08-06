# Installation Guide

## Quick Start

### Prerequisites
- Python 3.8 or higher
- PyTorch 1.9.0 or higher
- CUDA support (optional, for GPU acceleration)

### Install from GitHub

```bash
# Clone the repository
git clone https://github.com/yourusername/learnable-tensor-sketching.git
cd learnable-tensor-sketching

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

### Install Dependencies Only

If you just want to use the code without installing as a package:

```bash
pip install torch>=1.9.0 numpy>=1.21.0 scipy>=1.7.0 pandas>=1.3.0 scikit-learn>=1.0.0 matplotlib>=3.5.0 seaborn>=0.11.0
```

## Verification

Test your installation:

```python
import sys
sys.path.append('src')
from learnable.working_phase2_sketch import WorkingLearnableTensorSketch

# Create a simple test
sketch = WorkingLearnableTensorSketch()
print("✅ Installation successful!")
```

## GPU Support

For GPU acceleration (recommended for large datasets):

```bash
# Install PyTorch with CUDA support
pip install torch>=1.9.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
```

Check GPU availability:
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
```

## Development Installation

For development and testing:

```bash
# Install with development dependencies
pip install -r requirements.txt
pip install pytest jupyter

# Run tests
pytest src/evaluation/
```

## Common Issues

### ImportError with PyTorch
- Ensure PyTorch version >= 1.9.0
- Check CUDA compatibility if using GPU

### Memory Issues
- Reduce batch size for large sequences
- Use CPU instead of GPU for very large datasets

### Path Issues
- Ensure you're in the correct directory
- Check Python path includes src/ directory

## System Requirements

- **RAM**: Minimum 8GB, Recommended 16GB+
- **Storage**: 1GB free space
- **CPU**: Any modern multi-core processor
- **GPU**: Optional, NVIDIA with CUDA support for acceleration