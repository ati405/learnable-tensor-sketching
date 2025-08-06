# Learnable Tensor Sketching

A hierarchical learnable tensor sketching framework for genomic sequence similarity estimation.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%3E%3D1.9.0-red.svg)](https://pytorch.org/)

## Overview

This repository implements a learnable tensor sketching framework that extends existing approaches through a novel four-level parameter hierarchy specifically designed for genomic sequence similarity estimation. Building upon advances in learnable tensor methods (such as ISLET for tensor regression), our framework introduces architectural innovations tailored for similarity tasks.

## Key Features

- **325 learnable parameters** organized across four hierarchical levels
- **72.6% average improvement** across synthetic genomic datasets
- **66.9% discrimination improvement** on clinical HGMD records (211,291 sequences)
- **Computational efficiency**: >10,000 sequences/second processing speed
- **End-to-end gradient flow** for direct similarity estimation optimization

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/learnable-tensor-sketching.git
cd learnable-tensor-sketching

# Install dependencies
pip install -r requirements.txt

# Install package (optional)
pip install -e .
```

### Basic Usage

```python
import sys
sys.path.append('src')
from learnable.working_phase2_sketch import WorkingLearnableTensorSketch

# Initialize framework
sketch = WorkingLearnableTensorSketch(
    alphabet_size=4,      # DNA: A,T,G,C
    sketch_dim=64,        # Sketch dimensions
    subsequence_len=3,    # 3-mer analysis
    device='cpu'
)

# Generate sketch for a sequence
import torch
sequence = torch.randint(0, 4, (1, 200))  # Random DNA sequence
sketch_result = sketch.sketch_sequence(sequence)
```

### Run Examples

```bash
# Basic usage demonstration
python examples/basic_usage.py

# Reproduce synthetic results
python src/evaluation/benchmarking_framework.py

# Clinical genomic validation
python src/evaluation/enhanced_genomic_validation.py
```

## Architecture

The framework implements a hierarchical parameter structure:

```
θ = {α, β, γ, δ} (325 total parameters)
├── Global scaling: α ∈ ℝ (1 parameter)
├── Character-specific: β_c ∈ ℝ for A,T,G,C (4 parameters)  
├── Dimension weights: γ_j ∈ ℝ (64 parameters)
└── Interactions: δ_{c,j} ∈ ℝ (256 parameters)
```

Core sketch function:
```
sketch_θ[j] = α γ_j Σ_{i: h(m_i) = j} β_{c(m_i)} (1 + δ_{c(m_i),j}) · sign(m_i) · f(m_i)
```

## Performance Results

### Synthetic Datasets
| Dataset     | Baseline | Learnable | Improvement |
|-------------|----------|-----------|-------------|
| Periodic    | 0.000082 | 0.000138  | +68.9%      |
| Structured  | 0.001448 | 0.002424  | +67.4%      |
| Complex     | 0.000530 | 0.000942  | +77.8%      |
| Random      | 0.004288 | 0.007831  | +82.6%      |
| Scalability | 0.010252 | 0.017057  | +66.4%      |
| **Average** | **0.003320** | **0.005678** | **+72.6%** |

### Clinical Validation (HGMD Database, 211,291 records)
| Metric              | Baseline | Learnable | Improvement | P-value |
|---------------------|----------|-----------|-------------|---------|
| Discrimination Power| 0.285    | 0.475     | +66.9%      | 0.028   |
| Signal Strength     | 0.159    | 0.209     | +31.0%      | 0.045   |
| Composite Score     | 0.428    | 0.565     | +31.7%      | 0.019   |

**Statistical Analysis**: Cohen's d = 0.455 (medium effect), clinical validation achieves statistical significance (p = 0.019).

## Repository Structure

```
├── src/
│   ├── baseline/
│   │   └── pytorch_tensor_sketch.py       # Traditional tensor sketching
│   ├── learnable/
│   │   ├── working_phase2_sketch.py       # Main learnable framework
│   │   └── training_pipeline.py          # Training procedures
│   ├── evaluation/
│   │   ├── benchmarking_framework.py      # Comprehensive evaluation
│   │   ├── enhanced_genomic_validation.py # Clinical validation
│   │   └── benchmark_results/             # Verified results
│   └── advanced/                          # Additional architectures
├── examples/
│   └── basic_usage.py                     # Usage examples
├── docs/
│   └── installation.md                    # Detailed installation
└── requirements.txt                       # Dependencies
```

## Key Dependencies

- Python 3.8+
- PyTorch ≥1.9.0
- NumPy ≥1.21.0
- SciPy ≥1.7.0
- Pandas ≥1.3.0
- Scikit-learn ≥1.0.0

## Documentation

- [Installation Guide](docs/installation.md) - Detailed setup instructions
- [Basic Usage](examples/basic_usage.py) - Code examples and tutorials
- [API Reference](src/) - Complete source code documentation

## Citation

If you use this work, please cite:

```bibtex
@article{anonymous2025learnable,
  title={Learnable Tensor Sketching: A Hierarchical Framework for Genomic Sequence Similarity},
  author={Anonymous},
  journal={Under Review},
  year={2025},
  note={Submitted}
}
```

## License

Code will be made available under open source license upon publication.

## Contributing

We welcome contributions! Please feel free to submit issues, feature requests, or pull requests.

## Contact

**Anonymous Authors**  
Affiliation details will be provided upon publication  
Email: ati405[@]gmail.com

---

**Keywords**: tensor sketching, genomic sequences, machine learning, sequence similarity, bioinformatics, similarity estimation, clinical genomics
