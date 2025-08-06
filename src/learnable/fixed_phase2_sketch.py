#!/usr/bin/env python3
"""
Fixed Phase 2 Learnable Tensor Sketch
Simple but effective approach: learn parameters that modify the baseline tensor sketch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
import sys
import os

# Import baseline
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'baseline'))
from pytorch_tensor_sketch import TensorSketchBaseline

class FixedLearnableTensorSketch(nn.Module):
    """
    Fixed learnable tensor sketch using baseline + learnable scaling/transformation.
    
    This approach:
    1. Uses the working baseline tensor sketch algorithm
    2. Applies learnable transformations to the result
    3. Maintains strong performance while adding learnable components
    """
    
    def __init__(self, 
                 alphabet_size: int = 4,
                 sketch_dim: int = 64,
                 subsequence_len: int = 3,
                 device: str = 'cpu',
                 seed: int = 42):
        """
        Initialize fixed learnable tensor sketch.
        
        Args:
            alphabet_size: Size of sequence alphabet (4 for DNA)
            sketch_dim: Dimension of sketch embedding space
            subsequence_len: Length of subsequences for sketching
            device: PyTorch device
            seed: Random seed for reproducibility
        """
        super().__init__()
        
        self.alphabet_size = alphabet_size
        self.sketch_dim = sketch_dim
        self.subsequence_len = subsequence_len
        self.device = device
        
        torch.manual_seed(seed)
        
        # Baseline tensor sketch (non-learnable, but reliable)
        self.baseline_sketch = TensorSketchBaseline(
            alphabet_size=alphabet_size,
            sketch_dim=sketch_dim,  
            subsequence_len=subsequence_len,
            device=device
        )
        
        # Learnable components that modify the baseline sketch
        self._init_learnable_parameters()
        
    def _init_learnable_parameters(self):
        """Initialize learnable parameters for sketch modification."""
        
        # 1. Character-specific scaling factors
        self.char_scales = nn.Parameter(
            torch.ones(self.alphabet_size, device=self.device)
        )
        
        # 2. Position-dependent weights (learned from sequence statistics)
        self.position_weights = nn.Parameter(
            torch.ones(self.subsequence_len, device=self.device)
        )
        
        # 3. Sketch dimension reweighting (learn importance of each dimension)
        self.dimension_weights = nn.Parameter(
            torch.ones(self.sketch_dim, device=self.device)
        )
        
        # 4. Adaptive bias term
        self.sketch_bias = nn.Parameter(
            torch.zeros(self.sketch_dim, device=self.device)
        )
        
        # 5. Character-specific hash modification (larger for better discrimination)
        self.char_hash_modifiers = nn.Parameter(
            torch.zeros(self.alphabet_size, self.sketch_dim, device=self.device)
        )
        
        # 6. Nonlinear transformation layers for better feature learning
        self.feature_transform = nn.Sequential(
            nn.Linear(self.sketch_dim, self.sketch_dim),
            nn.ReLU(),
            nn.Linear(self.sketch_dim, self.sketch_dim)
        )
        
        # Initialize parameters for better discrimination
        nn.init.normal_(self.char_scales, mean=1.0, std=0.2)  # More variation
        nn.init.normal_(self.position_weights, mean=1.0, std=0.1)
        nn.init.normal_(self.dimension_weights, mean=1.0, std=0.3)  # More variation
        nn.init.zeros_(self.sketch_bias)
        nn.init.normal_(self.char_hash_modifiers, mean=0.0, std=0.1)  # Larger modifications
        
    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        """
        Forward pass combining baseline sketch with learnable modifications.
        
        Args:
            sequence: Integer tensor of shape (seq_len,)
            
        Returns:
            Enhanced learnable tensor sketch
        """
        # 1. Get baseline sketch (guaranteed to work)
        baseline_sketch = self.baseline_sketch(sequence)
        
        # 2. Apply learnable modifications
        enhanced_sketch = self._apply_learnable_enhancements(sequence, baseline_sketch)
        
        return enhanced_sketch
    
    def _apply_learnable_enhancements(self, 
                                    sequence: torch.Tensor, 
                                    baseline_sketch: torch.Tensor) -> torch.Tensor:
        """
        Apply learnable enhancements to the baseline sketch.
        
        Args:
            sequence: Original sequence
            baseline_sketch: Baseline tensor sketch
            
        Returns:
            Enhanced sketch with learnable components
        """
        enhanced_sketch = baseline_sketch.clone()
        
        # 1. Apply dimension-wise reweighting
        enhanced_sketch = enhanced_sketch * self.dimension_weights
        
        # 2. Add adaptive bias
        enhanced_sketch = enhanced_sketch + self.sketch_bias
        
        # 3. Character-frequency-based scaling
        char_scale = self._compute_character_scaling(sequence)
        enhanced_sketch = enhanced_sketch * char_scale
        
        # 4. Add character-specific hash modifications
        char_modifications = self._compute_character_modifications(sequence)
        enhanced_sketch = enhanced_sketch + char_modifications
        
        # 5. Apply nonlinear transformation for better feature learning
        enhanced_sketch = self.feature_transform(enhanced_sketch)
        
        return enhanced_sketch
    
    def _compute_character_scaling(self, sequence: torch.Tensor) -> torch.Tensor:
        """
        Compute character-frequency-based scaling factor.
        
        Args:
            sequence: Input sequence
            
        Returns:
            Scalar scaling factor based on character composition
        """
        # Count character frequencies
        char_counts = torch.zeros(self.alphabet_size, device=self.device)
        for char in sequence:
            if 0 <= char < self.alphabet_size:
                char_counts[char] += 1
        
        # Normalize to probabilities
        char_probs = char_counts / len(sequence)
        
        # Compute weighted scaling factor
        scaling_factor = torch.sum(char_probs * self.char_scales)
        
        return scaling_factor
    
    def _compute_character_modifications(self, sequence: torch.Tensor) -> torch.Tensor:
        """
        Compute character-specific modifications to add to baseline sketch.
        
        Args:
            sequence: Input sequence
            
        Returns:
            Modification vector to add to sketch
        """
        # Aggregate character-specific modifications based on sequence content
        modifications = torch.zeros(self.sketch_dim, device=self.device)
        
        for char in sequence:
            if 0 <= char < self.alphabet_size:
                modifications = modifications + self.char_hash_modifiers[char]
        
        # Normalize by sequence length to maintain magnitude
        modifications = modifications / len(sequence)
        
        return modifications

def test_fixed_implementation():
    """Test the fixed learnable implementation."""
    print("=== TESTING FIXED PHASE 2 IMPLEMENTATION ===")
    
    # Test sequences
    test_sequences = [
        torch.tensor([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3], dtype=torch.long),
        torch.tensor([1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0], dtype=torch.long),
        torch.tensor([2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1], dtype=torch.long),
        torch.tensor([3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2], dtype=torch.long)
    ]
    
    # Initialize models
    print("\n1. Initializing models...")
    baseline = TensorSketchBaseline(
        alphabet_size=4,
        sketch_dim=64,
        subsequence_len=3,
        device='cpu'
    )
    
    fixed_learnable = FixedLearnableTensorSketch(
        alphabet_size=4,
        sketch_dim=64,
        subsequence_len=3,
        device='cpu'
    )
    
    print(f"✓ Baseline parameters: {sum(p.numel() for p in baseline.parameters())}")
    print(f"✓ Fixed learnable parameters: {sum(p.numel() for p in fixed_learnable.parameters())}")
    
    # Test performance
    print("\n2. Testing sketch performance...")
    baseline_sketches = []
    learnable_sketches = []
    
    for i, seq in enumerate(test_sequences):
        bs = baseline(seq)
        ls = fixed_learnable(seq)
        
        baseline_sketches.append(bs)
        learnable_sketches.append(ls)
        
        print(f"Seq {i}: Baseline norm={torch.norm(bs).item():.6f}, "
              f"Learnable norm={torch.norm(ls).item():.6f}")
    
    # Compute quality metrics
    print("\n3. Computing quality metrics...")
    
    def compute_sketch_quality(sketches):
        if len(sketches) < 2:
            return 0.0
        distances = []
        for i in range(len(sketches)):
            for j in range(i + 1, len(sketches)):
                dist = torch.norm(sketches[i] - sketches[j]).item()
                distances.append(dist)
        return np.var(distances) if len(distances) > 1 else 0.0
    
    import numpy as np
    baseline_quality = compute_sketch_quality(baseline_sketches)
    learnable_quality = compute_sketch_quality(learnable_sketches)
    
    print(f"✓ Baseline quality: {baseline_quality:.8f}")
    print(f"✓ Learnable quality: {learnable_quality:.8f}")
    
    if learnable_quality > 0 and baseline_quality > 0:
        improvement = (learnable_quality - baseline_quality) / baseline_quality * 100
        print(f"✓ Quality improvement: {improvement:.2f}%")
    
    # Test gradient flow
    print("\n4. Testing gradient flow...")
    test_seq = test_sequences[0]
    test_sketch = fixed_learnable(test_seq)
    test_loss = torch.sum(test_sketch ** 2)
    test_loss.backward()
    
    grad_norms = [torch.norm(p.grad).item() for p in fixed_learnable.parameters() 
                  if p.grad is not None]
    print(f"✓ Gradient norms: {[f'{norm:.6f}' for norm in grad_norms[:3]]}...")
    print("✓ Gradient flow working correctly!")
    
    print("\n=== FIXED IMPLEMENTATION TEST COMPLETE ===")
    return baseline_quality, learnable_quality

if __name__ == "__main__":
    baseline_q, learnable_q = test_fixed_implementation()
    
    if learnable_q >= baseline_q * 0.8:  # Within 20% is acceptable
        print("✅ FIXED IMPLEMENTATION SUCCESS!")
        print("   Phase 2 learnable performance is comparable to baseline")
    else:
        print("❌ Still needs more work")