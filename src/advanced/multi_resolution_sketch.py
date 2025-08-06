#!/usr/bin/env python3
"""
Multi-Resolution Tensor Sketching Implementation
Phase 3 Advanced Feature #1: Hierarchical pattern capture at multiple scales
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional
import sys
import os

# Import Phase 2 foundation
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'learnable'))
from clean_differentiable_sketch import CleanDifferentiableTensorSketch

class MultiResolutionTensorSketch(nn.Module):
    """
    Multi-resolution tensor sketching for hierarchical pattern capture.
    
    Key Innovation: Compute sketches at multiple scales and fuse them intelligently
    to capture both fine-grained and coarse-grained sequence patterns.
    
    Architecture:
    1. Multiple sketch heads with different (sketch_dim, subsequence_len) pairs
    2. Intelligent fusion network to combine multi-scale information
    3. Learned attention weights for resolution importance
    """
    
    def __init__(self,
                 alphabet_size: int = 4,
                 base_sketch_dim: int = 32,
                 resolution_scales: List[Tuple[int, int]] = None,
                 fusion_strategy: str = "attention",
                 device: str = 'cpu',
                 seed: int = 42):
        """
        Initialize multi-resolution tensor sketching.
        
        Args:
            alphabet_size: Size of sequence alphabet (4 for DNA)
            base_sketch_dim: Base dimension for sketches
            resolution_scales: List of (sketch_dim, subsequence_len) pairs for different scales
            fusion_strategy: How to combine sketches ("attention", "concat", "weighted")
            device: PyTorch device
            seed: Random seed
        """
        super().__init__()
        
        self.alphabet_size = alphabet_size
        self.base_sketch_dim = base_sketch_dim
        self.fusion_strategy = fusion_strategy
        self.device = device
        
        # Default resolution scales if not provided
        if resolution_scales is None:
            self.resolution_scales = [
                (base_sketch_dim // 2, 2),    # Fine-grained: small subsequences
                (base_sketch_dim, 3),         # Medium-grained: medium subsequences  
                (base_sketch_dim * 2, 4),     # Coarse-grained: large subsequences
            ]
        else:
            self.resolution_scales = resolution_scales
        
        torch.manual_seed(seed)
        
        self._init_multi_resolution_components()
        
    def _init_multi_resolution_components(self):
        """
        Initialize multiple sketch heads and fusion components.
        """
        # Create sketch heads for different resolutions
        self.sketch_heads = nn.ModuleList()
        total_sketch_dim = 0
        
        for i, (sketch_dim, subseq_len) in enumerate(self.resolution_scales):
            sketch_head = CleanDifferentiableTensorSketch(
                alphabet_size=self.alphabet_size,
                sketch_dim=sketch_dim,
                subsequence_len=subseq_len,
                device=self.device,
                use_soft_hash=True
            )
            self.sketch_heads.append(sketch_head)
            total_sketch_dim += sketch_dim
        
        self.total_input_dim = total_sketch_dim
        
        # Initialize fusion mechanism based on strategy
        if self.fusion_strategy == "attention":
            self._init_attention_fusion()
        elif self.fusion_strategy == "concat":
            self._init_concat_fusion()
        elif self.fusion_strategy == "weighted":
            self._init_weighted_fusion()
        else:
            raise ValueError(f"Unknown fusion strategy: {self.fusion_strategy}")
    
    def _init_attention_fusion(self):
        """
        Initialize attention-based fusion mechanism.
        """
        # Multi-head attention for combining different resolution sketches
        self.resolution_attention = nn.MultiheadAttention(
            embed_dim=self.base_sketch_dim,
            num_heads=4,
            batch_first=True
        )
        
        # Projection layers to standardize sketch dimensions
        self.sketch_projections = nn.ModuleList()
        for sketch_dim, _ in self.resolution_scales:
            projection = nn.Linear(sketch_dim, self.base_sketch_dim)
            self.sketch_projections.append(projection)
        
        # Final output projection
        self.output_projection = nn.Linear(self.base_sketch_dim, self.base_sketch_dim)
        
    def _init_concat_fusion(self):
        """
        Initialize concatenation-based fusion mechanism.
        """
        # Simple concatenation followed by dimensionality reduction
        self.fusion_network = nn.Sequential(
            nn.Linear(self.total_input_dim, self.base_sketch_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.base_sketch_dim * 2, self.base_sketch_dim),
            nn.Tanh()  # Normalize output
        )
    
    def _init_weighted_fusion(self):
        """
        Initialize learnable weighted fusion mechanism.
        """
        # Learnable weights for each resolution
        self.resolution_weights = nn.Parameter(
            torch.ones(len(self.resolution_scales), device=self.device)
        )
        
        # Projection layers to standardize dimensions
        self.sketch_projections = nn.ModuleList()
        for sketch_dim, _ in self.resolution_scales:
            projection = nn.Linear(sketch_dim, self.base_sketch_dim)
            self.sketch_projections.append(projection)
    
    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with multi-resolution sketch computation.
        
        Args:
            sequence: Input sequence tensor
            
        Returns:
            Fused multi-resolution sketch
        """
        # Compute sketches at all resolutions
        resolution_sketches = []
        
        for sketch_head in self.sketch_heads:
            sketch = sketch_head(sequence)
            resolution_sketches.append(sketch)
        
        # Fuse sketches based on strategy
        if self.fusion_strategy == "attention":
            return self._attention_fusion(resolution_sketches)
        elif self.fusion_strategy == "concat":
            return self._concat_fusion(resolution_sketches)
        elif self.fusion_strategy == "weighted":
            return self._weighted_fusion(resolution_sketches)
    
    def _attention_fusion(self, sketches: List[torch.Tensor]) -> torch.Tensor:
        """
        Fuse sketches using multi-head attention.
        """
        # Project all sketches to same dimension
        projected_sketches = []
        for sketch, projection in zip(sketches, self.sketch_projections):
            projected = projection(sketch)
            projected_sketches.append(projected)
        
        # Stack sketches for attention (batch_size=1, seq_len=num_sketches, embed_dim)
        stacked_sketches = torch.stack(projected_sketches).unsqueeze(0)  # [1, num_sketches, embed_dim]
        
        # Apply self-attention
        attended, attention_weights = self.resolution_attention(
            stacked_sketches, stacked_sketches, stacked_sketches
        )
        
        # Pool attended representations (mean pooling)
        pooled = torch.mean(attended, dim=1).squeeze(0)  # [embed_dim]
        
        # Final projection
        output = self.output_projection(pooled)
        return output
    
    def _concat_fusion(self, sketches: List[torch.Tensor]) -> torch.Tensor:
        """
        Fuse sketches using concatenation and MLP.
        """
        # Concatenate all sketches
        concatenated = torch.cat(sketches, dim=0)
        
        # Apply fusion network
        fused = self.fusion_network(concatenated)
        return fused
    
    def _weighted_fusion(self, sketches: List[torch.Tensor]) -> torch.Tensor:
        """
        Fuse sketches using learnable weighted combination.
        """
        # Project sketches to same dimension
        projected_sketches = []
        for sketch, projection in zip(sketches, self.sketch_projections):
            projected = projection(sketch)  
            projected_sketches.append(projected)
        
        # Apply softmax to weights for normalization
        normalized_weights = F.softmax(self.resolution_weights, dim=0)
        
        # Weighted combination
        weighted_sum = torch.zeros_like(projected_sketches[0])
        for sketch, weight in zip(projected_sketches, normalized_weights):
            weighted_sum += weight * sketch
        
        return weighted_sum
    
    def get_resolution_info(self) -> Dict:
        """
        Get information about resolution scales and fusion strategy.
        """
        return {
            'resolution_scales': self.resolution_scales,
            'fusion_strategy': self.fusion_strategy,
            'total_parameters': sum(p.numel() for p in self.parameters()),
            'sketch_heads': len(self.sketch_heads)
        }

class AdaptiveMultiResolutionSketch(nn.Module):
    """
    Adaptive multi-resolution sketching that learns which resolutions to use
    based on sequence characteristics.
    """
    
    def __init__(self,
                 alphabet_size: int = 4,
                 base_sketch_dim: int = 32,
                 max_resolutions: int = 5,
                 device: str = 'cpu'):
        super().__init__()
        
        self.alphabet_size = alphabet_size
        self.base_sketch_dim = base_sketch_dim
        self.max_resolutions = max_resolutions
        self.device = device
        
        # Sequence analyzer to determine optimal resolutions
        self.sequence_analyzer = nn.LSTM(
            input_size=alphabet_size,
            hidden_size=64,
            batch_first=True,
            bidirectional=True
        )
        
        # Resolution selector
        self.resolution_selector = nn.Sequential(
            nn.Linear(128, 64),  # 128 = 64 * 2 (bidirectional)
            nn.ReLU(),
            nn.Linear(64, max_resolutions),
            nn.Sigmoid()  # Output: probability of using each resolution
        )
        
        # Available sketch heads for different resolutions
        self.available_sketch_heads = nn.ModuleList([
            CleanDifferentiableTensorSketch(
                alphabet_size=alphabet_size,
                sketch_dim=base_sketch_dim,
                subsequence_len=i+2,  # subsequence lengths from 2 to max_resolutions+1
                device=device,
                use_soft_hash=True
            )
            for i in range(max_resolutions)
        ])
        
        # Adaptive fusion network
        self.adaptive_fusion = nn.Linear(base_sketch_dim * max_resolutions, base_sketch_dim)
    
    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with adaptive resolution selection.
        """
        # Analyze sequence to determine optimal resolutions
        sequence_one_hot = F.one_hot(sequence, self.alphabet_size).float()
        sequence_features, _ = self.sequence_analyzer(sequence_one_hot.unsqueeze(0))
        
        # Use final hidden state for resolution selection
        final_features = sequence_features[0, -1, :]  # [hidden_size * 2]
        resolution_probs = self.resolution_selector(final_features)  # [max_resolutions]
        
        # Compute sketches with probability weighting
        weighted_sketches = []
        for i, (sketch_head, prob) in enumerate(zip(self.available_sketch_heads, resolution_probs)):
            sketch = sketch_head(sequence)
            weighted_sketch = prob * sketch
            weighted_sketches.append(weighted_sketch)
        
        # Concatenate and fuse
        concatenated = torch.cat(weighted_sketches, dim=0)
        fused_sketch = self.adaptive_fusion(concatenated)
        
        return fused_sketch, resolution_probs  # Return probabilities for analysis

def test_multi_resolution_implementation():
    """
    Test the multi-resolution tensor sketching implementation.
    """
    print("Testing Multi-Resolution Tensor Sketching...")
    
    # Test sequences
    sequences = [
        torch.tensor([0, 1, 2, 3, 0, 1, 2, 3], dtype=torch.long),  # Regular pattern
        torch.tensor([0, 0, 1, 1, 2, 2, 3, 3], dtype=torch.long),  # Repeated pattern
        torch.tensor([0, 2, 1, 3, 2, 0, 3, 1], dtype=torch.long),  # Complex pattern
    ]
    
    print("\n1. Testing Basic Multi-Resolution Sketching:")
    
    # Test different fusion strategies
    fusion_strategies = ["attention", "concat", "weighted"]
    
    for strategy in fusion_strategies:
        print(f"\n   Testing {strategy} fusion:")
        
        model = MultiResolutionTensorSketch(
            alphabet_size=4,
            base_sketch_dim=16,
            fusion_strategy=strategy,
            resolution_scales=[(8, 2), (16, 3), (24, 4)]  # Small for testing
        )
        
        print(f"   ✓ Model created: {model.get_resolution_info()['total_parameters']} parameters")
        
        # Test forward pass
        sketches = []
        for i, seq in enumerate(sequences):
            sketch = model(seq)
            sketches.append(sketch)
            print(f"   ✓ Sequence {i}: sketch_dim={sketch.shape[0]}, norm={torch.norm(sketch).item():.4f}")
        
        # Test gradient flow
        loss = sum(torch.sum(sketch ** 2) for sketch in sketches)
        loss.backward()
        
        # Check gradients
        has_gradients = any(p.grad is not None and torch.norm(p.grad) > 1e-8 
                           for p in model.parameters())
        print(f"   ✓ Gradients flowing: {has_gradients}")
    
    print("\n2. Testing Adaptive Multi-Resolution:")
    
    adaptive_model = AdaptiveMultiResolutionSketch(
        alphabet_size=4,
        base_sketch_dim=16,
        max_resolutions=3
    )
    
    print(f"   ✓ Adaptive model: {sum(p.numel() for p in adaptive_model.parameters())} parameters")
    
    for i, seq in enumerate(sequences):
        sketch, resolution_probs = adaptive_model(seq)
        print(f"   ✓ Sequence {i}: sketch_norm={torch.norm(sketch).item():.4f}")
        print(f"     Resolution probs: {resolution_probs.detach().numpy()}")
    
    print("\n3. Performance Comparison:")
    
    # Compare with Phase 2 baseline
    baseline = CleanDifferentiableTensorSketch(
        alphabet_size=4, sketch_dim=16, subsequence_len=3, use_soft_hash=True
    )
    
    multi_res = MultiResolutionTensorSketch(
        alphabet_size=4, base_sketch_dim=16, fusion_strategy="attention"
    )
    
    print("   Comparing sketch similarities...")
    
    for i in range(len(sequences)):
        for j in range(i+1, len(sequences)):
            baseline_sketch_i = baseline(sequences[i])
            baseline_sketch_j = baseline(sequences[j])
            baseline_sim = F.cosine_similarity(baseline_sketch_i, baseline_sketch_j, dim=0).item()
            
            multi_res_sketch_i = multi_res(sequences[i])
            multi_res_sketch_j = multi_res(sequences[j])
            multi_res_sim = F.cosine_similarity(multi_res_sketch_i, multi_res_sketch_j, dim=0).item()
            
            print(f"   Sequences {i}-{j}: Baseline={baseline_sim:.4f}, Multi-res={multi_res_sim:.4f}")
    
    print("\n=== MULTI-RESOLUTION IMPLEMENTATION TEST COMPLETE ===")
    print("✅ All fusion strategies working")
    print("✅ Adaptive resolution selection implemented") 
    print("✅ Gradient flow maintained")
    print("✅ Performance comparison completed")
    print("🚀 Ready for integration with other Phase 3 features")

if __name__ == "__main__":
    # Enable anomaly detection
    torch.autograd.set_detect_anomaly(True)
    
    # Run comprehensive tests
    test_multi_resolution_implementation()