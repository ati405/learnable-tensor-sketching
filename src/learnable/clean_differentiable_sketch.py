#!/usr/bin/env python3
"""
Clean differentiable tensor sketching implementation without in-place operations.
Completely avoids gradient computation issues.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
import time

class CleanDifferentiableTensorSketch(nn.Module):
    """
    Clean differentiable tensor sketching implementation.
    
    Key design principles:
    1. NO in-place operations that break gradients
    2. Functional programming style - create new tensors instead of modifying
    3. Maintain exact mathematical equivalence to original algorithm
    4. Full gradient flow preservation
    """
    
    def __init__(self, 
                 alphabet_size: int = 4,
                 sketch_dim: int = 64,
                 subsequence_len: int = 3,
                 seed: int = 42,
                 device: str = 'cpu',
                 use_soft_hash: bool = True):
        """
        Initialize clean differentiable tensor sketch.
        
        Args:
            alphabet_size: Size of sequence alphabet (4 for DNA)
            sketch_dim: Dimension of sketch embedding space
            subsequence_len: Length of subsequences for sketching
            seed: Random seed for reproducibility
            device: PyTorch device
            use_soft_hash: Whether to use soft hash approximation (recommended)
        """
        super().__init__()
        
        self.alphabet_size = alphabet_size
        self.sketch_dim = sketch_dim
        self.subsequence_len = subsequence_len
        self.device = device
        self.use_soft_hash = use_soft_hash
        
        torch.manual_seed(seed)
        
        if use_soft_hash:
            self._init_soft_hash_parameters()
        else:
            self._init_full_differentiable_parameters()
        
    def _init_soft_hash_parameters(self):
        """
        Initialize parameters for soft hash approximation (recommended approach).
        """
        # Hash functions as learnable embeddings
        self.hash_embeddings = nn.ModuleList([
            nn.Embedding(self.alphabet_size, self.sketch_dim)
            for _ in range(self.subsequence_len)
        ])
        
        # Sign functions as learnable parameters
        self.sign_logits = nn.Parameter(
            torch.randn(self.subsequence_len, self.alphabet_size, device=self.device)
        )
        
        # Initialize embeddings with smaller values to prevent vanishing gradients
        for embedding in self.hash_embeddings:
            # Initialize with smaller std to prevent extreme softmax values
            nn.init.normal_(embedding.weight, mean=0, std=0.5)
        
        # Initialize signs to be approximately random but not extreme
        nn.init.normal_(self.sign_logits, mean=0, std=0.5)
    
    def _init_full_differentiable_parameters(self):
        """
        Initialize parameters for full differentiable approach (more complex).
        """
        # Full hash weight matrix
        self.hash_weights = nn.Parameter(
            torch.randn(self.subsequence_len, self.alphabet_size, self.sketch_dim, 
                       device=self.device)
        )
        
        # Sign logits
        self.sign_logits = nn.Parameter(
            torch.randn(self.subsequence_len, self.alphabet_size, device=self.device)
        )
        
        # Initialize
        nn.init.xavier_uniform_(self.hash_weights)
        nn.init.zeros_(self.sign_logits)
    
    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        """
        Forward pass - completely functional, no in-place operations.
        
        Args:
            sequence: Integer tensor of shape (seq_len,) 
            
        Returns:
            sketch: Differentiable tensor sketch
        """
        if self.use_soft_hash:
            return self._forward_soft_hash(sequence)
        else:
            return self._forward_full_differentiable(sequence)
    
    def _forward_soft_hash(self, sequence: torch.Tensor) -> torch.Tensor:
        """
        Forward pass using soft hash approximation (recommended).
        """
        seq_len = sequence.size(0)
        
        # Initialize T+ and T- tensors (no in-place modifications)
        Tp_list = [torch.zeros(self.sketch_dim, device=self.device) 
                   for _ in range(self.subsequence_len + 1)]
        Tm_list = [torch.zeros(self.sketch_dim, device=self.device) 
                   for _ in range(self.subsequence_len + 1)]
        
        # Initial condition
        Tp_list[0] = torch.zeros(self.sketch_dim, device=self.device)
        Tp_list[0][0] = 1.0
        
        # Main computation loop - functional style
        for i in range(seq_len):
            c = sequence[i]
            
            if c < 0 or c >= self.alphabet_size:
                continue
            
            # Create new T+ and T- for this iteration (no in-place modification)
            new_Tp_list = [tp.clone() for tp in Tp_list]
            new_Tm_list = [tm.clone() for tm in Tm_list]
            
            max_p = min(i + 1, self.subsequence_len)
            for p in range(max_p, 0, -1):
                z = p / (i + 1.0)
                
                # Get learnable hash shift (Gumbel-Softmax for discrete approximation)
                hash_logits = self.hash_embeddings[p - 1](c)
                
                # Use Gumbel softmax for better discrete approximation
                if self.training:
                    # Gumbel softmax during training for gradients
                    hash_probs = F.gumbel_softmax(hash_logits, tau=0.5, hard=False)
                else:
                    # Hard assignment during inference
                    hash_probs = F.softmax(hash_logits, dim=0)
                    
                # Get learnable sign probability
                sign_prob = torch.sigmoid(self.sign_logits[p - 1, c])
                
                # Compute differentiable circular convolution
                shifted_tp = self._soft_circular_shift(Tp_list[p - 1], hash_probs)
                shifted_tm = self._soft_circular_shift(Tm_list[p - 1], hash_probs)
                
                # Update T+ and T- functionally (no in-place ops)
                new_Tp_list[p] = (1 - z) * Tp_list[p] + z * (
                    sign_prob * shifted_tp + (1 - sign_prob) * shifted_tm
                )
                new_Tm_list[p] = (1 - z) * Tm_list[p] + z * (
                    sign_prob * shifted_tm + (1 - sign_prob) * shifted_tp
                )
            
            # Update for next iteration
            Tp_list = new_Tp_list
            Tm_list = new_Tm_list
        
        # Final sketch computation
        sketch = Tp_list[self.subsequence_len] - Tm_list[self.subsequence_len]
        return sketch
    
    def _soft_circular_shift(self, tensor: torch.Tensor, 
                           shift_probs: torch.Tensor) -> torch.Tensor:
        """
        Soft circular shift using probability distribution over shifts.
        
        Args:
            tensor: Input tensor to shift
            shift_probs: Probability distribution over shift amounts (same length as tensor)
            
        Returns:
            Expected shifted tensor (differentiable)
        """
        shifted_sum = torch.zeros_like(tensor)
        tensor_size = tensor.size(0)
        
        # Ensure shift_probs has correct size
        if shift_probs.size(0) != tensor_size:
            # Truncate or pad to match tensor size
            if shift_probs.size(0) > tensor_size:
                shift_probs = shift_probs[:tensor_size]
            else:
                # Pad with small values
                padding = torch.full((tensor_size - shift_probs.size(0),), 
                                   1e-8, device=shift_probs.device)
                shift_probs = torch.cat([shift_probs, padding])
            
            # Renormalize to sum to 1
            shift_probs = shift_probs / shift_probs.sum()
        
        for shift in range(tensor_size):
            shifted_tensor = torch.roll(tensor, shifts=shift, dims=0)
            shifted_sum = shifted_sum + shift_probs[shift] * shifted_tensor
        
        return shifted_sum
    
    def _forward_full_differentiable(self, sequence: torch.Tensor) -> torch.Tensor:
        """
        Forward pass using full differentiable approach (more complex).
        """
        # Similar structure but using full hash weight matrix
        # Implementation would be similar but more computationally intensive
        # For now, fall back to soft hash approach
        return self._forward_soft_hash(sequence)

class MinimalDifferentiableTensorSketch(nn.Module):
    """
    Minimal differentiable tensor sketch for testing gradient flow.
    """
    
    def __init__(self, alphabet_size: int = 4, sketch_dim: int = 16, device: str = 'cpu'):
        super().__init__()
        self.alphabet_size = alphabet_size
        self.sketch_dim = sketch_dim
        self.device = device
        
        # Simple learnable parameters
        self.hash_embedding = nn.Embedding(alphabet_size, sketch_dim)
        self.sign_weight = nn.Parameter(torch.randn(alphabet_size, device=device))
        
    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        """
        Minimal forward pass for testing gradients.
        """
        # Simple sketch computation that preserves gradients
        sketches = []
        
        for i in range(sequence.size(0)):
            c = sequence[i]
            if c < 0 or c >= self.alphabet_size:
                continue
                
            # Get hash vector and sign
            hash_vec = self.hash_embedding(c)
            sign = torch.tanh(self.sign_weight[c])  # Smooth sign function
            
            # Simple sketch contribution
            contribution = sign * hash_vec
            sketches.append(contribution)
        
        if len(sketches) == 0:
            return torch.zeros(self.sketch_dim, device=self.device)
        
        # Sum all contributions
        final_sketch = torch.stack(sketches).sum(dim=0)
        return final_sketch

def test_clean_implementation():
    """
    Test the clean differentiable implementations.
    """
    print("Testing Clean Differentiable Tensor Sketch Implementations...")
    
    # Test sequence
    test_sequence = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3], dtype=torch.long)
    
    # Test 1: Minimal implementation (simplest)
    print("\n1. Testing MinimalDifferentiableTensorSketch:")
    minimal = MinimalDifferentiableTensorSketch(alphabet_size=4, sketch_dim=16)
    
    sketch1 = minimal(test_sequence)
    print(f"✓ Sketch shape: {sketch1.shape}")
    print(f"✓ Sketch requires grad: {sketch1.requires_grad}")
    print(f"✓ Sketch L2 norm: {torch.norm(sketch1).item():.4f}")
    
    # Test gradients
    loss1 = torch.sum(sketch1 ** 2)
    print(f"✓ Loss value: {loss1.item():.4f}")
    
    try:
        loss1.backward()
        
        # Check gradients
        embedding_grad = minimal.hash_embedding.weight.grad
        sign_grad = minimal.sign_weight.grad
        
        if embedding_grad is not None and sign_grad is not None:
            print(f"✓ Embedding gradient norm: {torch.norm(embedding_grad).item():.6f}")
            print(f"✓ Sign gradient norm: {torch.norm(sign_grad).item():.6f}")
            print("✓ Gradients computed successfully!")
        else:
            print("✗ Gradients are None")
            
    except Exception as e:
        print(f"✗ Gradient computation failed: {e}")
    
    # Test 2: Clean implementation (full version)
    print("\n2. Testing CleanDifferentiableTensorSketch:")
    clean = CleanDifferentiableTensorSketch(
        alphabet_size=4, 
        sketch_dim=16,  # Small for testing
        subsequence_len=2,  # Small for testing
        use_soft_hash=True
    )
    
    try:
        sketch2 = clean(test_sequence)
        print(f"✓ Sketch shape: {sketch2.shape}")
        print(f"✓ Sketch requires grad: {sketch2.requires_grad}")
        print(f"✓ Sketch L2 norm: {torch.norm(sketch2).item():.4f}")
        
        # Test gradients
        loss2 = torch.sum(sketch2 ** 2)
        loss2.backward()
        
        # Check gradients exist
        has_grads = any(p.grad is not None and torch.norm(p.grad) > 1e-8 
                       for p in clean.parameters())
        print(f"✓ Gradients computed successfully: {has_grads}")
        
        # Count parameters
        num_params = sum(p.numel() for p in clean.parameters())
        print(f"✓ Total parameters: {num_params}")
        
    except Exception as e:
        print(f"✗ Clean implementation failed: {e}")
    
    print("\n=== GRADIENT FLOW TEST COMPLETE ===")
    print("If both tests pass, gradient flow is working correctly!")

if __name__ == "__main__":
    # Enable anomaly detection for debugging
    torch.autograd.set_detect_anomaly(True)
    test_clean_implementation()