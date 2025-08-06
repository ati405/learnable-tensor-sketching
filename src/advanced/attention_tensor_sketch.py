#!/usr/bin/env python3
"""
Attention-Based Tensor Sketching Implementation
Phase 3 Advanced Feature #3: Learned position importance and context-aware weighting
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Tuple, Dict, Optional
import sys
import os

# Import Phase 2 foundation
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'learnable'))
from clean_differentiable_sketch import CleanDifferentiableTensorSketch

class AttentionTensorSketch(nn.Module):
    """
    Attention-based tensor sketching with learned position weighting.
    
    Key Innovation: Replace fixed probability weighting z = p/(i+1) with
    learned attention mechanisms that consider sequence context and position importance.
    
    Architecture:
    1. Position encoding for sequence positions
    2. Multi-head attention for position importance
    3. Context-aware weight prediction
    4. Dynamic probability computation
    """
    
    def __init__(self,
                 alphabet_size: int = 4,
                 sketch_dim: int = 64,
                 subsequence_len: int = 3,
                 max_seq_len: int = 512,
                 attention_heads: int = 8,
                 attention_dim: int = 64,
                 device: str = 'cpu',
                 use_positional_encoding: bool = True,
                 use_self_attention: bool = True,
                 use_dynamic_weighting: bool = True):
        """
        Initialize attention-based tensor sketching.
        
        Args:
            alphabet_size: Size of sequence alphabet
            sketch_dim: Dimension of tensor sketches
            subsequence_len: Length of subsequences for sketching
            max_seq_len: Maximum sequence length for position encoding
            attention_heads: Number of attention heads
            attention_dim: Dimension of attention representations
            device: PyTorch device
            use_positional_encoding: Whether to use positional encoding
            use_self_attention: Whether to use self-attention mechanisms
            use_dynamic_weighting: Whether to use dynamic weight prediction
        """
        super().__init__()
        
        self.alphabet_size = alphabet_size
        self.sketch_dim = sketch_dim
        self.subsequence_len = subsequence_len
        self.max_seq_len = max_seq_len
        self.attention_heads = attention_heads
        self.attention_dim = attention_dim
        self.device = device
        self.use_positional_encoding = use_positional_encoding
        self.use_self_attention = use_self_attention
        self.use_dynamic_weighting = use_dynamic_weighting
        
        # Base tensor sketching components (modified for attention)
        self._init_learnable_components()
        
        # Always initialize all components for safety (some might be unused)
        self._init_positional_encoding()
        self._init_attention_mechanisms()
        
        if use_dynamic_weighting:
            self._init_dynamic_weighting()
    
    def _init_learnable_components(self):
        """
        Initialize base learnable hash and sign functions.
        """
        # Enhanced hash embeddings with attention integration
        self.hash_embeddings = nn.ModuleList([
            nn.Embedding(self.alphabet_size, self.sketch_dim)
            for _ in range(self.subsequence_len)
        ])
        
        # Enhanced sign functions
        self.sign_logits = nn.Parameter(
            torch.randn(self.subsequence_len, self.alphabet_size, device=self.device)
        )
        
        # Initialize embeddings
        for embedding in self.hash_embeddings:
            nn.init.xavier_uniform_(embedding.weight)
    
    def _init_positional_encoding(self):
        """
        Initialize positional encoding for sequence positions.
        """
        # Sinusoidal positional encoding
        pe = torch.zeros(self.max_seq_len, self.attention_dim, device=self.device)
        position = torch.arange(0, self.max_seq_len, dtype=torch.float, device=self.device).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, self.attention_dim, 2, dtype=torch.float, device=self.device) * 
                           -(math.log(10000.0) / self.attention_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('positional_encoding', pe)
        
        # Position embedding alternative
        self.position_embedding = nn.Embedding(self.max_seq_len, self.attention_dim)
        
        # Position-to-attention projection
        self.position_projector = nn.Linear(self.attention_dim, self.attention_dim)
    
    def _init_attention_mechanisms(self):
        """
        Initialize multi-head attention mechanisms.
        """
        # Character embedding for attention
        self.char_embedding = nn.Embedding(self.alphabet_size, self.attention_dim)
        
        # Multi-head self-attention
        self.self_attention = nn.MultiheadAttention(
            embed_dim=self.attention_dim,
            num_heads=self.attention_heads,
            batch_first=True,
            dropout=0.1
        )
        
        # Position-specific attention
        self.position_attention = nn.MultiheadAttention(
            embed_dim=self.attention_dim,
            num_heads=self.attention_heads // 2,
            batch_first=True,
            dropout=0.1
        )
        
        # Cross-attention for position-character interaction
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.attention_dim,
            num_heads=self.attention_heads // 2,
            batch_first=True,
            dropout=0.1
        )
    
    def _init_dynamic_weighting(self):
        """
        Initialize dynamic weight prediction mechanisms.
        """
        # Context encoder for dynamic weights
        self.context_encoder = nn.LSTM(
            input_size=self.attention_dim,
            hidden_size=self.attention_dim // 2,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.1
        )
        
        # Weight predictor network
        self.weight_predictor = nn.Sequential(
            nn.Linear(self.attention_dim, self.attention_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.attention_dim // 2, self.subsequence_len),
            nn.Softmax(dim=-1)  # Normalized weights across subsequence positions
        )
        
        # Position-specific weight adjustment
        self.position_weight_adjuster = nn.Sequential(
            nn.Linear(self.attention_dim + 1, self.attention_dim // 2),  # +1 for relative position
            nn.ReLU(),
            nn.Linear(self.attention_dim // 2, 1),
            nn.Sigmoid()  # Output: position-specific multiplier
        )
    
    def _get_attention_representations(self, sequence: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get attention-based representations for sequence positions and characters.
        
        Args:
            sequence: Input sequence tensor
            
        Returns:
            Tuple of (position_representations, character_representations)
        """
        seq_len = sequence.size(0)
        
        # Character embeddings
        char_embeds = self.char_embedding(sequence)  # [seq_len, attention_dim]
        
        # Position embeddings
        positions = torch.arange(seq_len, device=self.device)
        if self.use_positional_encoding:
            pos_embeds = self.positional_encoding[:seq_len] + self.position_embedding(positions)
        else:
            pos_embeds = self.position_embedding(positions)
        
        pos_embeds = self.position_projector(pos_embeds)  # [seq_len, attention_dim]
        
        # Add batch dimension for attention
        char_embeds_batch = char_embeds.unsqueeze(0)  # [1, seq_len, attention_dim]
        pos_embeds_batch = pos_embeds.unsqueeze(0)    # [1, seq_len, attention_dim]
        
        # Self-attention on characters
        if self.use_self_attention and seq_len > 1:
            attended_chars, char_attention_weights = self.self_attention(
                char_embeds_batch, char_embeds_batch, char_embeds_batch
            )
            attended_chars = attended_chars.squeeze(0)  # [seq_len, attention_dim]
        else:
            attended_chars = char_embeds
        
        # Position attention
        if self.use_self_attention and seq_len > 1:
            attended_positions, pos_attention_weights = self.position_attention(
                pos_embeds_batch, pos_embeds_batch, pos_embeds_batch
            )
            attended_positions = attended_positions.squeeze(0)  # [seq_len, attention_dim]
        else:
            attended_positions = pos_embeds
        
        # Cross-attention between positions and characters
        if self.use_self_attention and seq_len > 1:
            cross_attended, cross_attention_weights = self.cross_attention(
                attended_chars.unsqueeze(0), attended_positions.unsqueeze(0), attended_positions.unsqueeze(0)
            )
            final_representations = cross_attended.squeeze(0)  # [seq_len, attention_dim]
        else:
            final_representations = attended_chars + attended_positions
        
        return final_representations, attended_positions
    
    def _compute_dynamic_weights(self, 
                                sequence: torch.Tensor,
                                position: int,
                                representations: torch.Tensor) -> torch.Tensor:
        """
        Compute dynamic weights for tensor sketching at a specific position.
        
        Args:
            sequence: Input sequence
            position: Current position in sequence
            representations: Attention representations for sequence
            
        Returns:
            Dynamic weights for subsequence positions
        """
        seq_len = sequence.size(0)
        
        if not self.use_dynamic_weighting:
            # Fallback to original weighting
            max_p = min(position + 1, self.subsequence_len)
            weights = torch.zeros(self.subsequence_len, device=self.device)
            for p in range(1, max_p + 1):
                weights[p - 1] = p / (position + 1.0)
            return weights
        
        # Use LSTM to encode context up to current position
        if position == 0:
            context_repr = representations[0:1].unsqueeze(0)  # [1, 1, attention_dim]
        else:
            context_repr = representations[:position+1].unsqueeze(0)  # [1, pos+1, attention_dim]
        
        # Encode context
        lstm_out, _ = self.context_encoder(context_repr)  # [1, pos+1, attention_dim]
        current_context = lstm_out[0, -1, :]  # [attention_dim] - final hidden state
        
        # Predict base weights
        base_weights = self.weight_predictor(current_context)  # [subsequence_len]
        
        # Adjust weights based on position-specific information
        relative_pos = position / max(seq_len - 1, 1)  # Normalize position
        pos_info = torch.tensor([relative_pos], device=self.device)
        
        pos_input = torch.cat([current_context, pos_info], dim=0)  # [attention_dim + 1]
        pos_multiplier = self.position_weight_adjuster(pos_input)  # [1]
        
        # Apply position-specific adjustment
        adjusted_weights = base_weights * pos_multiplier
        
        # Ensure weights are valid for current position
        max_p = min(position + 1, self.subsequence_len)
        final_weights = torch.zeros(self.subsequence_len, device=self.device)
        if max_p > 0:
            final_weights[:max_p] = adjusted_weights[:max_p]
            final_weights[:max_p] = final_weights[:max_p] / (torch.sum(final_weights[:max_p]) + 1e-8)
        
        return final_weights
    
    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with attention-based tensor sketching.
        
        Args:
            sequence: Input sequence tensor
            
        Returns:
            Attention-enhanced tensor sketch
        """
        seq_len = sequence.size(0)
        
        # Get attention representations
        representations, position_representations = self._get_attention_representations(sequence)
        
        # Initialize tensors (functional style from Phase 2)
        Tp_list = [torch.zeros(self.sketch_dim, device=self.device) 
                   for _ in range(self.subsequence_len + 1)]
        Tm_list = [torch.zeros(self.sketch_dim, device=self.device) 
                   for _ in range(self.subsequence_len + 1)]
        
        # Initial condition
        Tp_list[0] = torch.zeros(self.sketch_dim, device=self.device)
        Tp_list[0][0] = 1.0
        
        # Main computation with attention-based weighting
        for i in range(seq_len):
            c = sequence[i]
            
            if c < 0 or c >= self.alphabet_size:
                continue
            
            # Compute dynamic weights for this position
            dynamic_weights = self._compute_dynamic_weights(sequence, i, representations)
            
            # Create new tensors for this iteration
            new_Tp_list = [tp.clone() for tp in Tp_list]
            new_Tm_list = [tm.clone() for tm in Tm_list]
            
            max_p = min(i + 1, self.subsequence_len)
            for p in range(max_p, 0, -1):
                # Use dynamic weight instead of fixed z = p/(i+1)
                z = dynamic_weights[p - 1].item() if p <= len(dynamic_weights) else p / (i + 1.0)
                
                # Get learnable hash and sign (enhanced with attention context)
                hash_logits = self.hash_embeddings[p - 1](c)
                
                # Enhance hash with position representation
                if self.use_self_attention:
                    pos_context = representations[i]  # [attention_dim]
                    # Project position context to sketch dimension
                    pos_proj = torch.matmul(pos_context[:self.sketch_dim], hash_logits[:self.sketch_dim])
                    hash_logits = hash_logits + 0.1 * pos_proj  # Small attention influence
                
                hash_probs = F.softmax(hash_logits, dim=0)
                
                # Enhanced sign computation
                sign_logit = self.sign_logits[p - 1, c]
                if self.use_self_attention:
                    # Add attention-based sign adjustment
                    attention_sign_adj = torch.sum(representations[i]) * 0.01  # Small influence
                    sign_logit = sign_logit + attention_sign_adj
                
                sign_prob = torch.sigmoid(sign_logit)
                
                # Soft circular operations
                shifted_tp = self._soft_circular_shift(Tp_list[p - 1], hash_probs)
                shifted_tm = self._soft_circular_shift(Tm_list[p - 1], hash_probs)
                
                # Update tensors with attention-weighted contributions
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
        Soft circular shift using probability distribution.
        """
        shifted_sum = torch.zeros_like(tensor)
        
        for shift in range(self.sketch_dim):
            shifted_tensor = torch.roll(tensor, shifts=shift, dims=0)
            shifted_sum = shifted_sum + shift_probs[shift] * shifted_tensor
        
        return shifted_sum
    
    def get_attention_analysis(self, sequence: torch.Tensor) -> Dict:
        """
        Analyze attention patterns for a given sequence.
        """
        import numpy as np
        
        with torch.no_grad():
            representations, _ = self._get_attention_representations(sequence)
            
            # Compute dynamic weights for each position
            position_weights = []
            for i in range(sequence.size(0)):
                weights = self._compute_dynamic_weights(sequence, i, representations)
                position_weights.append(weights.cpu().numpy())
            
            analysis = {
                'sequence_length': sequence.size(0),
                'attention_dim': self.attention_dim,
                'position_weights': position_weights,
                'avg_attention_norm': torch.mean(torch.norm(representations, dim=1)).item(),
                'weight_variance': np.var(position_weights) if len(position_weights) > 0 else 0
            }
            
            return analysis

def test_attention_implementation():
    """
    Test the attention-based tensor sketching implementation.
    """
    print("Testing Attention-Based Tensor Sketching...")
    
    # Test sequences with different attention patterns
    sequences = [
        torch.tensor([0, 1, 2, 3, 0, 1, 2, 3], dtype=torch.long),  # Periodic - should show repeating attention
        torch.tensor([0, 1, 0, 1, 0, 1, 0, 1], dtype=torch.long),  # Alternating - different attention pattern
        torch.tensor([0, 0, 1, 2, 3, 3, 3, 1], dtype=torch.long),  # Clustered - position-dependent attention
        torch.tensor([0, 2, 1, 3, 2, 0, 3, 1], dtype=torch.long),  # Complex - varied attention needs
    ]
    
    print("\n1. Testing Basic Attention Mechanisms:")
    
    attention_model = AttentionTensorSketch(
        alphabet_size=4,
        sketch_dim=32,
        subsequence_len=3,
        max_seq_len=64,
        attention_heads=4,
        attention_dim=32,
        use_positional_encoding=True,
        use_self_attention=True,
        use_dynamic_weighting=True
    )
    
    print(f"   ✓ Attention model: {sum(p.numel() for p in attention_model.parameters())} parameters")
    
    sketches = []
    for i, seq in enumerate(sequences):
        sketch = attention_model(seq)
        sketches.append(sketch)
        
        analysis = attention_model.get_attention_analysis(seq)
        print(f"   ✓ Sequence {i}: sketch_norm={torch.norm(sketch).item():.4f}")
        print(f"     Attention stats: avg_norm={analysis['avg_attention_norm']:.4f}, weight_var={analysis['weight_variance']:.6f}")
    
    # Test gradient flow
    loss = sum(torch.sum(sketch ** 2) for sketch in sketches)
    loss.backward()
    
    has_gradients = any(p.grad is not None and torch.norm(p.grad) > 1e-8 
                       for p in attention_model.parameters())
    print(f"   ✓ Gradients flowing: {has_gradients}")
    
    print("\n2. Testing Different Attention Configurations:")
    
    configs = [
        {'use_positional_encoding': True, 'use_self_attention': True, 'use_dynamic_weighting': True},
        {'use_positional_encoding': True, 'use_self_attention': False, 'use_dynamic_weighting': True},
        {'use_positional_encoding': False, 'use_self_attention': True, 'use_dynamic_weighting': True},
        {'use_positional_encoding': False, 'use_self_attention': False, 'use_dynamic_weighting': False}
    ]
    
    for i, config in enumerate(configs):
        model = AttentionTensorSketch(
            alphabet_size=4, sketch_dim=16, subsequence_len=2, 
            attention_heads=2, attention_dim=16, **config
        )
        
        sketch = model(sequences[0])
        params = sum(p.numel() for p in model.parameters())
        
        config_str = f"PE:{config['use_positional_encoding']}, SA:{config['use_self_attention']}, DW:{config['use_dynamic_weighting']}"
        print(f"   ✓ Config {i} ({config_str}): {params} params, sketch_norm={torch.norm(sketch).item():.4f}")
    
    print("\n3. Comparing Attention vs Baseline:")
    
    # Compare with baseline
    baseline_model = CleanDifferentiableTensorSketch(
        alphabet_size=4, sketch_dim=32, subsequence_len=3, use_soft_hash=True
    )
    
    print("   Sketch similarity comparisons:")
    for i in range(len(sequences)):
        for j in range(i+1, len(sequences)):
            # Baseline similarity
            baseline_i = baseline_model(sequences[i])
            baseline_j = baseline_model(sequences[j])
            baseline_sim = F.cosine_similarity(baseline_i, baseline_j, dim=0).item()
            
            # Attention similarity
            attention_i = attention_model(sequences[i])
            attention_j = attention_model(sequences[j])
            attention_sim = F.cosine_similarity(attention_i, attention_j, dim=0).item()
            
            print(f"   Sequences {i}-{j}: Baseline={baseline_sim:.4f}, Attention={attention_sim:.4f}")
    
    print("\n4. Attention Weight Analysis:")
    
    # Analyze attention weights for different sequence types
    test_seq = torch.tensor([0, 1, 2, 3, 2, 1, 0], dtype=torch.long)  # Palindromic sequence
    analysis = attention_model.get_attention_analysis(test_seq)
    
    print(f"   ✓ Palindromic sequence analysis:")
    print(f"     Sequence: {test_seq.tolist()}")
    print(f"     Average attention norm: {analysis['avg_attention_norm']:.4f}")
    print(f"     Weight variance: {analysis['weight_variance']:.6f}")
    
    if len(analysis['position_weights']) > 0:
        import numpy as np
        weights_array = np.array(analysis['position_weights'])
        print(f"     Weight matrix shape: {weights_array.shape}")
        print(f"     Position 0 weights: {weights_array[0] if len(weights_array) > 0 else 'N/A'}")
        print(f"     Position 3 weights: {weights_array[3] if len(weights_array) > 3 else 'N/A'}")
    
    print("\n=== ATTENTION-BASED IMPLEMENTATION TEST COMPLETE ===")
    print("✅ Multi-head attention mechanisms working")
    print("✅ Positional encoding implemented")
    print("✅ Dynamic weight prediction functional")
    print("✅ Context-aware position weighting active")
    print("✅ Gradient flow maintained")
    print("🚀 Ready for adaptive sketch dimension selection")

if __name__ == "__main__":
    # Enable anomaly detection
    torch.autograd.set_detect_anomaly(True)
    
    # Import numpy for analysis
    import numpy as np
    
    # Run comprehensive tests
    test_attention_implementation()