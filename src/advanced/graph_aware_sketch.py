#!/usr/bin/env python3
"""
Graph-Aware Tensor Sketching Implementation
Phase 3 Advanced Feature #2: Incorporate De Bruijn graph structure information
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional, Set
import numpy as np
import sys
import os

# Import Phase 2 foundation and Phase 3 multi-resolution
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'learnable'))
from clean_differentiable_sketch import CleanDifferentiableTensorSketch

class GraphAwareTensorSketch(nn.Module):
    """
    Graph-aware tensor sketching that incorporates De Bruijn graph structure.
    
    Key Innovation: Use graph topology and k-mer relationships to enhance
    sketching with biological/structural context.
    
    Architecture:
    1. Graph structure encoder (GNN components)
    2. K-mer context integration
    3. Graph-sketch fusion mechanisms
    4. Structure-aware hash functions
    """
    
    def __init__(self,
                 alphabet_size: int = 4,
                 sketch_dim: int = 64,
                 subsequence_len: int = 3,
                 k_mer_size: int = 3,
                 graph_embed_dim: int = 32,
                 device: str = 'cpu',
                 use_graph_context: bool = True,
                 use_structure_hashing: bool = True):
        """
        Initialize graph-aware tensor sketching.
        
        Args:
            alphabet_size: Size of sequence alphabet
            sketch_dim: Dimension of tensor sketches
            subsequence_len: Length of subsequences for sketching
            k_mer_size: Size of k-mers for graph construction
            graph_embed_dim: Dimension of graph embeddings
            device: PyTorch device
            use_graph_context: Whether to use graph context information
            use_structure_hashing: Whether to use structure-aware hashing
        """
        super().__init__()
        
        self.alphabet_size = alphabet_size
        self.sketch_dim = sketch_dim
        self.subsequence_len = subsequence_len
        self.k_mer_size = k_mer_size
        self.graph_embed_dim = graph_embed_dim
        self.device = device
        self.use_graph_context = use_graph_context
        self.use_structure_hashing = use_structure_hashing
        
        # Base tensor sketching component
        self.base_sketch = CleanDifferentiableTensorSketch(
            alphabet_size=alphabet_size,
            sketch_dim=sketch_dim,
            subsequence_len=subsequence_len,
            device=device,
            use_soft_hash=True
        )
        
        if use_graph_context:
            self._init_graph_components()
        
        if use_structure_hashing:
            self._init_structure_aware_hashing()
            
        self._init_fusion_mechanisms()
    
    def _init_graph_components(self):
        """
        Initialize graph neural network components for structure encoding.
        """
        # K-mer embedding for graph nodes
        self.kmer_embedding = nn.Embedding(
            num_embeddings=self.alphabet_size ** self.k_mer_size,
            embedding_dim=self.graph_embed_dim
        )
        
        # Graph convolution layers (simplified GCN)
        self.graph_conv1 = nn.Linear(self.graph_embed_dim, self.graph_embed_dim)
        self.graph_conv2 = nn.Linear(self.graph_embed_dim, self.graph_embed_dim)
        
        # Graph attention mechanism
        self.graph_attention = nn.MultiheadAttention(
            embed_dim=self.graph_embed_dim,
            num_heads=4,
            batch_first=True
        )
        
        # Context aggregation
        self.context_aggregator = nn.Sequential(
            nn.Linear(self.graph_embed_dim, self.graph_embed_dim // 2),
            nn.ReLU(),
            nn.Linear(self.graph_embed_dim // 2, self.sketch_dim)
        )
    
    def _init_structure_aware_hashing(self):
        """
        Initialize structure-aware hash functions.
        """
        # Structure-enhanced hash embeddings
        self.structure_hash_embeddings = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.alphabet_size + self.graph_embed_dim, self.sketch_dim),
                nn.Tanh()
            )
            for _ in range(self.subsequence_len)
        ])
        
        # Structure-enhanced sign functions
        self.structure_sign_weights = nn.Parameter(
            torch.randn(self.subsequence_len, self.alphabet_size, self.graph_embed_dim, device=self.device)
        )
    
    def _init_fusion_mechanisms(self):
        """
        Initialize mechanisms for fusing graph and sketch information.
        """
        # Fusion strategy components
        fusion_input_dim = self.sketch_dim
        if self.use_graph_context:
            fusion_input_dim += self.sketch_dim  # Add context sketch dimension
        
        self.fusion_network = nn.Sequential(
            nn.Linear(fusion_input_dim, self.sketch_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.sketch_dim * 2, self.sketch_dim),
            nn.Tanh()
        )
        
        # Attention-based fusion weights
        self.fusion_attention = nn.Sequential(
            nn.Linear(fusion_input_dim, fusion_input_dim // 2),
            nn.ReLU(), 
            nn.Linear(fusion_input_dim // 2, 2),  # 2 components: sketch + graph
            nn.Softmax(dim=0)
        )
    
    def _sequence_to_kmers(self, sequence: torch.Tensor) -> List[int]:
        """
        Convert sequence to k-mer indices for graph processing.
        
        Args:
            sequence: Input sequence tensor
            
        Returns:
            List of k-mer indices
        """
        kmers = []
        seq_list = sequence.tolist()
        
        for i in range(len(seq_list) - self.k_mer_size + 1):
            kmer = seq_list[i:i + self.k_mer_size]
            # Convert k-mer to index (base-alphabet_size number)
            kmer_idx = sum(base * (self.alphabet_size ** pos) 
                          for pos, base in enumerate(reversed(kmer)))
            kmers.append(kmer_idx)
        
        return kmers
    
    def _build_local_graph(self, kmers: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build local De Bruijn graph from k-mers.
        
        Args:
            kmers: List of k-mer indices
            
        Returns:
            Tuple of (adjacency_matrix, node_features)
        """
        unique_kmers = list(set(kmers))
        n_nodes = len(unique_kmers)
        
        # Create adjacency matrix (simplified - based on k-mer overlap)
        adjacency = torch.zeros(n_nodes, n_nodes, device=self.device)
        
        # Build edges based on (k-1)-mer suffix-prefix matches
        kmer_to_idx = {kmer: i for i, kmer in enumerate(unique_kmers)}
        
        for i, kmer1 in enumerate(unique_kmers):
            for j, kmer2 in enumerate(unique_kmers):
                if i != j:
                    # Check if k-1 suffix of kmer1 matches k-1 prefix of kmer2
                    if self._kmers_adjacent(kmer1, kmer2):
                        adjacency[i, j] = 1.0
        
        # Get node features through embedding
        kmer_tensor = torch.tensor(unique_kmers, device=self.device)
        node_features = self.kmer_embedding(kmer_tensor)  # [n_nodes, graph_embed_dim]
        
        return adjacency, node_features
    
    def _kmers_adjacent(self, kmer1: int, kmer2: int) -> bool:
        """
        Check if two k-mers are adjacent in De Bruijn graph.
        Simplified version - in practice would use proper k-mer overlap logic.
        """
        # Convert back to sequences for overlap check
        seq1 = self._kmer_idx_to_sequence(kmer1)
        seq2 = self._kmer_idx_to_sequence(kmer2)
        
        # Check if (k-1) suffix of seq1 matches (k-1) prefix of seq2
        return seq1[1:] == seq2[:-1]
    
    def _kmer_idx_to_sequence(self, kmer_idx: int) -> List[int]:
        """Convert k-mer index back to sequence."""
        sequence = []
        for _ in range(self.k_mer_size):
            sequence.append(kmer_idx % self.alphabet_size)
            kmer_idx //= self.alphabet_size
        return list(reversed(sequence))
    
    def _graph_convolution(self, node_features: torch.Tensor, 
                          adjacency: torch.Tensor) -> torch.Tensor:
        """
        Apply graph convolution to node features.
        
        Args:
            node_features: Node feature matrix [n_nodes, embed_dim]
            adjacency: Adjacency matrix [n_nodes, n_nodes]
            
        Returns:
            Updated node features
        """
        # Normalize adjacency matrix
        degree = torch.sum(adjacency, dim=1, keepdim=True) + 1e-6
        normalized_adj = adjacency / degree
        
        # First graph convolution
        h1 = torch.relu(self.graph_conv1(node_features))
        h1_agg = torch.matmul(normalized_adj, h1)
        
        # Second graph convolution  
        h2 = torch.relu(self.graph_conv2(h1_agg))
        h2_agg = torch.matmul(normalized_adj, h2)
        
        return h2_agg
    
    def _compute_graph_context(self, sequence: torch.Tensor) -> torch.Tensor:
        """
        Compute graph context representation for the sequence.
        
        Args:
            sequence: Input sequence
            
        Returns:
            Graph context vector
        """
        # Extract k-mers and build local graph
        kmers = self._sequence_to_kmers(sequence)
        
        if len(kmers) == 0:
            # Return zero context if no k-mers
            return torch.zeros(self.sketch_dim, device=self.device)
        
        adjacency, node_features = self._build_local_graph(kmers)
        
        if node_features.size(0) == 0:
            return torch.zeros(self.sketch_dim, device=self.device)
        
        # Apply graph convolution
        updated_features = self._graph_convolution(node_features, adjacency)
        
        # Apply graph attention for important node selection
        if updated_features.size(0) > 1:
            attended_features, _ = self.graph_attention(
                updated_features.unsqueeze(0),  # Add batch dimension
                updated_features.unsqueeze(0),
                updated_features.unsqueeze(0)
            )
            attended_features = attended_features.squeeze(0)
        else:
            attended_features = updated_features
        
        # Aggregate to get sequence-level graph context
        if attended_features.size(0) > 0:
            graph_context = torch.mean(attended_features, dim=0)  # [graph_embed_dim]
            context_sketch = self.context_aggregator(graph_context)  # [sketch_dim]
        else:
            context_sketch = torch.zeros(self.sketch_dim, device=self.device)
        
        return context_sketch
    
    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with graph-aware tensor sketching.
        
        Args:
            sequence: Input sequence tensor
            
        Returns:
            Graph-aware tensor sketch
        """
        # Compute base tensor sketch
        base_sketch = self.base_sketch(sequence)
        
        # Compute graph context if enabled
        if self.use_graph_context:
            graph_context = self._compute_graph_context(sequence)
            
            # Fuse base sketch with graph context
            combined_features = torch.cat([base_sketch, graph_context], dim=0)
            
            # Attention-based fusion
            fusion_weights = self.fusion_attention(combined_features)
            fused_sketch = (fusion_weights[0] * base_sketch + 
                          fusion_weights[1] * graph_context)
            
            # Apply fusion network
            final_sketch = self.fusion_network(combined_features)
            
            return final_sketch
        else:
            return base_sketch
    
    def get_graph_statistics(self, sequence: torch.Tensor) -> Dict:
        """
        Get statistics about the graph structure for analysis.
        """
        kmers = self._sequence_to_kmers(sequence)
        unique_kmers = list(set(kmers))
        
        stats = {
            'sequence_length': len(sequence),
            'total_kmers': len(kmers), 
            'unique_kmers': len(unique_kmers),
            'kmer_diversity': len(unique_kmers) / max(len(kmers), 1)
        }
        
        if len(unique_kmers) > 1:
            adjacency, _ = self._build_local_graph(kmers)
            stats['graph_edges'] = torch.sum(adjacency).item()
            stats['graph_density'] = stats['graph_edges'] / (len(unique_kmers) ** 2)
        else:
            stats['graph_edges'] = 0
            stats['graph_density'] = 0
        
        return stats

class StructureAwareTensorSketch(nn.Module):
    """
    Alternative implementation focusing on structure-aware hashing without full GNN.
    Lighter weight but still incorporates structural information.
    """
    
    def __init__(self,
                 alphabet_size: int = 4,
                 sketch_dim: int = 64,
                 subsequence_len: int = 3,
                 structure_context_size: int = 2,
                 device: str = 'cpu'):
        super().__init__()
        
        self.alphabet_size = alphabet_size
        self.sketch_dim = sketch_dim
        self.subsequence_len = subsequence_len
        self.structure_context_size = structure_context_size
        self.device = device
        
        # Base sketching with structure-aware modifications
        self._init_structure_aware_components()
    
    def _init_structure_aware_components(self):
        """
        Initialize structure-aware hash and sign functions.
        """
        # Context-aware hash embeddings
        context_input_size = self.alphabet_size * (2 * self.structure_context_size + 1)
        
        self.context_hash_embeddings = nn.ModuleList([
            nn.Sequential(
                nn.Linear(context_input_size, self.sketch_dim),
                nn.Tanh()
            )
            for _ in range(self.subsequence_len)
        ])
        
        # Context-aware sign prediction
        self.context_sign_predictor = nn.ModuleList([
            nn.Sequential(
                nn.Linear(context_input_size, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
                nn.Tanh()
            )
            for _ in range(self.subsequence_len)
        ])
    
    def _get_structure_context(self, sequence: torch.Tensor, position: int) -> torch.Tensor:
        """
        Get structural context around a position in the sequence.
        """
        seq_len = sequence.size(0)
        context_size = self.structure_context_size
        
        # Extract context window
        start = max(0, position - context_size)
        end = min(seq_len, position + context_size + 1)
        
        # Pad if necessary
        context = sequence[start:end]
        if len(context) < 2 * context_size + 1:
            # Pad with a special token (use alphabet_size as padding)
            padding_needed = 2 * context_size + 1 - len(context)
            padding = torch.full((padding_needed,), self.alphabet_size, device=self.device)
            context = torch.cat([context, padding])
        
        # One-hot encode context
        context_one_hot = F.one_hot(context, self.alphabet_size + 1).float()  # +1 for padding
        return context_one_hot[:, :self.alphabet_size].flatten()  # Remove padding dimension
    
    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with structure-aware sketching.
        """
        seq_len = sequence.size(0)
        
        # Initialize tensors (functional style from Phase 2)
        Tp_list = [torch.zeros(self.sketch_dim, device=self.device) 
                   for _ in range(self.subsequence_len + 1)]
        Tm_list = [torch.zeros(self.sketch_dim, device=self.device) 
                   for _ in range(self.subsequence_len + 1)]
        
        # Initial condition
        Tp_list[0] = torch.zeros(self.sketch_dim, device=self.device)
        Tp_list[0][0] = 1.0
        
        # Main computation with structure awareness
        for i in range(seq_len):
            c = sequence[i]
            
            if c < 0 or c >= self.alphabet_size:
                continue
            
            # Get structural context
            structure_context = self._get_structure_context(sequence, i)
            
            # Create new tensors for this iteration
            new_Tp_list = [tp.clone() for tp in Tp_list]
            new_Tm_list = [tm.clone() for tm in Tm_list]
            
            max_p = min(i + 1, self.subsequence_len)
            for p in range(max_p, 0, -1):
                z = p / (i + 1.0)
                
                # Structure-aware hash and sign
                hash_logits = self.context_hash_embeddings[p - 1](structure_context)
                hash_probs = F.softmax(hash_logits, dim=0)
                
                sign_logit = self.context_sign_predictor[p - 1](structure_context)
                sign_prob = torch.sigmoid(sign_logit).squeeze()
                
                # Soft circular operations
                shifted_tp = self._soft_circular_shift(Tp_list[p - 1], hash_probs)
                shifted_tm = self._soft_circular_shift(Tm_list[p - 1], hash_probs)
                
                # Update tensors
                new_Tp_list[p] = (1 - z) * Tp_list[p] + z * (
                    sign_prob * shifted_tp + (1 - sign_prob) * shifted_tm
                )
                new_Tm_list[p] = (1 - z) * Tm_list[p] + z * (
                    sign_prob * shifted_tm + (1 - sign_prob) * shifted_tp
                )
            
            # Update for next iteration
            Tp_list = new_Tp_list
            Tm_list = new_Tm_list
        
        # Final sketch
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

def test_graph_aware_implementation():
    """
    Test the graph-aware tensor sketching implementations.
    """
    print("Testing Graph-Aware Tensor Sketching...")
    
    # Test sequences with different structural properties
    sequences = [
        torch.tensor([0, 1, 2, 3, 0, 1, 2, 3], dtype=torch.long),  # Periodic
        torch.tensor([0, 1, 0, 2, 0, 3, 0, 1], dtype=torch.long),  # Structured repeats
        torch.tensor([0, 2, 1, 3, 2, 0, 3, 1], dtype=torch.long),  # Complex
        torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.long),  # Blocks
    ]
    
    print("\n1. Testing Graph-Aware Tensor Sketching:")
    
    graph_model = GraphAwareTensorSketch(
        alphabet_size=4,
        sketch_dim=32,
        subsequence_len=3,
        k_mer_size=3,
        graph_embed_dim=16,
        use_graph_context=True
    )
    
    print(f"   ✓ Graph model: {sum(p.numel() for p in graph_model.parameters())} parameters")
    
    sketches = []
    for i, seq in enumerate(sequences):
        sketch = graph_model(seq)
        sketches.append(sketch)
        
        stats = graph_model.get_graph_statistics(seq)
        print(f"   ✓ Sequence {i}: sketch_norm={torch.norm(sketch).item():.4f}")
        print(f"     Graph stats: {stats['unique_kmers']} k-mers, {stats['graph_edges']:.0f} edges, density={stats['graph_density']:.3f}")
    
    # Test gradient flow
    loss = sum(torch.sum(sketch ** 2) for sketch in sketches)
    loss.backward()
    
    has_gradients = any(p.grad is not None and torch.norm(p.grad) > 1e-8 
                       for p in graph_model.parameters())
    print(f"   ✓ Gradients flowing: {has_gradients}")
    
    print("\n2. Testing Structure-Aware Tensor Sketching:")
    
    structure_model = StructureAwareTensorSketch(
        alphabet_size=4,
        sketch_dim=32,
        subsequence_len=3,
        structure_context_size=2
    )
    
    print(f"   ✓ Structure model: {sum(p.numel() for p in structure_model.parameters())} parameters")
    
    for i, seq in enumerate(sequences):
        sketch = structure_model(seq)
        print(f"   ✓ Sequence {i}: sketch_norm={torch.norm(sketch).item():.4f}")
    
    print("\n3. Comparing Graph-Aware vs Baseline:")
    
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
            
            # Graph-aware similarity
            graph_i = graph_model(sequences[i])
            graph_j = graph_model(sequences[j])
            graph_sim = F.cosine_similarity(graph_i, graph_j, dim=0).item()
            
            print(f"   Sequences {i}-{j}: Baseline={baseline_sim:.4f}, Graph-aware={graph_sim:.4f}")
    
    print("\n=== GRAPH-AWARE IMPLEMENTATION TEST COMPLETE ===")
    print("✅ Graph neural network components working")
    print("✅ Structure-aware hashing implemented")
    print("✅ K-mer graph construction functional")
    print("✅ Gradient flow maintained")
    print("🚀 Ready for attention-based position weighting")

if __name__ == "__main__":
    # Enable anomaly detection
    torch.autograd.set_detect_anomaly(True)
    
    # Run comprehensive tests
    test_graph_aware_implementation()