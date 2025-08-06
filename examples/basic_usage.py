#!/usr/bin/env python3
"""
Basic Usage Example - Learnable Tensor Sketching

This example demonstrates how to use the learnable tensor sketching framework
for genomic sequence similarity estimation.
"""

import sys
import os
import torch
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from learnable.working_phase2_sketch import WorkingLearnableTensorSketch
from baseline.pytorch_tensor_sketch import TensorSketchBaseline

def generate_sample_sequences(n_sequences=10, length=100, seed=42):
    """Generate sample genomic sequences for demonstration."""
    np.random.seed(seed)
    
    # DNA alphabet: A=0, T=1, G=2, C=3
    sequences = []
    for i in range(n_sequences):
        # Create some structure - periodic patterns for first half
        if i < n_sequences // 2:
            # Periodic sequences (more similar to each other)
            base_pattern = np.array([0, 1, 2, 3] * (length // 4 + 1))[:length]
            noise = np.random.randint(0, 4, length) 
            # Add 20% noise
            mask = np.random.random(length) < 0.2
            sequence = base_pattern.copy()
            sequence[mask] = noise[mask]
        else:
            # Random sequences
            sequence = np.random.randint(0, 4, length)
        
        sequences.append(sequence)
    
    return sequences

def main():
    """Main example demonstrating learnable tensor sketching."""
    print("🧬 Learnable Tensor Sketching - Basic Usage Example")
    print("=" * 55)
    
    # 1. Generate sample data
    print("\n📊 Generating sample genomic sequences...")
    sequences = generate_sample_sequences(n_sequences=6, length=200)
    print(f"Generated {len(sequences)} sequences of length {len(sequences[0])}")
    
    # Convert to torch tensors
    sequences_tensor = torch.tensor(sequences, dtype=torch.long)
    
    # 2. Initialize models
    print("\n🏗️ Initializing models...")
    
    # Baseline model
    baseline_sketch = TensorSketchBaseline(
        alphabet_size=4,
        sketch_dim=64,
        subsequence_len=3,
        device='cpu'
    )
    
    # Learnable model
    learnable_sketch = WorkingLearnableTensorSketch(
        alphabet_size=4,
        sketch_dim=64,
        subsequence_len=3,
        device='cpu'
    )
    
    print("✅ Models initialized")
    print(f"   Baseline parameters: {sum(p.numel() for p in baseline_sketch.parameters())}")
    print(f"   Learnable parameters: {sum(p.numel() for p in learnable_sketch.parameters())}")
    
    # 3. Generate sketches
    print("\n⚡ Computing tensor sketches...")
    
    # Baseline sketches
    with torch.no_grad():
        baseline_sketches = []
        for seq in sequences_tensor:
            sketch = baseline_sketch(seq)
            baseline_sketches.append(sketch.squeeze())
        baseline_sketches = torch.stack(baseline_sketches)
    
    # Learnable sketches (before training)
    with torch.no_grad():
        learnable_sketches = []
        for seq in sequences_tensor:
            sketch = learnable_sketch(seq)
            learnable_sketches.append(sketch.squeeze())
        learnable_sketches = torch.stack(learnable_sketches)
    
    print(f"   Sketch dimensions: {baseline_sketches.shape}")
    
    # 4. Compute similarities
    print("\n📏 Computing pairwise similarities...")
    
    def compute_similarities(sketches):
        """Compute pairwise cosine similarities."""
        # Normalize sketches
        sketches_norm = torch.nn.functional.normalize(sketches, p=2, dim=1)
        # Compute similarity matrix
        similarities = torch.mm(sketches_norm, sketches_norm.t())
        return similarities
    
    baseline_similarities = compute_similarities(baseline_sketches)
    learnable_similarities = compute_similarities(learnable_sketches)
    
    # 5. Display results
    print("\n📈 Similarity Analysis:")
    print("-" * 30)
    
    # Expected: sequences 0-2 should be more similar (periodic)
    # sequences 3-5 should be less similar (random)
    
    def analyze_similarities(sim_matrix, name):
        print(f"\n{name} Similarities:")
        print("Sequence pairs (0-indexed):")
        
        # Within-group similarities (periodic: 0-2, random: 3-5)
        periodic_pairs = [(0,1), (0,2), (1,2)]
        random_pairs = [(3,4), (3,5), (4,5)]
        cross_pairs = [(0,3), (1,4), (2,5)]
        
        periodic_sim = np.mean([sim_matrix[i,j].item() for i,j in periodic_pairs])
        random_sim = np.mean([sim_matrix[i,j].item() for i,j in random_pairs])
        cross_sim = np.mean([sim_matrix[i,j].item() for i,j in cross_pairs])
        
        print(f"  Periodic group similarity: {periodic_sim:.4f}")
        print(f"  Random group similarity:   {random_sim:.4f}")
        print(f"  Cross-group similarity:    {cross_sim:.4f}")
        print(f"  Discrimination ratio:      {periodic_sim/cross_sim:.2f}")
        
        return periodic_sim, random_sim, cross_sim
    
    baseline_stats = analyze_similarities(baseline_similarities, "Baseline")
    learnable_stats = analyze_similarities(learnable_similarities, "Learnable (untrained)")
    
    # 6. Quick training demonstration
    print("\n🎓 Quick training demonstration...")
    print("(This is a simplified training loop - see training_pipeline.py for full implementation)")
    
    # Create simple training data: pairs of similar sequences should have high similarity
    optimizer = torch.optim.Adam(learnable_sketch.parameters(), lr=0.01)
    
    for epoch in range(10):
        optimizer.zero_grad()
        
        # Forward pass
        sketches = []
        for seq in sequences_tensor:
            sketch = learnable_sketch(seq)
            sketches.append(sketch.squeeze())
        sketches = torch.stack(sketches)
        
        # Simple loss: encourage periodic sequences to be more similar
        similarities = compute_similarities(sketches)
        
        # Loss: maximize similarity within periodic group, minimize cross-group
        periodic_loss = -torch.mean(similarities[:3, :3]) # negative to maximize
        cross_loss = torch.mean(similarities[:3, 3:])     # positive to minimize
        loss = periodic_loss + cross_loss
        
        loss.backward()
        optimizer.step()
        
        if epoch % 3 == 0:
            print(f"  Epoch {epoch}: Loss = {loss.item():.4f}")
    
    # 7. Post-training analysis
    print("\n📊 Post-training analysis:")
    
    with torch.no_grad():
        trained_sketches = []
        for seq in sequences_tensor:
            sketch = learnable_sketch(seq)
            trained_sketches.append(sketch.squeeze())
        trained_sketches = torch.stack(trained_sketches)
    
    trained_similarities = compute_similarities(trained_sketches)
    trained_stats = analyze_similarities(trained_similarities, "Learnable (trained)")
    
    # 8. Summary
    print("\n🎯 Summary:")
    print("=" * 40)
    discrimination_improvement = (trained_stats[0]/trained_stats[2]) / (baseline_stats[0]/baseline_stats[2])
    print(f"Discrimination improvement: {discrimination_improvement:.2f}x")
    
    print("\n✨ This example demonstrates:")
    print("  • Basic usage of both baseline and learnable models")
    print("  • Similarity computation and analysis")
    print("  • Simple training loop (see training_pipeline.py for advanced training)")
    print("  • Performance comparison before and after learning")
    
    print(f"\n📚 Next steps:")
    print("  • See src/evaluation/ for comprehensive benchmarking")
    print("  • Review docs/installation.md for advanced setup")
    print("  • Explore advanced architectures in src/advanced/")

if __name__ == "__main__":
    main()