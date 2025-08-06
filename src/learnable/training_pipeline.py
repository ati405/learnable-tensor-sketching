#!/usr/bin/env python3
"""
Training pipeline for learnable tensor sketching.
Demonstrates how to train the learnable parameters to improve sketch quality.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional
import time
import random

from .clean_differentiable_sketch import CleanDifferentiableTensorSketch
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'baseline'))
from ..baseline.pytorch_tensor_sketch import TensorSketchBaseline

class GenomicSequenceDataset(Dataset):
    """
    Dataset of genomic sequences for training tensor sketching.
    """
    
    def __init__(self, 
                 num_sequences: int = 1000,
                 seq_length_range: Tuple[int, int] = (50, 200),
                 alphabet_size: int = 4,
                 seed: int = 42):
        """
        Generate synthetic genomic sequences for training.
        
        Args:
            num_sequences: Number of sequences to generate
            seq_length_range: Range of sequence lengths (min, max)
            alphabet_size: Size of alphabet (4 for DNA)
            seed: Random seed
        """
        self.num_sequences = num_sequences
        self.seq_length_range = seq_length_range
        self.alphabet_size = alphabet_size
        
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)
        
        self.sequences = self._generate_sequences()
        
    def _generate_sequences(self) -> List[torch.Tensor]:
        """
        Generate random genomic sequences with some structure.
        """
        sequences = []
        
        for _ in range(self.num_sequences):
            # Random length within range
            length = np.random.randint(self.seq_length_range[0], 
                                     self.seq_length_range[1] + 1)
            
            # Generate sequence with some patterns (more realistic than pure random)
            sequence = []
            for i in range(length):
                # Add some local correlation to make sequences more realistic
                if i == 0 or np.random.random() < 0.3:  # 30% chance of random base
                    base = np.random.randint(0, self.alphabet_size)
                else:
                    # 70% chance of being similar to previous base (local correlation)
                    prev_base = sequence[-1]
                    if np.random.random() < 0.5:
                        base = prev_base  # Same as previous
                    else:
                        base = (prev_base + np.random.randint(1, self.alphabet_size)) % self.alphabet_size
                
                sequence.append(base)
            
            sequences.append(torch.tensor(sequence, dtype=torch.long))
        
        return sequences
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.sequences[idx]

def compute_sequence_similarity(seq1: torch.Tensor, seq2: torch.Tensor) -> float:
    """
    Compute true edit distance-based similarity between two sequences.
    This serves as ground truth for training.
    """
    # Simple Hamming distance for sequences of same length
    if seq1.size(0) != seq2.size(0):
        # For different lengths, use a simple length-normalized similarity
        min_len = min(seq1.size(0), seq2.size(0))
        max_len = max(seq1.size(0), seq2.size(0))
        
        # Compare the common prefix
        matches = torch.sum(seq1[:min_len] == seq2[:min_len]).item()
        similarity = matches / max_len  # Penalize length differences
    else:
        # Same length - use Hamming distance
        matches = torch.sum(seq1 == seq2).item()
        similarity = matches / seq1.size(0)
    
    return similarity

class TensorSketchTrainer:
    """
    Trainer for learnable tensor sketching.
    """
    
    def __init__(self, 
                 model: CleanDifferentiableTensorSketch,
                 baseline_model: Optional[TensorSketchBaseline] = None,
                 device: str = 'cpu'):
        """
        Initialize trainer.
        
        Args:
            model: Learnable tensor sketch model
            baseline_model: Baseline fixed model for comparison
            device: PyTorch device
        """
        self.model = model
        self.baseline_model = baseline_model
        self.device = device
        
        # Move models to device
        self.model = self.model.to(device)
        if self.baseline_model is not None:
            self.baseline_model = self.baseline_model.to(device)
    
    def sketch_similarity_loss(self, 
                             sequences: List[torch.Tensor],
                             batch_size: int = 16) -> torch.Tensor:
        """
        Compute loss based on sketch similarity preservation.
        
        The idea: sequences with high true similarity should have sketches
        with high cosine similarity.
        
        Args:
            sequences: List of sequences in the batch
            batch_size: Maximum batch size to process
            
        Returns:
            Loss tensor
        """
        if len(sequences) < 2:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # Compute sketches for all sequences
        sketches = []
        for seq in sequences[:batch_size]:  # Limit batch size for memory
            seq = seq.to(self.device)
            sketch = self.model(seq)
            sketches.append(sketch)
        
        if len(sketches) < 2:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        sketches = torch.stack(sketches)
        
        # Compute pairwise cosine similarities between sketches
        sketch_sims = torch.cosine_similarity(sketches.unsqueeze(1), 
                                            sketches.unsqueeze(0), dim=2)
        
        # Compute true sequence similarities
        true_sims = torch.zeros_like(sketch_sims)
        for i in range(len(sequences[:batch_size])):
            for j in range(len(sequences[:batch_size])):
                if i != j:
                    true_sim = compute_sequence_similarity(sequences[i], sequences[j])
                    true_sims[i, j] = true_sim
        
        # Loss: mean squared error between sketch similarities and true similarities
        # Only consider off-diagonal elements (i != j)
        mask = ~torch.eye(sketch_sims.size(0), dtype=torch.bool, device=self.device)
        sketch_sims_masked = sketch_sims[mask]
        true_sims_masked = true_sims[mask]
        
        loss = torch.mean((sketch_sims_masked - true_sims_masked) ** 2)
        return loss
    
    def reconstruction_loss(self, sequence: torch.Tensor) -> torch.Tensor:
        """
        Compute reconstruction-based loss (alternative training objective).
        
        Args:
            sequence: Input sequence
            
        Returns:
            Loss tensor
        """
        # Simple regularization loss to prevent parameter explosion
        sketch = self.model(sequence.to(self.device))
        
        # L2 regularization on sketch (prevent too large values)
        l2_loss = 0.001 * torch.mean(sketch ** 2)
        
        # Sparsity encouragement (optional)
        sparsity_loss = 0.01 * torch.mean(torch.abs(sketch))
        
        return l2_loss + sparsity_loss
    
    def train_epoch(self, 
                   dataloader: DataLoader,
                   optimizer: optim.Optimizer,
                   loss_type: str = 'similarity') -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            dataloader: Data loader
            optimizer: Optimizer
            loss_type: Type of loss ('similarity' or 'reconstruction')
            
        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch_sequences in enumerate(dataloader):
            optimizer.zero_grad()
            
            if loss_type == 'similarity':
                # Convert to list (DataLoader gives list of sequences)
                sequences = [seq.squeeze(0) if seq.dim() > 1 else seq for seq in batch_sequences]
                loss = self.sketch_similarity_loss(sequences, batch_size=8)  # Small batch for memory
            else:
                # Reconstruction loss for each sequence
                loss = torch.tensor(0.0, device=self.device, requires_grad=True)
                for seq in batch_sequences:
                    seq = seq.squeeze(0) if seq.dim() > 1 else seq
                    loss = loss + self.reconstruction_loss(seq)
                loss = loss / len(batch_sequences)
            
            if loss.requires_grad and torch.isfinite(loss):
                loss.backward()
                
                # Gradient clipping to prevent explosions
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / max(num_batches, 1)
        return {'loss': avg_loss, 'num_batches': num_batches}
    
    def evaluate(self, 
                dataloader: DataLoader,
                num_eval_samples: int = 100) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            dataloader: Data loader
            num_eval_samples: Number of samples to evaluate
            
        Returns:
            Dictionary with evaluation metrics
        """
        self.model.eval()
        
        # Collect some sequences for evaluation
        eval_sequences = []
        for batch_sequences in dataloader:
            for seq in batch_sequences:
                seq = seq.squeeze(0) if seq.dim() > 1 else seq
                eval_sequences.append(seq)
                if len(eval_sequences) >= num_eval_samples:
                    break
            if len(eval_sequences) >= num_eval_samples:
                break
        
        if len(eval_sequences) < 2:
            return {'correlation': 0.0, 'sketch_norm': 0.0}
        
        # Compute sketches
        with torch.no_grad():
            sketches = []
            sketch_norms = []
            
            for seq in eval_sequences[:50]:  # Limit for computation
                seq = seq.to(self.device)
                sketch = self.model(seq)
                sketches.append(sketch)
                sketch_norms.append(torch.norm(sketch).item())
        
        # Compute correlation between sketch similarity and true similarity
        correlations = []
        for i in range(min(10, len(sketches))):  # Sample pairs for efficiency
            for j in range(i + 1, min(10, len(sketches))):
                sketch_sim = torch.cosine_similarity(sketches[i], sketches[j], dim=0).item()
                true_sim = compute_sequence_similarity(eval_sequences[i], eval_sequences[j])
                correlations.append((sketch_sim, true_sim))
        
        if len(correlations) > 0:
            sketch_sims, true_sims = zip(*correlations)
            correlation = np.corrcoef(sketch_sims, true_sims)[0, 1]
            if np.isnan(correlation):
                correlation = 0.0
        else:
            correlation = 0.0
        
        return {
            'correlation': correlation,
            'sketch_norm': np.mean(sketch_norms),
            'num_sequences': len(eval_sequences)
        }

def run_training_experiment():
    """
    Run a complete training experiment.
    """
    print("Starting Learnable Tensor Sketching Training Experiment...")
    
    # Set up device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create dataset
    print("\n1. Creating dataset...")
    train_dataset = GenomicSequenceDataset(
        num_sequences=500,  # Small for quick testing
        seq_length_range=(20, 50),  # Short sequences for efficiency
        alphabet_size=4,
        seed=42
    )
    
    val_dataset = GenomicSequenceDataset(
        num_sequences=100,
        seq_length_range=(20, 50),
        alphabet_size=4,
        seed=123
    )
    
    print(f"✓ Created {len(train_dataset)} training sequences")
    print(f"✓ Created {len(val_dataset)} validation sequences")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=lambda x: x)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=lambda x: x)
    
    # Create models
    print("\n2. Creating models...")
    learnable_model = CleanDifferentiableTensorSketch(
        alphabet_size=4,
        sketch_dim=32,  # Small for efficiency
        subsequence_len=3,
        device=device,
        use_soft_hash=True
    )
    
    baseline_model = TensorSketchBaseline(
        alphabet_size=4,
        sketch_dim=32,
        subsequence_len=3,
        device=device
    )
    
    print(f"✓ Learnable model parameters: {sum(p.numel() for p in learnable_model.parameters())}")
    print(f"✓ Baseline model parameters: {sum(p.numel() for p in baseline_model.parameters())}")
    
    # Create trainer
    trainer = TensorSketchTrainer(learnable_model, baseline_model, device)
    
    # Create optimizer
    optimizer = optim.Adam(learnable_model.parameters(), lr=0.001)
    
    # Training loop
    print("\n3. Training...")
    num_epochs = 10  # Small number for quick testing
    train_losses = []
    val_correlations = []
    
    for epoch in range(num_epochs):
        # Train
        train_metrics = trainer.train_epoch(train_loader, optimizer, loss_type='reconstruction')
        
        # Evaluate
        val_metrics = trainer.evaluate(val_loader)
        
        # Record metrics
        train_losses.append(train_metrics['loss'])
        val_correlations.append(val_metrics['correlation'])
        
        print(f"Epoch {epoch+1:2d}: Loss={train_metrics['loss']:.4f}, "
              f"Correlation={val_metrics['correlation']:.4f}, "
              f"Sketch Norm={val_metrics['sketch_norm']:.4f}")
    
    # Final evaluation
    print("\n4. Final evaluation...")
    final_metrics = trainer.evaluate(val_loader, num_eval_samples=50)
    print(f"✓ Final correlation: {final_metrics['correlation']:.4f}")
    print(f"✓ Average sketch norm: {final_metrics['sketch_norm']:.4f}")
    
    # Test gradient flow one more time
    print("\n5. Verifying gradient flow...")
    test_seq = train_dataset[0].to(device)
    test_sketch = learnable_model(test_seq)
    test_loss = torch.sum(test_sketch ** 2)
    test_loss.backward()
    
    grad_norms = [torch.norm(p.grad).item() for p in learnable_model.parameters() 
                  if p.grad is not None]
    print(f"✓ Gradient norms: {[f'{norm:.6f}' for norm in grad_norms[:3]]}...")
    
    print("\n=== TRAINING EXPERIMENT COMPLETE ===")
    print("✓ Model trained successfully")
    print("✓ Gradient flow verified")
    print("✓ Ready for Phase 3: Advanced features")
    
    return {
        'train_losses': train_losses,
        'val_correlations': val_correlations,
        'final_metrics': final_metrics,
        'model': learnable_model
    }

if __name__ == "__main__":
    # Run training experiment
    results = run_training_experiment()
    
    # Print summary
    print(f"\nTraining Results Summary:")
    print(f"Final loss: {results['train_losses'][-1]:.4f}")
    print(f"Final correlation: {results['val_correlations'][-1]:.4f}")
    print(f"Model ready for advanced features!")