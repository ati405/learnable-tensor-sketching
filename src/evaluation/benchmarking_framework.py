#!/usr/bin/env python3
"""
Comprehensive Benchmarking Framework for Tensor Sketching Methods
Phase 4 Evaluation: Systematic comparison and statistical validation
"""

import torch
import torch.nn.functional as F
import numpy as np
import time
import psutil
import os
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pandas as pd
import sys

# Import all tensor sketching implementations
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'baseline'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'learnable'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'advanced'))

from pytorch_tensor_sketch import TensorSketchBaseline
from working_phase2_sketch import WorkingLearnableTensorSketch
from unified_advanced_sketch import AdvancedTensorSketchingSystem

@dataclass
class BenchmarkResult:
    """Data structure for storing benchmark results."""
    method: str
    dataset: str
    accuracy: float
    computation_time: float
    memory_usage: float
    sketch_quality: float
    similarity_correlation: float
    statistical_significance: Optional[float] = None

class PerformanceProfiler:
    """Profile computation time and memory usage."""
    
    def __init__(self):
        self.process = psutil.Process(os.getpid())
    
    def __enter__(self):
        self.start_time = time.time()
        self.start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        self.end_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        
        self.computation_time = self.end_time - self.start_time
        self.memory_usage = self.end_memory - self.start_memory

class TensorSketchBenchmark:
    """
    Comprehensive benchmarking framework for tensor sketching methods.
    
    Compares multiple implementations across various metrics:
    - Computational performance (time, memory)
    - Sketch quality and accuracy
    - Biological relevance and applications
    - Statistical significance of improvements
    """
    
    def __init__(self, 
                 alphabet_size: int = 4,
                 sketch_dim: int = 64,
                 device: str = 'cpu',
                 seed: int = 42):
        """
        Initialize benchmarking framework.
        
        Args:
            alphabet_size: Size of sequence alphabet
            sketch_dim: Dimension for tensor sketches
            device: PyTorch device
            seed: Random seed for reproducibility
        """
        self.alphabet_size = alphabet_size
        self.sketch_dim = sketch_dim
        self.device = device
        self.seed = seed
        
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Initialize all methods to be compared
        self.methods = self._initialize_methods()
        
        # Storage for results
        self.results = []
        self.detailed_results = defaultdict(list)
        
    def _initialize_methods(self) -> Dict[str, Any]:
        """Initialize all tensor sketching methods for comparison."""
        methods = {}
        
        # Baseline PyTorch implementation
        methods['baseline'] = TensorSketchBaseline(
            alphabet_size=self.alphabet_size,
            sketch_dim=self.sketch_dim,
            subsequence_len=3,
            device=self.device
        )
        
        # Phase 2: Working learnable implementation
        methods['phase2_learnable'] = WorkingLearnableTensorSketch(
            alphabet_size=self.alphabet_size,
            sketch_dim=self.sketch_dim,
            subsequence_len=3,
            device=self.device
        )
        
        # Phase 3: Advanced unified system (all features)
        methods['phase3_full'] = AdvancedTensorSketchingSystem(
            alphabet_size=self.alphabet_size,
            base_sketch_dim=self.sketch_dim,
            subsequence_len=3,
            device=self.device,
            use_multi_resolution=True,
            use_graph_awareness=True,
            use_attention=True,
            use_adaptive_dimension=True
        )
        
        # Phase 3: Individual features for ablation study
        methods['phase3_multi_res'] = AdvancedTensorSketchingSystem(
            alphabet_size=self.alphabet_size,
            base_sketch_dim=self.sketch_dim,
            subsequence_len=3,
            device=self.device,
            use_multi_resolution=True,
            use_graph_awareness=False,
            use_attention=False,
            use_adaptive_dimension=False
        )
        
        methods['phase3_graph_aware'] = AdvancedTensorSketchingSystem(
            alphabet_size=self.alphabet_size,
            base_sketch_dim=self.sketch_dim,
            subsequence_len=3,
            device=self.device,
            use_multi_resolution=False,
            use_graph_awareness=True,
            use_attention=False,
            use_adaptive_dimension=False
        )
        
        methods['phase3_attention'] = AdvancedTensorSketchingSystem(
            alphabet_size=self.alphabet_size,
            base_sketch_dim=self.sketch_dim,
            subsequence_len=3,
            device=self.device,
            use_multi_resolution=False,
            use_graph_awareness=False,
            use_attention=True,
            use_adaptive_dimension=False
        )
        
        return methods
    
    def generate_test_datasets(self) -> Dict[str, List[torch.Tensor]]:
        """Generate comprehensive test datasets for evaluation."""
        datasets = {}
        
        # Synthetic patterns for controlled evaluation
        datasets['periodic'] = [
            torch.tensor([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3], dtype=torch.long),
            torch.tensor([1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0], dtype=torch.long),
            torch.tensor([2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1], dtype=torch.long),
            torch.tensor([3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2], dtype=torch.long)
        ]
        
        datasets['structured'] = [
            torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, 0, 0, 1, 1], dtype=torch.long),
            torch.tensor([0, 1, 0, 1, 2, 3, 2, 3, 0, 1, 0, 1], dtype=torch.long),
            torch.tensor([0, 1, 2, 0, 1, 3, 0, 2, 1, 3, 2, 0], dtype=torch.long),
            torch.tensor([3, 2, 1, 0, 3, 2, 1, 0, 3, 2, 1, 0], dtype=torch.long)
        ]
        
        datasets['complex'] = [
            torch.tensor([0, 2, 1, 3, 2, 0, 3, 1, 0, 2, 3, 1], dtype=torch.long),
            torch.tensor([1, 3, 0, 2, 3, 1, 0, 2, 1, 3, 2, 0], dtype=torch.long),
            torch.tensor([2, 0, 3, 1, 0, 2, 1, 3, 2, 0, 1, 3], dtype=torch.long),
            torch.tensor([3, 1, 2, 0, 1, 3, 0, 2, 3, 1, 0, 2], dtype=torch.long)
        ]
        
        # Random sequences for robustness testing
        datasets['random'] = [
            torch.randint(0, self.alphabet_size, (12,), dtype=torch.long)
            for _ in range(4)
        ]
        
        # Variable length sequences for scalability testing
        datasets['scalability'] = [
            torch.randint(0, self.alphabet_size, (length,), dtype=torch.long)
            for length in [8, 16, 32, 64, 128]
        ]
        
        return datasets
    
    def evaluate_method(self, 
                       method_name: str, 
                       method: Any, 
                       sequences: List[torch.Tensor]) -> Dict[str, float]:
        """
        Evaluate a single method on a set of sequences.
        
        Args:
            method_name: Name of the method
            method: Method implementation
            sequences: List of test sequences
            
        Returns:
            Dictionary of evaluation metrics
        """
        sketches = []
        computation_times = []
        memory_usages = []
        
        # Compute sketches and profile performance
        for seq in sequences:
            with PerformanceProfiler() as profiler:
                if hasattr(method, '__call__'):
                    sketch = method(seq)
                else:
                    # For baseline method
                    sketch = method.forward(seq)
                
                sketches.append(sketch)
            
            computation_times.append(profiler.computation_time)
            memory_usages.append(max(profiler.memory_usage, 0))  # Ensure non-negative
        
        # Compute evaluation metrics
        metrics = {
            'avg_computation_time': np.mean(computation_times),
            'std_computation_time': np.std(computation_times),
            'avg_memory_usage': np.mean(memory_usages),
            'avg_sketch_norm': np.mean([torch.norm(sketch).item() for sketch in sketches]),
            'sketch_variance': np.var([torch.norm(sketch).item() for sketch in sketches])
        }
        
        # Compute pairwise similarities for quality assessment
        similarities = []
        for i in range(len(sketches)):
            for j in range(i + 1, len(sketches)):
                sim = F.cosine_similarity(sketches[i], sketches[j], dim=0).item()
                similarities.append(sim)
        
        metrics['avg_similarity'] = np.mean(similarities) if similarities else 0.0
        metrics['similarity_variance'] = np.var(similarities) if similarities else 0.0
        
        # Additional quality metrics
        metrics['sketch_quality'] = self._compute_sketch_quality(sketches)
        
        return metrics
    
    def _compute_sketch_quality(self, sketches: List[torch.Tensor]) -> float:
        """
        Compute overall sketch quality metric.
        
        Quality is measured as the ability to distinguish between
        different sequences while maintaining stability.
        """
        if len(sketches) < 2:
            return 0.0
        
        # Compute pairwise distances
        distances = []
        for i in range(len(sketches)):
            for j in range(i + 1, len(sketches)):
                dist = torch.norm(sketches[i] - sketches[j]).item()
                distances.append(dist)
        
        # Quality is measured as the variance in distances
        # Higher variance = better discrimination
        quality = np.var(distances) if len(distances) > 1 else 0.0
        return quality
    
    def run_comprehensive_benchmark(self) -> pd.DataFrame:
        """
        Run comprehensive benchmark across all methods and datasets.
        
        Returns:
            DataFrame with detailed benchmark results
        """
        print("Running Comprehensive Tensor Sketching Benchmark...")
        print(f"Methods: {list(self.methods.keys())}")
        
        datasets = self.generate_test_datasets()
        print(f"Datasets: {list(datasets.keys())}")
        
        benchmark_data = []
        
        for dataset_name, sequences in datasets.items():
            print(f"\n--- Evaluating on {dataset_name} dataset ---")
            
            for method_name, method in self.methods.items():
                print(f"  Testing {method_name}...")
                
                try:
                    metrics = self.evaluate_method(method_name, method, sequences)
                    
                    # Store results
                    result = {
                        'method': method_name,
                        'dataset': dataset_name,
                        'num_sequences': len(sequences),
                        **metrics
                    }
                    benchmark_data.append(result)
                    
                    print(f"    ✓ Avg time: {metrics['avg_computation_time']:.4f}s")
                    print(f"    ✓ Sketch norm: {metrics['avg_sketch_norm']:.4f}")
                    print(f"    ✓ Quality: {metrics['sketch_quality']:.4f}")
                    
                except Exception as e:
                    print(f"    ✗ Error: {e}")
                    continue
        
        # Convert to DataFrame for easy analysis
        results_df = pd.DataFrame(benchmark_data)
        return results_df
    
    def compute_statistical_significance(self, 
                                       results_df: pd.DataFrame,
                                       baseline_method: str = 'baseline') -> pd.DataFrame:
        """
        Compute statistical significance of improvements over baseline.
        
        Args:
            results_df: Benchmark results DataFrame
            baseline_method: Method to use as baseline for comparison
            
        Returns:
            DataFrame with statistical significance results
        """
        print(f"\nComputing statistical significance vs {baseline_method}...")
        
        significance_results = []
        
        # Get baseline performance for each dataset
        baseline_data = results_df[results_df['method'] == baseline_method]
        
        for dataset in results_df['dataset'].unique():
            baseline_dataset = baseline_data[baseline_data['dataset'] == dataset]
            if baseline_dataset.empty:
                continue
                
            baseline_quality = baseline_dataset['sketch_quality'].iloc[0]
            baseline_time = baseline_dataset['avg_computation_time'].iloc[0]
            
            # Compare all other methods
            for method in results_df['method'].unique():
                if method == baseline_method:
                    continue
                    
                method_data = results_df[
                    (results_df['method'] == method) & 
                    (results_df['dataset'] == dataset)
                ]
                
                if method_data.empty:
                    continue
                
                method_quality = method_data['sketch_quality'].iloc[0]
                method_time = method_data['avg_computation_time'].iloc[0]
                
                # Compute improvements
                quality_improvement = (method_quality - baseline_quality) / baseline_quality * 100
                time_ratio = method_time / baseline_time
                
                significance_results.append({
                    'method': method,
                    'dataset': dataset,
                    'baseline_method': baseline_method,
                    'quality_improvement_percent': quality_improvement,
                    'time_ratio': time_ratio,
                    'baseline_quality': baseline_quality,
                    'method_quality': method_quality
                })
        
        significance_df = pd.DataFrame(significance_results)
        return significance_df
    
    def generate_performance_plots(self, 
                                 results_df: pd.DataFrame,
                                 output_dir: str = './benchmark_results'):
        """
        Generate publication-quality performance plots.
        
        Args:
            results_df: Benchmark results DataFrame
            output_dir: Directory to save plots
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Set publication style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Plot 1: Performance comparison across methods
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Sketch quality comparison
        sns.boxplot(data=results_df, x='method', y='sketch_quality', ax=axes[0,0])
        axes[0,0].set_title('Sketch Quality Comparison')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # Computation time comparison
        sns.boxplot(data=results_df, x='method', y='avg_computation_time', ax=axes[0,1])
        axes[0,1].set_title('Computation Time Comparison')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # Sketch norm comparison
        sns.boxplot(data=results_df, x='method', y='avg_sketch_norm', ax=axes[1,0])
        axes[1,0].set_title('Sketch Norm Comparison')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # Memory usage comparison
        sns.boxplot(data=results_df, x='method', y='avg_memory_usage', ax=axes[1,1])
        axes[1,1].set_title('Memory Usage Comparison')
        axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot 2: Performance by dataset
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Quality by dataset
        sns.barplot(data=results_df, x='dataset', y='sketch_quality', hue='method', ax=axes[0,0])
        axes[0,0].set_title('Sketch Quality by Dataset')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # Time by dataset
        sns.barplot(data=results_df, x='dataset', y='avg_computation_time', hue='method', ax=axes[0,1])
        axes[0,1].set_title('Computation Time by Dataset')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # Similarity by dataset
        sns.barplot(data=results_df, x='dataset', y='avg_similarity', hue='method', ax=axes[1,0])
        axes[1,0].set_title('Average Similarity by Dataset')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # Sketch variance by dataset
        sns.barplot(data=results_df, x='dataset', y='sketch_variance', hue='method', ax=axes[1,1])
        axes[1,1].set_title('Sketch Variance by Dataset')
        axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/performance_by_dataset.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Performance plots saved to {output_dir}/")
    
    def generate_summary_report(self, 
                              results_df: pd.DataFrame,
                              significance_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate comprehensive summary report.
        
        Args:
            results_df: Benchmark results DataFrame
            significance_df: Statistical significance DataFrame
            
        Returns:
            Dictionary with summary statistics
        """
        summary = {
            'total_methods_tested': len(results_df['method'].unique()),
            'total_datasets': len(results_df['dataset'].unique()),
            'total_evaluations': len(results_df),
        }
        
        # Best performing method by metric
        best_quality = results_df.loc[results_df['sketch_quality'].idxmax()]
        best_speed = results_df.loc[results_df['avg_computation_time'].idxmin()]
        best_efficiency = results_df.loc[results_df['avg_memory_usage'].idxmin()]
        
        summary['best_quality'] = {
            'method': best_quality['method'],
            'dataset': best_quality['dataset'],
            'quality': best_quality['sketch_quality']
        }
        
        summary['best_speed'] = {
            'method': best_speed['method'],
            'dataset': best_speed['dataset'],
            'time': best_speed['avg_computation_time']
        }
        
        summary['best_efficiency'] = {
            'method': best_efficiency['method'],
            'dataset': best_efficiency['dataset'],
            'memory': best_efficiency['avg_memory_usage']
        }
        
        # Average improvements
        if not significance_df.empty:
            avg_improvement = significance_df.groupby('method')['quality_improvement_percent'].mean()
            summary['average_improvements'] = avg_improvement.to_dict()
            
            max_improvement = significance_df.groupby('method')['quality_improvement_percent'].max()
            summary['maximum_improvements'] = max_improvement.to_dict()
        
        return summary

def run_full_benchmark():
    """Run the complete benchmarking suite."""
    print("🚀 Starting Comprehensive Tensor Sketching Benchmark")
    print("=" * 60)
    
    # Initialize benchmark
    benchmark = TensorSketchBenchmark(
        alphabet_size=4,
        sketch_dim=32,  # Smaller for faster testing
        device='cpu',
        seed=42
    )
    
    # Run comprehensive evaluation
    results_df = benchmark.run_comprehensive_benchmark()
    
    # Compute statistical significance
    significance_df = benchmark.compute_statistical_significance(results_df)
    
    # Generate performance plots
    benchmark.generate_performance_plots(results_df)
    
    # Generate summary report
    summary = benchmark.generate_summary_report(results_df, significance_df)
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 BENCHMARK SUMMARY REPORT")
    print("=" * 60)
    
    print(f"Total Methods Tested: {summary['total_methods_tested']}")
    print(f"Total Datasets: {summary['total_datasets']}")
    print(f"Total Evaluations: {summary['total_evaluations']}")
    
    print(f"\n🏆 Best Quality: {summary['best_quality']['method']} "
          f"(Quality: {summary['best_quality']['quality']:.4f})")
    
    print(f"⚡ Fastest Method: {summary['best_speed']['method']} "
          f"(Time: {summary['best_speed']['time']:.4f}s)")
    
    print(f"💾 Most Efficient: {summary['best_efficiency']['method']} "
          f"(Memory: {summary['best_efficiency']['memory']:.2f}MB)")
    
    if 'average_improvements' in summary:
        print(f"\n📈 Average Quality Improvements:")
        for method, improvement in summary['average_improvements'].items():
            print(f"  {method}: {improvement:+.1f}%")
    
    # Save detailed results
    results_df.to_csv('./benchmark_results/detailed_results.csv', index=False)
    significance_df.to_csv('./benchmark_results/significance_results.csv', index=False)
    
    print(f"\n📁 Detailed results saved to ./benchmark_results/")
    print("✅ Comprehensive benchmark completed successfully!")
    
    return results_df, significance_df, summary

if __name__ == "__main__":
    # Run the complete benchmark
    results_df, significance_df, summary = run_full_benchmark()