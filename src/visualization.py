"""Visualization utilities for experiment results."""

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

from src.metrics import ClassificationMetrics


def set_style() -> None:
    """Set matplotlib style for consistent plots."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['font.size'] = 12


def plot_metrics_comparison(
    results: Dict[str, ClassificationMetrics],
    save_path: Optional[Path] = None,
) -> None:
    """Plot bar chart comparing metrics across approaches.
    
    Args:
        results: Dictionary mapping approach names to metrics.
        save_path: Optional path to save the plot.
    """
    set_style()
    
    approaches = list(results.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    
    x = np.arange(len(approaches))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6']
    
    for i, metric in enumerate(metrics):
        values = [getattr(results[a], metric) for a in approaches]
        bars = ax.bar(x + i * width, values, width, label=metric.capitalize(), color=colors[i])
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            ax.annotate(
                f'{val:.3f}',
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom',
                fontsize=9,
            )
    
    ax.set_xlabel('Approach')
    ax.set_ylabel('Score')
    ax.set_title('Performance Comparison: GA vs GA+AL vs GA+AL+EL')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(approaches)
    ax.legend(loc='lower right')
    ax.set_ylim(0, 1.15)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[VIZ] Saved metrics comparison to {save_path}")
    
    plt.close()


def plot_training_time_comparison(
    results: Dict[str, ClassificationMetrics],
    save_path: Optional[Path] = None,
) -> None:
    """Plot training time comparison.
    
    Args:
        results: Dictionary mapping approach names to metrics.
        save_path: Optional path to save the plot.
    """
    set_style()
    
    approaches = list(results.keys())
    times = [results[a].training_time for a in approaches]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    colors = ['#3498db', '#2ecc71', '#9b59b6']
    bars = ax.bar(approaches, times, color=colors[:len(approaches)])
    
    # Add value labels
    for bar, t in zip(bars, times):
        ax.annotate(
            f'{t:.1f}s',
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 3),
            textcoords="offset points",
            ha='center', va='bottom',
            fontsize=11, fontweight='bold',
        )
    
    ax.set_xlabel('Approach')
    ax.set_ylabel('Training Time (seconds)')
    ax.set_title('Training Time Comparison')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[VIZ] Saved training time comparison to {save_path}")
    
    plt.close()


def plot_fitness_evolution(
    fitness_histories: Dict[str, List[dict]],
    save_path: Optional[Path] = None,
) -> None:
    """Plot fitness evolution over generations.
    
    Args:
        fitness_histories: Dictionary mapping approach names to fitness history.
        save_path: Optional path to save the plot.
    """
    set_style()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Different colors and line styles for each approach
    styles = {
        'GA': {'color': '#3498db', 'linestyle': '-', 'marker': 'o', 'markersize': 4},
        'GA+AL': {'color': '#2ecc71', 'linestyle': '-', 'marker': 's', 'markersize': 4},
        'GA+AL+EL': {'color': '#9b59b6', 'linestyle': ':', 'marker': '^', 'markersize': 5},
    }
    
    for name, history in fitness_histories.items():
        if not history:
            continue
        
        generations = [h['gen'] for h in history]
        max_fitness = [h['max'] for h in history]
        avg_fitness = [h['avg'] for h in history]
        
        style = styles.get(name, {'color': '#333333', 'linestyle': '-', 'marker': 'o', 'markersize': 4})
        
        ax.plot(generations, max_fitness, 
                linestyle=style['linestyle'], 
                marker=style['marker'],
                markersize=style['markersize'],
                label=f'{name} (max)', 
                color=style['color'], 
                linewidth=2)
        ax.plot(generations, avg_fitness, 
                linestyle='--', 
                marker=style['marker'],
                markersize=style['markersize'] - 1,
                label=f'{name} (avg)', 
                color=style['color'], 
                alpha=0.7)
    
    ax.set_xlabel('Generation')
    ax.set_ylabel('Fitness (F1-Score)')
    ax.set_title('Fitness Evolution Over Generations')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[VIZ] Saved fitness evolution to {save_path}")
    
    plt.close()


def plot_confusion_matrices(
    results: Dict[str, ClassificationMetrics],
    save_path: Optional[Path] = None,
) -> None:
    """Plot confusion matrices for all approaches.
    
    Args:
        results: Dictionary mapping approach names to metrics.
        save_path: Optional path to save the plot.
    """
    set_style()
    
    n_approaches = len(results)
    fig, axes = plt.subplots(1, n_approaches, figsize=(5 * n_approaches, 4))
    
    if n_approaches == 1:
        axes = [axes]
    
    for ax, (name, metrics) in zip(axes, results.items()):
        cm = metrics.confusion_matrix
        
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax, shrink=0.6)
        
        # Labels
        ax.set(
            xticks=[0, 1],
            yticks=[0, 1],
            xticklabels=['Background', 'Signal'],
            yticklabels=['Background', 'Signal'],
            ylabel='True label',
            xlabel='Predicted label',
            title=name,
        )
        
        # Add text annotations
        thresh = cm.max() / 2
        for i in range(2):
            for j in range(2):
                ax.text(
                    j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=14,
                )
    
    plt.suptitle('Confusion Matrices', fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[VIZ] Saved confusion matrices to {save_path}")
    
    plt.close()


def generate_all_plots(
    results: Dict[str, ClassificationMetrics],
    fitness_histories: Dict[str, List[dict]],
    output_dir: Path,
) -> List[Path]:
    """Generate all visualization plots.
    
    Args:
        results: Dictionary mapping approach names to metrics.
        fitness_histories: Fitness evolution data.
        output_dir: Directory to save plots.
        
    Returns:
        List of saved file paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    saved_paths = []
    
    # Metrics comparison
    path = output_dir / "metrics_comparison.png"
    plot_metrics_comparison(results, path)
    saved_paths.append(path)
    
    # Training time
    path = output_dir / "training_time.png"
    plot_training_time_comparison(results, path)
    saved_paths.append(path)
    
    # Fitness evolution
    path = output_dir / "fitness_evolution.png"
    plot_fitness_evolution(fitness_histories, path)
    saved_paths.append(path)
    
    # Confusion matrices
    path = output_dir / "confusion_matrices.png"
    plot_confusion_matrices(results, path)
    saved_paths.append(path)
    
    return saved_paths
