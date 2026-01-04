"""Utility functions for reproducibility, timing, and logging."""

import random
import time
from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable

import numpy as np


def set_global_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility across all libraries.
    
    Args:
        seed: The random seed to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    # DEAP uses random module internally


def get_timestamp() -> str:
    """Get current timestamp string for filenames."""
    return time.strftime("%Y%m%d_%H%M%S")


@contextmanager
def timer(name: str = "Operation"):
    """Context manager for timing code blocks.
    
    Args:
        name: Name to display in timing output.
        
    Yields:
        Dict that will contain 'elapsed' key after context exits.
        
    Example:
        with timer("Training") as t:
            model.fit(X, y)
        print(f"Took {t['elapsed']:.2f}s")
    """
    result = {}
    start = time.perf_counter()
    try:
        yield result
    finally:
        elapsed = time.perf_counter() - start
        result["elapsed"] = elapsed
        print(f"[TIMER] {name}: {elapsed:.2f}s")


def timed(func: Callable) -> Callable:
    """Decorator to time function execution.
    
    Args:
        func: Function to time.
        
    Returns:
        Wrapped function that prints timing info.
    """
    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"[TIMER] {func.__name__}: {elapsed:.2f}s")
        return result
    return wrapper


class CheckpointManager:
    """Manages saving and loading of intermediate results."""
    
    def __init__(self, results_dir: str | None = None, use_timestamps: bool = False):
        """Initialize checkpoint manager.
        
        Args:
            results_dir: Directory to save checkpoints. Uses 'results/' by default.
            use_timestamps: If True, add timestamps to filenames. If False, use fixed names.
        """
        from pathlib import Path
        self.results_dir = Path(results_dir) if results_dir else Path("results")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.use_timestamps = use_timestamps
    
    def _get_filename(self, name: str, extension: str) -> str:
        """Get filename with or without timestamp."""
        if self.use_timestamps:
            return f"{name}_{get_timestamp()}.{extension}"
        return f"{name}.{extension}"
    
    def save_metrics(self, metrics: dict, name: str) -> None:
        """Save metrics dictionary to CSV.
        
        Args:
            metrics: Dictionary of metric names to values.
            name: Name for the checkpoint file.
        """
        import pandas as pd
        filepath = self.results_dir / self._get_filename(name, "csv")
        pd.DataFrame([metrics]).to_csv(filepath, index=False)
        print(f"[CHECKPOINT] Saved metrics to {filepath}")
    
    def save_population(self, population: list, fitness_values: list, name: str) -> None:
        """Save population state for recovery.
        
        Args:
            population: List of individuals (as strings).
            fitness_values: Corresponding fitness values.
            name: Name for the checkpoint file.
        """
        import json
        filepath = self.results_dir / self._get_filename(f"{name}_population", "json")
        data = {
            "population": [str(ind) for ind in population],
            "fitness": fitness_values,
        }
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[CHECKPOINT] Saved population to {filepath}")
