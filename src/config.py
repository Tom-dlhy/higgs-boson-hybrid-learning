"""Configuration settings for the Higgs Boson Hybrid Learning project."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


@dataclass
class Config:
    """Central configuration for all experiments."""

    # Paths
    data_path: Path = field(default_factory=lambda: Path("data/atlas-higgs.csv"))
    results_dir: Path = field(default_factory=lambda: Path("results"))
    
    # Data settings
    sample_size: int | None = None  # None = full dataset, int = subsample
    test_size: float = 0.2
    random_state: int = 42
    
    # Preprocessing
    add_missing_indicators: bool = True
    normalize: bool = True
    
    # GP Settings (production-tuned for diversity)
    population_size: int = 150
    n_generations: int = 40
    tournament_size: int = 3
    crossover_prob: float = 0.8
    mutation_prob: float = 0.2
    max_tree_depth: int = 5
    bloat_penalty: float = 0.001  # Penalize large trees for diversity
    
    # Active Learning Settings (production-tuned)
    initial_pool_size: int = 2500  # Larger initial training set
    query_batch_size: int = 400   # More samples per AL round
    al_frequency: int = 5  # Apply AL every N generations
    sampling_strategy: Literal["uncertainty", "diversity", "random"] = "uncertainty"
    
    # Ensemble Settings
    ensemble_size: int = 10  # Top-k individuals for ensemble
    voting_strategy: Literal["hard", "soft", "weighted"] = "soft"
    use_similarity_filter: bool = True  # Filter similar individuals in ensemble
    
    # Experiment modes
    quick_test: bool = False
    
    def __post_init__(self):
        """Validate and set up directories."""
        self.results_dir.mkdir(parents=True, exist_ok=True)
        if self.quick_test:
            self.sample_size = 1000
            self.population_size = 20
            self.n_generations = 5
    
    @classmethod
    def for_quick_test(cls) -> "Config":
        """Factory for quick testing configuration."""
        return cls(quick_test=True)
    
    @classmethod
    def for_experiment(cls, sample_size: int = 50000) -> "Config":
        """Factory for full experiment configuration."""
        return cls(sample_size=sample_size)
