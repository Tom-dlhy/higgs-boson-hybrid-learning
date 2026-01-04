"""GA classifier with Active Learning integration."""

import random
from typing import List, Optional, Literal

import numpy as np
from deap import base, creator, gp, tools
from sklearn.base import BaseEstimator, ClassifierMixin

from src.evolutionary.primitives import create_primitive_set
from src.evolutionary.fitness import (
    evaluate_individual,
    predict_individual,
    get_population_predictions,
)
from src.active_learning.sampling import (
    uncertainty_sampling,
    diversity_sampling,
    random_sampling,
    hybrid_sampling,
)
from src.utils import set_global_seed, CheckpointManager


class GAActiveClassifier(BaseEstimator, ClassifierMixin):
    """GP classifier with Active Learning integration.
    
    This classifier combines Genetic Programming with Active Learning.
    At regular intervals during evolution, it queries the most informative
    samples from the pool and adds them to the training set.
    
    Attributes:
        population_size: Number of individuals in population.
        n_generations: Number of evolution generations.
        tournament_size: Size of tournament selection.
        crossover_prob: Probability of crossover.
        mutation_prob: Probability of mutation.
        max_depth: Maximum depth of GP trees.
        metric: Fitness metric ("f1" or "accuracy").
        al_frequency: Apply active learning every N generations.
        query_batch_size: Number of samples to query each AL round.
        sampling_strategy: Strategy for sample selection.
        random_state: Random seed for reproducibility.
        verbose: Whether to print progress.
    """
    
    def __init__(
        self,
        population_size: int = 100,
        n_generations: int = 20,
        tournament_size: int = 3,
        crossover_prob: float = 0.8,
        mutation_prob: float = 0.2,
        max_depth: int = 5,
        metric: str = "f1",
        al_frequency: int = 5,
        query_batch_size: int = 20,
        initial_pool_size: int = 100,
        sampling_strategy: Literal["uncertainty", "diversity", "random", "hybrid"] = "uncertainty",
        random_state: int = 42,
        verbose: bool = True,
        checkpoint_manager: Optional[CheckpointManager] = None,
    ):
        self.population_size = population_size
        self.n_generations = n_generations
        self.tournament_size = tournament_size
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.max_depth = max_depth
        self.metric = metric
        self.al_frequency = al_frequency
        self.query_batch_size = query_batch_size
        self.initial_pool_size = initial_pool_size
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state
        self.verbose = verbose
        self.checkpoint_manager = checkpoint_manager
        
        # Will be set during fit
        self.pset_: Optional[gp.PrimitiveSet] = None
        self.toolbox_: Optional[base.Toolbox] = None
        self.population_: Optional[List] = None
        self.best_individual_: Optional[gp.PrimitiveTree] = None
        self.hof_: Optional[tools.HallOfFame] = None
        self.classes_: np.ndarray = np.array([0, 1])
        self.n_features_in_: int = 0
        self.fitness_history_: List[dict] = []
        self.al_history_: List[dict] = []  # Track AL queries
    
    def _setup_deap(self) -> None:
        """Initialize DEAP creator and toolbox."""
        if hasattr(creator, "FitnessMax"):
            del creator.FitnessMax
        if hasattr(creator, "Individual"):
            del creator.Individual
            
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)
        
        self.toolbox_ = base.Toolbox()
        
        self.toolbox_.register(
            "expr", gp.genHalfAndHalf, pset=self.pset_, min_=1, max_=self.max_depth
        )
        self.toolbox_.register(
            "individual", tools.initIterate, creator.Individual, self.toolbox_.expr
        )
        self.toolbox_.register(
            "population", tools.initRepeat, list, self.toolbox_.individual
        )
        self.toolbox_.register(
            "select", tools.selTournament, tournsize=self.tournament_size
        )
        self.toolbox_.register("mate", gp.cxOnePoint)
        self.toolbox_.register(
            "expr_mut", gp.genFull, pset=self.pset_, min_=0, max_=2
        )
        self.toolbox_.register(
            "mutate", gp.mutUniform, expr=self.toolbox_.expr_mut, pset=self.pset_
        )
        
        # Bloat control
        import operator
        self.toolbox_.decorate(
            "mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=self.max_depth + 2)
        )
        self.toolbox_.decorate(
            "mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=self.max_depth + 2)
        )
    
    def _query_samples(
        self,
        X_pool: np.ndarray,
        y_pool: np.ndarray,
        generation: int,
    ) -> tuple:
        """Query informative samples from pool.
        
        Args:
            X_pool: Pool features.
            y_pool: Pool labels.
            generation: Current generation number.
            
        Returns:
            Tuple of (queried_X, queried_y, remaining_X, remaining_y).
        """
        if len(X_pool) == 0:
            return np.array([]), np.array([]), X_pool, y_pool
        
        n_query = min(self.query_batch_size, len(X_pool))
        
        if self.sampling_strategy == "uncertainty":
            # Get predictions from population
            pop_proba = get_population_predictions(
                self.population_, self.pset_, X_pool, return_proba=True
            )
            query_indices = uncertainty_sampling(pop_proba, n_query)
            
        elif self.sampling_strategy == "diversity":
            query_indices = diversity_sampling(X_pool, n_query, self.random_state + generation)
            
        elif self.sampling_strategy == "random":
            query_indices = random_sampling(len(X_pool), n_query, self.random_state + generation)
            
        elif self.sampling_strategy == "hybrid":
            pop_proba = get_population_predictions(
                self.population_, self.pset_, X_pool, return_proba=True
            )
            query_indices = hybrid_sampling(
                pop_proba, X_pool, n_query, 0.5, self.random_state + generation
            )
        else:
            raise ValueError(f"Unknown sampling strategy: {self.sampling_strategy}")
        
        # Get queried samples
        queried_X = X_pool[query_indices]
        queried_y = y_pool[query_indices]
        
        # Remove from pool
        remaining_mask = np.ones(len(X_pool), dtype=bool)
        remaining_mask[query_indices] = False
        remaining_X = X_pool[remaining_mask]
        remaining_y = y_pool[remaining_mask]
        
        return queried_X, queried_y, remaining_X, remaining_y
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> "GAActiveClassifier":
        """Fit the GA+AL classifier.
        
        The training data is split into initial training set and pool.
        During evolution, samples are queried from pool and added to training.
        
        Args:
            X: Training features (n_samples, n_features).
            y: Training labels (n_samples,).
            
        Returns:
            self
        """
        set_global_seed(self.random_state)
        
        self.n_features_in_ = X.shape[1]
        
        # Split into initial training set and pool
        indices = np.random.permutation(len(X))
        initial_size = min(self.initial_pool_size, len(X) // 2)
        
        initial_indices = indices[:initial_size]
        pool_indices = indices[initial_size:]
        
        X_train = X[initial_indices].copy()
        y_train = y[initial_indices].copy()
        X_pool = X[pool_indices].copy()
        y_pool = y[pool_indices].copy()
        
        if self.verbose:
            print(f"[GA+AL] Initial training: {len(X_train)}, Pool: {len(X_pool)}")
        
        # Create primitive set and setup DEAP
        self.pset_ = create_primitive_set(self.n_features_in_)
        self._setup_deap()
        
        # Create initial population
        self.population_ = self.toolbox_.population(n=self.population_size)
        self.hof_ = tools.HallOfFame(10)
        
        # Custom evolution loop with Active Learning
        self.fitness_history_ = []
        self.al_history_ = []
        
        # Initial evaluation
        fitnesses = [
            evaluate_individual(ind, self.pset_, X_train, y_train, self.metric)
            for ind in self.population_
        ]
        for ind, fit in zip(self.population_, fitnesses):
            ind.fitness.values = fit
        
        self.hof_.update(self.population_)
        
        if self.verbose:
            print("[GA+AL] Starting evolution...")
        
        for gen in range(1, self.n_generations + 1):
            # Selection
            offspring = self.toolbox_.select(self.population_, len(self.population_))
            offspring = [self.toolbox_.clone(ind) for ind in offspring]
            
            # Crossover
            for i in range(0, len(offspring) - 1, 2):
                if random.random() < self.crossover_prob:
                    offspring[i], offspring[i + 1] = self.toolbox_.mate(
                        offspring[i], offspring[i + 1]
                    )
                    del offspring[i].fitness.values
                    del offspring[i + 1].fitness.values
            
            # Mutation
            for i in range(len(offspring)):
                if random.random() < self.mutation_prob:
                    offspring[i], = self.toolbox_.mutate(offspring[i])
                    del offspring[i].fitness.values
            
            # Active Learning step
            if gen % self.al_frequency == 0 and len(X_pool) > 0:
                queried_X, queried_y, X_pool, y_pool = self._query_samples(
                    X_pool, y_pool, gen
                )
                
                if len(queried_X) > 0:
                    X_train = np.vstack([X_train, queried_X])
                    y_train = np.concatenate([y_train, queried_y])
                    
                    self.al_history_.append({
                        "generation": gen,
                        "queried": len(queried_X),
                        "total_train": len(X_train),
                        "remaining_pool": len(X_pool),
                    })
                    
                    if self.verbose:
                        print(f"[AL] Gen {gen}: Queried {len(queried_X)} samples, "
                              f"Training set: {len(X_train)}, Pool: {len(X_pool)}")
            
            # Evaluate offspring with updated training set
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = [
                evaluate_individual(ind, self.pset_, X_train, y_train, self.metric)
                for ind in invalid_ind
            ]
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            
            # Replace population
            self.population_[:] = offspring
            self.hof_.update(self.population_)
            
            # Record statistics
            fits = [ind.fitness.values[0] for ind in self.population_]
            self.fitness_history_.append({
                "gen": gen,
                "avg": np.mean(fits),
                "max": np.max(fits),
                "min": np.min(fits),
            })
            
            if self.verbose and gen % 5 == 0:
                print(f"[GA+AL] Gen {gen}: avg={np.mean(fits):.4f}, max={np.max(fits):.4f}")
        
        self.best_individual_ = self.hof_[0]
        
        if self.checkpoint_manager:
            fitness_vals = [ind.fitness.values[0] for ind in self.population_]
            self.checkpoint_manager.save_population(
                self.population_, fitness_vals, "ga_active_classifier"
            )
        
        if self.verbose:
            print(f"[GA+AL] Complete. Best fitness: {self.best_individual_.fitness.values[0]:.4f}")
            print(f"[GA+AL] Final training set size: {len(X_train)}")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        self._check_is_fitted()
        return predict_individual(
            self.best_individual_, self.pset_, X, return_proba=False
        ).astype(int)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities using sigmoid."""
        self._check_is_fitted()
        proba_class1 = predict_individual(
            self.best_individual_, self.pset_, X, return_proba=True
        )
        proba_class0 = 1 - proba_class1
        return np.column_stack([proba_class0, proba_class1])
    
    def get_population_predictions_proba(self, X: np.ndarray) -> np.ndarray:
        """Get probability predictions from entire population."""
        self._check_is_fitted()
        return get_population_predictions(
            self.population_, self.pset_, X, return_proba=True
        )
    
    def _check_is_fitted(self) -> None:
        """Check if the classifier has been fitted."""
        if self.best_individual_ is None:
            raise RuntimeError("Classifier not fitted. Call fit() first.")


if __name__ == "__main__":
    # Quick test
    from src.data_preprocessing import load_and_preprocess
    
    dataset = load_and_preprocess(sample_size=500)
    
    clf = GAActiveClassifier(
        population_size=20,
        n_generations=10,
        al_frequency=3,
        query_batch_size=10,
        initial_pool_size=50,
        sampling_strategy="uncertainty",
        verbose=True,
    )
    
    clf.fit(dataset.X_train, dataset.y_train)
    
    predictions = clf.predict(dataset.X_test)
    print(f"\nTest accuracy: {np.mean(predictions == dataset.y_test):.4f}")
    print(f"AL queries: {len(clf.al_history_)}")
