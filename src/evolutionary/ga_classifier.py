"""GP-based classifier with scikit-learn compatible interface."""

from typing import List, Optional

import numpy as np
from deap import base, creator, gp, tools, algorithms
from sklearn.base import BaseEstimator, ClassifierMixin

from src.evolutionary.primitives import create_primitive_set
from src.evolutionary.fitness import (
    evaluate_individual,
    predict_individual,
    get_population_predictions
)
from src.utils import set_global_seed, CheckpointManager
import operator


class GPClassifier(BaseEstimator, ClassifierMixin):
    """Genetic Programming classifier compatible with scikit-learn.
    
    This classifier evolves mathematical expression trees to classify
    binary targets. It supports:
    - predict(): Binary class predictions
    - predict_proba(): Probability estimates via sigmoid
    - Access to final population for ensemble methods
    
    Attributes:
        population_size: Number of individuals in population.
        n_generations: Number of evolution generations.
        tournament_size: Size of tournament selection.
        crossover_prob: Probability of crossover.
        mutation_prob: Probability of mutation.
        max_depth: Maximum depth of GP trees.
        metric: Fitness metric ("f1" or "accuracy").
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
        self.random_state = random_state
        self.verbose = verbose
        self.checkpoint_manager = checkpoint_manager
        
        # Will be set during fit
        self.pset_: Optional[gp.PrimitiveSet] = None
        self.toolbox_: Optional[base.Toolbox] = None
        self.population_: Optional[List] = None
        self.best_individual_: Optional[gp.PrimitiveTree] = None
        self.logbook_: Optional[tools.Logbook] = None
        self.hof_: Optional[tools.HallOfFame] = None
        self.classes_: np.ndarray = np.array([0, 1])
        self.n_features_in_: int = 0
        self.feature_names_: List[str] = []
        self.fitness_history_: List[dict] = []
    
    def _setup_deap(self) -> None:
        """Initialize DEAP creator and toolbox."""
        # Clear any existing creator types (for re-fitting)
        if hasattr(creator, "FitnessMax"):
            del creator.FitnessMax
        if hasattr(creator, "Individual"):
            del creator.Individual
            
        # Define fitness (maximize F1 or accuracy)
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)
        
        # Create toolbox
        self.toolbox_ = base.Toolbox()
        
        # Tree generation
        self.toolbox_.register(
            "expr",
            gp.genHalfAndHalf,
            pset=self.pset_,
            min_=1,
            max_=self.max_depth,
        )
        self.toolbox_.register(
            "individual",
            tools.initIterate,
            creator.Individual,
            self.toolbox_.expr,
        )
        self.toolbox_.register(
            "population",
            tools.initRepeat,
            list,
            self.toolbox_.individual,
        )
        
        # Genetic operators
        self.toolbox_.register(
            "select",
            tools.selTournament,
            tournsize=self.tournament_size,
        )
        self.toolbox_.register(
            "mate",
            gp.cxOnePoint,
        )
        self.toolbox_.register(
            "expr_mut",
            gp.genFull,
            pset=self.pset_,
            min_=0,
            max_=2,
        )
        self.toolbox_.register(
            "mutate",
            gp.mutUniform,
            expr=self.toolbox_.expr_mut,
            pset=self.pset_,
        )
        
        # Bloat control
        self.toolbox_.decorate(
            "mate",
            gp.staticLimit(key=operator.attrgetter("height"), max_value=self.max_depth + 2),
        )
        self.toolbox_.decorate(
            "mutate",
            gp.staticLimit(key=operator.attrgetter("height"), max_value=self.max_depth + 2),
        )
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> "GPClassifier":
        """Fit the GP classifier.
        
        Args:
            X: Training features (n_samples, n_features).
            y: Training labels (n_samples,).
            
        Returns:
            self
        """
        # Set seeds for reproducibility
        set_global_seed(self.random_state)
        
        # Store feature info
        self.n_features_in_ = X.shape[1]
        self.feature_names_ = [f"x{i}" for i in range(self.n_features_in_)]
        
        # Create primitive set
        self.pset_ = create_primitive_set(self.n_features_in_, self.feature_names_)
        
        # Setup DEAP toolbox
        self._setup_deap()
        
        # Register fitness function with current data
        self.toolbox_.register(
            "evaluate",
            lambda ind: evaluate_individual(ind, self.pset_, X, y, self.metric),
        )
        
        # Create initial population
        self.population_ = self.toolbox_.population(n=self.population_size)
        
        # Hall of Fame (best individuals)
        self.hof_ = tools.HallOfFame(10)
        
        # Statistics
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("min", np.min)
        stats.register("max", np.max)
        stats.register("std", np.std)
        
        # Evolution
        if self.verbose:
            print(f"[GP] Starting evolution: {self.population_size} individuals, {self.n_generations} generations")
        
        self.population_, self.logbook_ = algorithms.eaSimple(
            self.population_,
            self.toolbox_,
            cxpb=self.crossover_prob,
            mutpb=self.mutation_prob,
            ngen=self.n_generations,
            stats=stats,
            halloffame=self.hof_,
            verbose=self.verbose,
        )
        
        # Store best individual
        self.best_individual_ = self.hof_[0]
        
        # Store fitness history
        self.fitness_history_ = [
            {"gen": record["gen"], "avg": record["avg"], "max": record["max"]}
            for record in self.logbook_
        ]
        
        # Checkpoint
        if self.checkpoint_manager:
            fitness_vals = [ind.fitness.values[0] for ind in self.population_]
            self.checkpoint_manager.save_population(
                self.population_, fitness_vals, "gp_classifier"
            )
        
        if self.verbose:
            best_fitness = self.best_individual_.fitness.values[0]
            print(f"[GP] Evolution complete. Best fitness: {best_fitness:.4f}")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels.
        
        Args:
            X: Features (n_samples, n_features).
            
        Returns:
            Predicted labels (n_samples,).
        """
        self._check_is_fitted()
        return predict_individual(
            self.best_individual_, self.pset_, X, return_proba=False
        ).astype(int)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities using sigmoid transformation.
        
        This method is required for ModAL compatibility.
        
        Args:
            X: Features (n_samples, n_features).
            
        Returns:
            Probability matrix (n_samples, 2) for classes [0, 1].
        """
        self._check_is_fitted()
        proba_class1 = predict_individual(
            self.best_individual_, self.pset_, X, return_proba=True
        )
        proba_class0 = 1 - proba_class1
        return np.column_stack([proba_class0, proba_class1])
    
    def get_population_predictions_proba(self, X: np.ndarray) -> np.ndarray:
        """Get probability predictions from entire population.
        
        Useful for ensemble methods and uncertainty estimation.
        
        Args:
            X: Features (n_samples, n_features).
            
        Returns:
            Array (n_individuals, n_samples) of class 1 probabilities.
        """
        self._check_is_fitted()
        return get_population_predictions(
            self.population_, self.pset_, X, return_proba=True
        )
    
    def get_best_individual_str(self) -> str:
        """Get string representation of best individual."""
        self._check_is_fitted()
        return str(self.best_individual_)
    
    def _check_is_fitted(self) -> None:
        """Check if the classifier has been fitted."""
        if self.best_individual_ is None:
            raise RuntimeError("Classifier not fitted. Call fit() first.")


# Required for bloat control decorator



if __name__ == "__main__":
    # Quick test
    from src.data_preprocessing import load_and_preprocess
    
    dataset = load_and_preprocess(sample_size=500)
    
    clf = GPClassifier(
        population_size=20,
        n_generations=5,
        verbose=True,
    )
    
    clf.fit(dataset.X_train, dataset.y_train)
    
    predictions = clf.predict(dataset.X_test)
    proba = clf.predict_proba(dataset.X_test)
    
    print(f"\nTest predictions shape: {predictions.shape}")
    print(f"Test proba shape: {proba.shape}")
    print(f"Test accuracy: {np.mean(predictions == dataset.y_test):.4f}")
    print(f"Best individual: {clf.get_best_individual_str()[:100]}...")
