"""GA + Active Learning + Ensemble classifier."""

from typing import List, Optional, Literal

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

from src.active_learning.ga_active import GAActiveClassifier
from src.evolutionary.fitness import predict_individual
from src.ensemble.voting import (
    hard_voting,
    soft_voting,
    soft_voting_proba,
    weighted_voting,
    weighted_voting_proba,
    select_diverse_top_k,
)
from src.utils import CheckpointManager


class GAEnsembleClassifier(BaseEstimator, ClassifierMixin):
    """Complete GA + Active Learning + Ensemble classifier.
    
    This classifier combines:
    1. Genetic Programming for evolving expression trees
    2. Active Learning for intelligent sample selection
    3. Ensemble Learning for final predictions using top-k individuals
    
    Attributes:
        Inherits all GA+AL parameters plus:
        ensemble_size: Number of top individuals to use for ensemble.
        voting_strategy: Voting method ("hard", "soft", "weighted").
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
        ensemble_size: int = 10,
        voting_strategy: Literal["hard", "soft", "weighted"] = "soft",
        use_similarity_filter: bool = True,
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
        self.ensemble_size = ensemble_size
        self.voting_strategy = voting_strategy
        self.use_similarity_filter = use_similarity_filter
        self.random_state = random_state
        self.verbose = verbose
        self.checkpoint_manager = checkpoint_manager
        
        # Internal GA+AL classifier
        self._ga_al_clf: Optional[GAActiveClassifier] = None
        
        # Ensemble members
        self.ensemble_individuals_: Optional[List] = None
        self.ensemble_weights_: Optional[np.ndarray] = None
        
        self.classes_: np.ndarray = np.array([0, 1])
        self.n_features_in_: int = 0
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> "GAEnsembleClassifier":
        """Fit the GA+AL+EL classifier.
        
        Args:
            X: Training features (n_samples, n_features).
            y: Training labels (n_samples,).
            
        Returns:
            self
        """
        # Create and fit the GA+AL classifier
        self._ga_al_clf = GAActiveClassifier(
            population_size=self.population_size,
            n_generations=self.n_generations,
            tournament_size=self.tournament_size,
            crossover_prob=self.crossover_prob,
            mutation_prob=self.mutation_prob,
            max_depth=self.max_depth,
            metric=self.metric,
            al_frequency=self.al_frequency,
            query_batch_size=self.query_batch_size,
            initial_pool_size=self.initial_pool_size,
            sampling_strategy=self.sampling_strategy,
            random_state=self.random_state,
            verbose=self.verbose,
            checkpoint_manager=self.checkpoint_manager,
        )
        
        self._ga_al_clf.fit(X, y)
        
        self.n_features_in_ = self._ga_al_clf.n_features_in_
        
        # Select top-k diverse individuals for ensemble
        k = min(self.ensemble_size, len(self._ga_al_clf.population_))
        self.ensemble_individuals_, fitness_values = select_diverse_top_k(
            self._ga_al_clf.population_, k, use_similarity_filter=self.use_similarity_filter
        )
        self.ensemble_weights_ = np.array(fitness_values)
        
        if self.verbose:
            print(f"[ENSEMBLE] Selected top {len(self.ensemble_individuals_)} individuals")
            print(f"[ENSEMBLE] Fitness range: {min(fitness_values):.4f} - {max(fitness_values):.4f}")
            print(f"[ENSEMBLE] Voting strategy: {self.voting_strategy}")
        
        return self
    
    def _get_ensemble_predictions(
        self,
        X: np.ndarray,
        return_proba: bool = True,
    ) -> np.ndarray:
        """Get predictions from all ensemble members.
        
        Args:
            X: Features.
            return_proba: Whether to return probabilities.
            
        Returns:
            Array (ensemble_size, n_samples) of predictions.
        """
        predictions = []
        for individual in self.ensemble_individuals_:
            preds = predict_individual(
                individual,
                self._ga_al_clf.pset_,
                X,
                return_proba=return_proba,
            )
            predictions.append(preds)
        
        return np.array(predictions)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels using ensemble voting.
        
        Args:
            X: Features (n_samples, n_features).
            
        Returns:
            Predicted labels (n_samples,).
        """
        self._check_is_fitted()
        
        if self.voting_strategy == "hard":
            # Get class predictions from each ensemble member
            class_preds = self._get_ensemble_predictions(X, return_proba=False)
            return hard_voting(class_preds)
        
        elif self.voting_strategy == "soft":
            proba_preds = self._get_ensemble_predictions(X, return_proba=True)
            return soft_voting(proba_preds)
        
        elif self.voting_strategy == "weighted":
            proba_preds = self._get_ensemble_predictions(X, return_proba=True)
            return weighted_voting(proba_preds, self.ensemble_weights_)
        
        else:
            raise ValueError(f"Unknown voting strategy: {self.voting_strategy}")
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities using ensemble voting.
        
        Args:
            X: Features (n_samples, n_features).
            
        Returns:
            Probability matrix (n_samples, 2).
        """
        self._check_is_fitted()
        
        proba_preds = self._get_ensemble_predictions(X, return_proba=True)
        
        if self.voting_strategy == "weighted":
            return weighted_voting_proba(proba_preds, self.ensemble_weights_)
        else:
            return soft_voting_proba(proba_preds)
    
    def get_ensemble_diversity(self, X: np.ndarray) -> dict:
        """Compute diversity metrics for the ensemble.
        
        Args:
            X: Features to evaluate on.
            
        Returns:
            Dictionary with diversity metrics.
        """
        self._check_is_fitted()
        
        proba_preds = self._get_ensemble_predictions(X, return_proba=True)
        
        # Pairwise disagreement
        n_members = len(self.ensemble_individuals_)
        disagreements = []
        for i in range(n_members):
            for j in range(i + 1, n_members):
                pred_i = (proba_preds[i] > 0.5).astype(int)
                pred_j = (proba_preds[j] > 0.5).astype(int)
                disagreement = np.mean(pred_i != pred_j)
                disagreements.append(disagreement)
        
        return {
            "avg_disagreement": np.mean(disagreements) if disagreements else 0,
            "std_across_ensemble": np.mean(np.std(proba_preds, axis=0)),
            "n_members": n_members,
        }
    
    @property
    def fitness_history_(self) -> List[dict]:
        """Get fitness history from underlying GA+AL classifier."""
        if self._ga_al_clf:
            return self._ga_al_clf.fitness_history_
        return []
    
    @property
    def al_history_(self) -> List[dict]:
        """Get active learning history from underlying GA+AL classifier."""
        if self._ga_al_clf:
            return self._ga_al_clf.al_history_
        return []
    
    def _check_is_fitted(self) -> None:
        """Check if the classifier has been fitted."""
        if self.ensemble_individuals_ is None:
            raise RuntimeError("Classifier not fitted. Call fit() first.")


if __name__ == "__main__":
    # Quick test
    from src.data_preprocessing import load_and_preprocess
    
    dataset = load_and_preprocess(sample_size=500)
    
    clf = GAEnsembleClassifier(
        population_size=20,
        n_generations=10,
        al_frequency=3,
        query_batch_size=10,
        initial_pool_size=50,
        ensemble_size=5,
        voting_strategy="soft",
        verbose=True,
    )
    
    clf.fit(dataset.X_train, dataset.y_train)
    
    predictions = clf.predict(dataset.X_test)
    proba = clf.predict_proba(dataset.X_test)
    diversity = clf.get_ensemble_diversity(dataset.X_test)
    
    print(f"\nTest accuracy: {np.mean(predictions == dataset.y_test):.4f}")
    print(f"Proba shape: {proba.shape}")
    print(f"Ensemble diversity: {diversity}")
