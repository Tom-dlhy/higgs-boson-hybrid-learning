"""Fitness evaluation for GP individuals."""

import math
from typing import Callable, Tuple

import numpy as np
from deap import gp
from sklearn.metrics import f1_score, accuracy_score


def sigmoid(x: float) -> float:
    """Sigmoid activation to convert raw output to probability.
    
    Args:
        x: Raw GP tree output.
        
    Returns:
        Probability in [0, 1].
    """
    # Clamp to prevent overflow
    x = max(-500, min(500, x))
    return 1.0 / (1.0 + math.exp(-x))


def compile_individual(individual: gp.PrimitiveTree, pset: gp.PrimitiveSet) -> Callable:
    """Compile a GP individual into a callable function.
    
    Args:
        individual: GP tree to compile.
        pset: Primitive set used to create the individual.
        
    Returns:
        Callable function that takes feature values and returns output.
    """
    return gp.compile(expr=individual, pset=pset)


def predict_individual(
    individual: gp.PrimitiveTree,
    pset: gp.PrimitiveSet,
    X: np.ndarray,
    return_proba: bool = False,
) -> np.ndarray:
    """Generate predictions for an individual on data X.
    
    Args:
        individual: GP tree.
        pset: Primitive set.
        X: Feature matrix (n_samples, n_features).
        return_proba: If True, return probabilities instead of classes.
        
    Returns:
        Array of predictions (classes 0/1) or probabilities.
    """
    func = compile_individual(individual, pset)
    
    predictions = []
    for sample in X:
        try:
            output = func(*sample)
            if return_proba:
                predictions.append(sigmoid(output))
            else:
                predictions.append(1 if output > 0 else 0)
        except (OverflowError, ValueError, ZeroDivisionError):
            # Handle any remaining edge cases
            predictions.append(0.5 if return_proba else 0)
    
    return np.array(predictions)


def evaluate_individual(
    individual: gp.PrimitiveTree,
    pset: gp.PrimitiveSet,
    X: np.ndarray,
    y: np.ndarray,
    metric: str = "f1",
    bloat_penalty: float = 0.0,
) -> Tuple[float]:
    """Evaluate fitness of a GP individual with optional bloat penalty.
    
    Args:
        individual: GP tree to evaluate.
        pset: Primitive set.
        X: Feature matrix.
        y: True labels.
        metric: Metric to use ("f1", "accuracy").
        bloat_penalty: Penalty coefficient for tree size (higher = more penalty).
                      This encourages simpler, more diverse trees.
        
    Returns:
        Tuple containing fitness value (for DEAP compatibility).
    """
    try:
        predictions = predict_individual(individual, pset, X, return_proba=False)
        
        if metric == "f1":
            score = f1_score(y, predictions, zero_division=0)
        elif metric == "accuracy":
            score = accuracy_score(y, predictions)
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        # Apply bloat penalty to encourage smaller, diverse trees
        if bloat_penalty > 0:
            tree_size = len(individual)
            penalty = bloat_penalty * tree_size
            score = max(0, score - penalty)
            
        return (score,)
        
    except Exception:
        # Return worst possible fitness on error
        return (0.0,)


def batch_evaluate(
    population: list,
    pset: gp.PrimitiveSet,
    X: np.ndarray,
    y: np.ndarray,
    metric: str = "f1",
) -> list:
    """Evaluate fitness for entire population.
    
    Args:
        population: List of GP individuals.
        pset: Primitive set.
        X: Feature matrix.
        y: True labels.
        metric: Metric to use.
        
    Returns:
        List of fitness values (as tuples).
    """
    return [evaluate_individual(ind, pset, X, y, metric) for ind in population]


def get_population_predictions(
    population: list,
    pset: gp.PrimitiveSet,
    X: np.ndarray,
    return_proba: bool = True,
) -> np.ndarray:
    """Get predictions from all individuals in population.
    
    Args:
        population: List of GP individuals.
        pset: Primitive set.
        X: Feature matrix.
        return_proba: Return probabilities instead of classes.
        
    Returns:
        Array of shape (n_individuals, n_samples) with predictions.
    """
    all_predictions = []
    for individual in population:
        preds = predict_individual(individual, pset, X, return_proba=return_proba)
        all_predictions.append(preds)
    
    return np.array(all_predictions)
