"""Voting strategies for ensemble learning."""

from typing import List

import numpy as np


def hard_voting(predictions: np.ndarray) -> np.ndarray:
    """Hard voting: majority class wins.
    
    Args:
        predictions: Array (n_individuals, n_samples) of class predictions (0 or 1).
        
    Returns:
        Array (n_samples,) of final class predictions.
    """
    # Sum predictions across individuals (each is 0 or 1)
    votes = np.sum(predictions, axis=0)
    # Majority vote: class 1 if more than half voted for it
    threshold = predictions.shape[0] / 2
    return (votes > threshold).astype(int)


def soft_voting(probabilities: np.ndarray) -> np.ndarray:
    """Soft voting: average probabilities.
    
    Args:
        probabilities: Array (n_individuals, n_samples) of class 1 probabilities.
        
    Returns:
        Array (n_samples,) of final class predictions.
    """
    # Average probability across individuals
    avg_proba = np.mean(probabilities, axis=0)
    return (avg_proba > 0.5).astype(int)


def soft_voting_proba(probabilities: np.ndarray) -> np.ndarray:
    """Soft voting returning probability estimates.
    
    Args:
        probabilities: Array (n_individuals, n_samples) of class 1 probabilities.
        
    Returns:
        Array (n_samples, 2) of class probabilities [class0, class1].
    """
    avg_proba = np.mean(probabilities, axis=0)
    return np.column_stack([1 - avg_proba, avg_proba])


def weighted_voting(
    probabilities: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Weighted voting: weight predictions by fitness scores.
    
    Args:
        probabilities: Array (n_individuals, n_samples) of class 1 probabilities.
        weights: Array (n_individuals,) of weights (e.g., fitness scores).
        
    Returns:
        Array (n_samples,) of final class predictions.
    """
    # Normalize weights
    weights = np.array(weights)
    weights = weights / weights.sum()
    
    # Weighted average
    weighted_proba = np.average(probabilities, axis=0, weights=weights)
    return (weighted_proba > 0.5).astype(int)


def weighted_voting_proba(
    probabilities: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Weighted voting returning probability estimates.
    
    Args:
        probabilities: Array (n_individuals, n_samples) of class 1 probabilities.
        weights: Array (n_individuals,) of weights.
        
    Returns:
        Array (n_samples, 2) of class probabilities.
    """
    weights = np.array(weights)
    weights = weights / weights.sum()
    
    weighted_proba = np.average(probabilities, axis=0, weights=weights)
    return np.column_stack([1 - weighted_proba, weighted_proba])


def select_top_k(
    population: List,
    k: int,
    fitness_key=None,
) -> tuple:
    """Select top-k individuals by fitness.
    
    Args:
        population: List of individuals with fitness attribute.
        k: Number of top individuals to select.
        fitness_key: Optional function to extract fitness. Default uses .fitness.values[0].
        
    Returns:
        Tuple of (top_individuals, fitness_values).
    """
    if fitness_key is None:
        fitness_key = lambda ind: ind.fitness.values[0]
    
    # Sort by fitness (descending)
    sorted_pop = sorted(population, key=fitness_key, reverse=True)
    top_k = sorted_pop[:k]
    
    top_individuals = top_k
    fitness_values = [fitness_key(ind) for ind in top_k]
    
    return top_individuals, fitness_values


def select_diverse_top_k(
    population: List,
    k: int,
    fitness_key=None,
    use_similarity_filter: bool = True,
) -> tuple:
    """Select top-k diverse individuals by fitness with similarity filtering.
    
    Filters out structurally identical individuals to ensure ensemble diversity.
    
    Args:
        population: List of individuals with fitness attribute.
        k: Number of top individuals to select.
        fitness_key: Optional function to extract fitness.
        use_similarity_filter: If True, skip individuals identical to already selected ones.
        
    Returns:
        Tuple of (top_individuals, fitness_values).
    """
    if fitness_key is None:
        fitness_key = lambda ind: ind.fitness.values[0]
    
    # Sort by fitness (descending)
    sorted_pop = sorted(population, key=fitness_key, reverse=True)
    
    if not use_similarity_filter:
        top_k = sorted_pop[:k]
        return top_k, [fitness_key(ind) for ind in top_k]
    
    # Select diverse individuals (no duplicates)
    selected = []
    seen_structures = set()
    
    for ind in sorted_pop:
        if len(selected) >= k:
            break
        
        # Use string representation as structure fingerprint
        structure = str(ind)
        
        if structure not in seen_structures:
            selected.append(ind)
            seen_structures.add(structure)
    
    fitness_values = [fitness_key(ind) for ind in selected]
    
    return selected, fitness_values
