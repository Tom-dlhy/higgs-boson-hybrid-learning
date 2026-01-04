"""Sampling strategies for Active Learning."""

import numpy as np
from sklearn.cluster import KMeans


def uncertainty_sampling(
    predictions_proba: np.ndarray,
    n_samples: int,
) -> np.ndarray:
    """Select samples with highest uncertainty.
    
    Uncertainty is measured as the standard deviation of predictions
    across the population of GP individuals.
    
    Args:
        predictions_proba: Array (n_individuals, n_pool_samples) of probabilities.
        n_samples: Number of samples to select.
        
    Returns:
        Indices of selected samples from the pool.
    """
    # Compute uncertainty as std across population predictions
    uncertainty = np.std(predictions_proba, axis=0)
    
    # Select indices with highest uncertainty
    n_samples = min(n_samples, len(uncertainty))
    selected_indices = np.argsort(uncertainty)[-n_samples:]
    
    return selected_indices


def entropy_sampling(
    predictions_proba: np.ndarray,
    n_samples: int,
) -> np.ndarray:
    """Select samples with highest entropy.
    
    Uses the average probability across population and computes entropy.
    
    Args:
        predictions_proba: Array (n_individuals, n_pool_samples) of probabilities.
        n_samples: Number of samples to select.
        
    Returns:
        Indices of selected samples from the pool.
    """
    # Average probability across population
    avg_proba = np.mean(predictions_proba, axis=0)
    
    # Compute entropy: -p*log(p) - (1-p)*log(1-p)
    eps = 1e-10  # Avoid log(0)
    entropy = -(
        avg_proba * np.log(avg_proba + eps) +
        (1 - avg_proba) * np.log(1 - avg_proba + eps)
    )
    
    # Select highest entropy samples
    n_samples = min(n_samples, len(entropy))
    selected_indices = np.argsort(entropy)[-n_samples:]
    
    return selected_indices


def diversity_sampling(
    X_pool: np.ndarray,
    n_samples: int,
    random_state: int = 42,
) -> np.ndarray:
    """Select diverse samples using K-Means clustering.
    
    Selects samples closest to cluster centers to ensure diversity.
    
    Args:
        X_pool: Feature matrix of pool samples (n_pool, n_features).
        n_samples: Number of samples to select.
        random_state: Random seed for K-Means.
        
    Returns:
        Indices of selected samples from the pool.
    """
    n_samples = min(n_samples, len(X_pool))
    
    if n_samples >= len(X_pool):
        return np.arange(len(X_pool))
    
    # Cluster the pool
    kmeans = KMeans(n_clusters=n_samples, random_state=random_state, n_init=10)
    kmeans.fit(X_pool)
    
    # Find samples closest to each cluster center
    selected_indices = []
    for center in kmeans.cluster_centers_:
        distances = np.linalg.norm(X_pool - center, axis=1)
        # Avoid selecting the same sample twice
        for idx in np.argsort(distances):
            if idx not in selected_indices:
                selected_indices.append(idx)
                break
    
    return np.array(selected_indices)


def random_sampling(
    n_pool: int,
    n_samples: int,
    random_state: int = 42,
) -> np.ndarray:
    """Randomly select samples from the pool.
    
    Args:
        n_pool: Total number of samples in pool.
        n_samples: Number of samples to select.
        random_state: Random seed.
        
    Returns:
        Indices of selected samples.
    """
    rng = np.random.default_rng(random_state)
    n_samples = min(n_samples, n_pool)
    return rng.choice(n_pool, size=n_samples, replace=False)


def hybrid_sampling(
    predictions_proba: np.ndarray,
    X_pool: np.ndarray,
    n_samples: int,
    uncertainty_ratio: float = 0.5,
    random_state: int = 42,
) -> np.ndarray:
    """Combine uncertainty and diversity sampling.
    
    Args:
        predictions_proba: Array (n_individuals, n_pool_samples) of probabilities.
        X_pool: Feature matrix of pool samples.
        n_samples: Total number of samples to select.
        uncertainty_ratio: Fraction selected by uncertainty (rest by diversity).
        random_state: Random seed.
        
    Returns:
        Indices of selected samples.
    """
    n_uncertainty = int(n_samples * uncertainty_ratio)
    n_diversity = n_samples - n_uncertainty
    
    # Get uncertainty-based samples
    uncertainty_indices = uncertainty_sampling(predictions_proba, n_uncertainty)
    
    # Get diversity-based samples from remaining pool
    remaining_mask = np.ones(len(X_pool), dtype=bool)
    remaining_mask[uncertainty_indices] = False
    remaining_X = X_pool[remaining_mask]
    
    if len(remaining_X) > 0 and n_diversity > 0:
        # Get indices relative to remaining pool
        diversity_relative = diversity_sampling(remaining_X, n_diversity, random_state)
        # Map back to original indices
        remaining_indices = np.where(remaining_mask)[0]
        diversity_indices = remaining_indices[diversity_relative]
    else:
        diversity_indices = np.array([], dtype=int)
    
    return np.concatenate([uncertainty_indices, diversity_indices])
