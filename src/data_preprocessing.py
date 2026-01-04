"""Data preprocessing for the Higgs Boson dataset."""

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Features to exclude from training (metadata columns)
EXCLUDE_COLUMNS = ["EventId", "Weight", "Label", "KaggleSet", "KaggleWeight"]

# Missing value marker in ATLAS dataset
MISSING_VALUE = -999.0


@dataclass
class HiggsDataset:
    """Container for preprocessed Higgs dataset splits."""
    
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    scaler: StandardScaler
    
    @property
    def n_features(self) -> int:
        """Number of features."""
        return self.X_train.shape[1]
    
    @property
    def n_train_samples(self) -> int:
        """Number of training samples."""
        return self.X_train.shape[0]
    
    @property
    def n_test_samples(self) -> int:
        """Number of test samples."""
        return self.X_test.shape[0]


def load_and_preprocess(
    data_path: str | Path = "data/atlas-higgs.csv",
    sample_size: int | None = None,
    test_size: float = 0.2,
    add_missing_indicators: bool = True,
    normalize: bool = True,
    random_state: int = 42,
) -> HiggsDataset:
    """Load and preprocess the Higgs Boson dataset.
    
    Args:
        data_path: Path to the CSV file.
        sample_size: Optional subsample size. None means use full dataset.
        test_size: Fraction of data for testing.
        add_missing_indicators: Whether to add is_missing_* columns.
        normalize: Whether to apply StandardScaler.
        random_state: Random seed for reproducibility.
        
    Returns:
        HiggsDataset with preprocessed train/test splits.
    """
    print(f"[DATA] Loading dataset from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"[DATA] Loaded {len(df):,} rows, {len(df.columns)} columns")
    
    # Subsample if requested (for faster development/testing)
    if sample_size is not None and sample_size < len(df):
        print(f"[DATA] Subsampling to {sample_size:,} rows...")
        df = df.sample(n=sample_size, random_state=random_state).reset_index(drop=True)
    
    # Separate features from target and metadata
    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLUMNS]
    X = df[feature_cols].copy()
    y = (df["Label"] == "s").astype(int).values  # s=1 (signal), b=0 (background)
    
    print(f"[DATA] Features: {len(feature_cols)}, Target distribution: {y.mean():.2%} signal")
    
    # Handle missing values (-999.0)
    features_with_missing = []
    for col in feature_cols:
        missing_mask = (X[col] == MISSING_VALUE)
        if missing_mask.any():
            features_with_missing.append(col)
            # Add binary indicator column if requested
            if add_missing_indicators:
                X[f"is_missing_{col}"] = missing_mask.astype(float)
            # Impute with median (excluding missing values)
            valid_values = X.loc[~missing_mask, col]
            median_value = valid_values.median()
            X.loc[missing_mask, col] = median_value
    
    print(f"[DATA] Features with missing values: {len(features_with_missing)}")
    if add_missing_indicators:
        print(f"[DATA] Added {len(features_with_missing)} is_missing_* indicator columns")
    
    feature_names = list(X.columns)
    X_values = X.values
    
    # Train/test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X_values, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    print(f"[DATA] Split: {len(X_train):,} train, {len(X_test):,} test")
    
    # Normalize features
    scaler = StandardScaler()
    if normalize:
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        print("[DATA] Applied StandardScaler normalization")
    else:
        scaler.fit(X_train)  # Fit anyway for consistency
    
    return HiggsDataset(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        feature_names=feature_names,
        scaler=scaler,
    )


def get_pool_and_initial(
    X_train: np.ndarray,
    y_train: np.ndarray,
    initial_size: int = 100,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split training data into initial labeled set and unlabeled pool.
    
    For active learning, we start with a small labeled set and query from
    the remaining pool.
    
    Args:
        X_train: Training features.
        y_train: Training labels.
        initial_size: Size of initial labeled set.
        random_state: Random seed.
        
    Returns:
        Tuple of (X_initial, y_initial, X_pool, y_pool).
    """
    rng = np.random.default_rng(random_state)
    indices = rng.permutation(len(X_train))
    
    initial_indices = indices[:initial_size]
    pool_indices = indices[initial_size:]
    
    return (
        X_train[initial_indices],
        y_train[initial_indices],
        X_train[pool_indices],
        y_train[pool_indices],
    )


if __name__ == "__main__":
    # Quick test
    dataset = load_and_preprocess(sample_size=1000)
    print("\nDataset loaded successfully!")
    print(f"  Train shape: {dataset.X_train.shape}")
    print(f"  Test shape: {dataset.X_test.shape}")
    print(f"  Features: {dataset.n_features}")
