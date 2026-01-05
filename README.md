# Higgs Boson Hybrid Learning

Combining Evolutionary Learning, Active Learning and Ensemble Learning to solve the Higgs Boson Detection problem.

**Course:** Advanced ML II  
**Institution:** ESILV-DIA5  
**Academic Year:** 2025-2026  
**Student:** Tom DELAHAYE

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Installation](#installation)
3. [Project Structure](#project-structure)
4. [Methodology](#methodology)
5. [Usage](#usage)
6. [Results](#results)
7. [Design Decisions](#design-decisions)
8. [References](#references)

---

## Project Overview

This project implements a hybrid machine learning approach for the Higgs Boson detection problem, combining three learning paradigms:

| Paradigm | Description |
|----------|-------------|
| **GA (Genetic Programming)** | Evolving mathematical expression trees as classifiers using DEAP |
| **AL (Active Learning)** | Smart selection of the most informative samples |
| **EL (Ensemble Learning)** | Combining multiple classifiers via voting |

### Dataset

- **Source:** [Kaggle Higgs Boson Challenge](https://www.kaggle.com/c/higgs-boson)
- **Size:** ~818,000 events
- **Features:** 30 physics characteristics (DER_*, PRI_*)
- **Target:** Signal (s) vs Background (b)

---

## Installation

### Prerequisites

- Python >= 3.12
- [uv](https://github.com/astral-sh/uv) package manager (recommended)

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/higgs-boson-hybrid-learning.git
cd higgs-boson-hybrid-learning

# Install dependencies using uv
uv sync

# Or using pip
pip install -e .
```

### Download Dataset

Download the Higgs Boson dataset from Kaggle and place it in the `data/` directory:

```
data/atlas-higgs.csv
```

---

## Project Structure

```
higgs-boson-hybrid-learning/
├── data/                           # Dataset directory
│   └── atlas-higgs.csv
├── project/                        # Project documentation
│   ├── homework.md                 # Assignment description
│   └── project.md                  # Design decisions
├── results/                        # Experiment outputs
│   ├── plots/                      # Generated visualizations
│   ├── metrics_comparison.csv
│   └── *.json                      # Checkpoint files
├── src/                            # Source code
│   ├── evolutionary/               # Genetic Programming module
│   │   ├── ga_classifier.py        # GPClassifier implementation
│   │   ├── primitives.py           # GP primitive set definition
│   │   └── fitness.py              # Fitness evaluation functions
│   ├── active_learning/            # Active Learning module
│   │   ├── ga_active.py            # GAActiveClassifier
│   │   └── sampling.py             # Sampling strategies
│   ├── ensemble/                   # Ensemble Learning module
│   │   ├── ga_ensemble.py          # GAEnsembleClassifier
│   │   └── voting.py               # Voting strategies
│   ├── config.py                   # Configuration settings
│   ├── data_preprocessing.py       # Data loading and preprocessing
│   ├── experiments.py              # Command-line experiment runner
│   ├── metrics.py                  # Performance metrics
│   ├── utils.py                    # Utility functions
│   └── visualization.py            # Plotting functions
├── notebook.ipynb                  # Interactive experiment notebook
├── main.py                         # Entry point
├── pyproject.toml                  # Project dependencies
└── README.md                       # This file
```

---

## Methodology

### Step 1: Data Preprocessing

1. **Missing Value Handling (-999.0):** Median imputation + binary indicator columns (`is_missing_*`)
2. **Normalization:** StandardScaler
3. **Split:** 80% train / 20% test (stratified)

### Step 2: Evolutionary Learning (GA)

Using the DEAP library to evolve mathematical expression trees:

**Primitives (Operations):**
| Category | Operators |
|----------|-----------|
| Arithmetic | `add`, `sub`, `mul`, `div` (protected) |
| Mathematical | `sin`, `cos`, `sqrt`, `exp`, `abs` |
| Conditional | `if_then_else`, `>`, `<` |

**Terminals:**
- Dataset features (x0, x1, ..., xN)
- Ephemeral constants (random values [-1, 1])

**Fitness Function:** F1-Score (balance between precision and recall)

### Step 3: GA + Active Learning

At regular intervals during evolution, the classifier queries the most informative samples from a pool:

**Sampling Strategies:**
| Strategy | Description |
|----------|-------------|
| **Uncertainty** | Selects samples with highest prediction variance across population |
| **Diversity** | Uses K-Means to select diverse samples |
| **Random** | Random baseline |
| **Hybrid** | Combines uncertainty and diversity |

### Step 4: GA + AL + Ensemble Learning

The final population is used for ensemble prediction:

**Voting Strategies:**
| Strategy | Description |
|----------|-------------|
| **Hard Voting** | Majority vote on predicted classes |
| **Soft Voting** | Average of probabilities |
| **Weighted Voting** | Weighted average by individual fitness |

**Similarity Filter:** To avoid clones in the ensemble, structurally identical individuals are filtered out.

---

## Usage

### Using the Notebook

The recommended way to explore the project is through the Jupyter notebook:

```bash
uv run jupyter notebook notebook.ipynb
```

### Command Line

Run experiments from the command line:

```bash
# Quick test (small sample)
uv run python -m src.experiments --quick-test

# Full experiment
uv run python -m src.experiments --sample-size 150000 --generations 40 --population 150
```

### Programmatic Usage

```python
from src.config import Config
from src.data_preprocessing import load_and_preprocess
from src.evolutionary.ga_classifier import GPClassifier
from src.active_learning.ga_active import GAActiveClassifier
from src.ensemble.ga_ensemble import GAEnsembleClassifier
from src.metrics import compute_metrics

# Load data
config = Config(sample_size=50000)
dataset = load_and_preprocess(
    data_path=config.data_path,
    sample_size=config.sample_size,
    test_size=config.test_size,
)

# Train GA classifier
ga_clf = GPClassifier(
    population_size=config.population_size,
    n_generations=config.n_generations,
    verbose=True,
)
ga_clf.fit(dataset.X_train, dataset.y_train)

# Train GA+AL classifier
ga_al_clf = GAActiveClassifier(
    population_size=config.population_size,
    n_generations=config.n_generations,
    al_frequency=config.al_frequency,
    query_batch_size=config.query_batch_size,
)
ga_al_clf.fit(dataset.X_train, dataset.y_train)

# Train GA+AL+EL classifier
ga_al_el_clf = GAEnsembleClassifier(
    population_size=config.population_size,
    n_generations=config.n_generations,
    ensemble_size=config.ensemble_size,
    voting_strategy=config.voting_strategy,
)
ga_al_el_clf.fit(dataset.X_train, dataset.y_train)

# Evaluate
for name, clf in [('GA', ga_clf), ('GA+AL', ga_al_clf), ('GA+AL+EL', ga_al_el_clf)]:
    predictions = clf.predict(dataset.X_test)
    metrics = compute_metrics(dataset.y_test, predictions)
    print(f"{name}: {metrics}")
```

---

## Results

### Performance Comparison (150k samples, 40 generations)

| Approach | Accuracy | Precision | Recall | F1-Score | Training Time (s) |
|----------|----------|-----------|--------|----------|-------------------|
| GA       | 0.7031   | 0.5453    | 0.7841 | 0.6433   | 1056.34           |
| GA+AL    | 0.5876   | 0.4485    | 0.9053 | 0.5998   | 354.30            |
| GA+AL+EL | 0.5927   | 0.4518    | 0.9041 | 0.6025   | 356.84            |

### Key Observations

**Performance:**
- **GA** achieves the best F1-Score (0.6433) and Accuracy (0.7031) when trained on the full dataset
- **GA+AL** and **GA+AL+EL** show higher Recall, indicating better detection of Higgs signals
- Ensemble Learning slightly improves over GA+AL alone

**Efficiency:**
- **GA+AL** and **GA+AL+EL** are approximately 3x faster than pure GA
- Active Learning reduces training data requirements significantly

**Strengths and Weaknesses:**
| Approach | Strengths | Weaknesses |
|----------|-----------|------------|
| GA | Best overall performance, simple | Slow training on full dataset |
| GA+AL | Fast training, intelligent data selection | Risk of over-specialization |
| GA+AL+EL | Robustness via ensemble voting, diversity | Complexity overhead |

---

## Design Decisions

### 1. Missing Value Handling (-999.0)

The ATLAS dataset uses -999.0 for physically undefined values (e.g., mass of a non-existent jet). We use median imputation combined with `is_missing_*` indicator columns to allow GP trees to exploit missing value patterns.

### 2. Sigmoid Transformation for ModAL Compatibility

GP trees output raw real numbers. We apply sigmoid transformation (`P(y=1|x) = 1/(1 + exp(-output))`) to convert to probabilities, enabling uncertainty-based sampling in Active Learning.

### 3. Subsampling for Performance

The full 818k dataset is too large for GP evaluation. We use configurable subsampling (1k for tests, 50-150k for experiments) to balance performance and development speed.

### 4. Bloat Penalty

To encourage diversity in the population, we apply a complexity penalty: `score = score - 0.001 * len(tree)`. This forces trees to find different mathematical paths rather than converging to clones.

### 5. Similarity Filter

Identical individuals are filtered from the top-k ensemble to guarantee structural diversity and meaningful ensemble predictions.

### 6. Reproducibility

Global seeds are set via `utils.set_global_seed()` to ensure fair comparisons and reproducible results.

### 7. Checkpointing

Automatic saving of intermediate results protects against data loss during long GP runs.

---

## Dependencies

- deap >= 1.4.3 (Genetic Programming)
- scikit-learn >= 1.8.0 (ML utilities)
- numpy >= 2.3.5
- pandas >= 2.3.3
- matplotlib >= 3.10.8
- modal-python >= 0.4.2.1
- tqdm >= 4.67.1

---

## References

- [DEAP Documentation](https://deap.readthedocs.io/)
- [Higgs Boson Challenge on Kaggle](https://www.kaggle.com/c/higgs-boson)
- [Active Learning with ModAL](https://modal-python.readthedocs.io/)

---

## License

This project is developed for academic purposes as part of the Advanced ML II course at ESILV.
