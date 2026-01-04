"""Main experiment runner for comparing GA, GA+AL, and GA+AL+EL approaches."""

import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd

from src.config import Config
from src.data_preprocessing import load_and_preprocess, HiggsDataset
from src.evolutionary.ga_classifier import GPClassifier
from src.active_learning.ga_active import GAActiveClassifier
from src.ensemble.ga_ensemble import GAEnsembleClassifier
from src.metrics import ClassificationMetrics, compute_metrics, print_comparison
from src.visualization import generate_all_plots
from src.utils import set_global_seed, timer, CheckpointManager


def run_ga_experiment(
    dataset: HiggsDataset,
    config: Config,
    checkpoint_manager: CheckpointManager,
) -> tuple:
    """Run GA-only experiment.
    
    Returns:
        Tuple of (metrics, fitness_history).
    """
    print("\n" + "=" * 50)
    print("EXPERIMENT 1: Genetic Programming (GA)")
    print("=" * 50)
    
    clf = GPClassifier(
        population_size=config.population_size,
        n_generations=config.n_generations,
        tournament_size=config.tournament_size,
        crossover_prob=config.crossover_prob,
        mutation_prob=config.mutation_prob,
        max_depth=config.max_tree_depth,
        metric="f1",
        random_state=config.random_state,
        verbose=True,
        checkpoint_manager=checkpoint_manager,
    )
    
    with timer("GA Training") as t:
        clf.fit(dataset.X_train, dataset.y_train)
    training_time = t["elapsed"]
    
    with timer("GA Inference") as t:
        predictions = clf.predict(dataset.X_test)
    inference_time = t["elapsed"]
    
    metrics = compute_metrics(
        dataset.y_test,
        predictions,
        training_time=training_time,
        inference_time=inference_time,
    )
    
    print("\n[GA] Results:")
    print(metrics)
    print(f"\n[GA] Best individual: {clf.get_best_individual_str()[:100]}...")
    
    return metrics, clf.fitness_history_


def run_ga_al_experiment(
    dataset: HiggsDataset,
    config: Config,
    checkpoint_manager: CheckpointManager,
) -> tuple:
    """Run GA + Active Learning experiment.
    
    Returns:
        Tuple of (metrics, fitness_history).
    """
    print("\n" + "=" * 50)
    print("EXPERIMENT 2: Genetic Programming + Active Learning (GA+AL)")
    print("=" * 50)
    
    clf = GAActiveClassifier(
        population_size=config.population_size,
        n_generations=config.n_generations,
        tournament_size=config.tournament_size,
        crossover_prob=config.crossover_prob,
        mutation_prob=config.mutation_prob,
        max_depth=config.max_tree_depth,
        metric="f1",
        al_frequency=config.al_frequency,
        query_batch_size=config.query_batch_size,
        initial_pool_size=config.initial_pool_size,
        sampling_strategy=config.sampling_strategy,
        random_state=config.random_state,
        verbose=True,
        checkpoint_manager=checkpoint_manager,
    )
    
    with timer("GA+AL Training") as t:
        clf.fit(dataset.X_train, dataset.y_train)
    training_time = t["elapsed"]
    
    with timer("GA+AL Inference") as t:
        predictions = clf.predict(dataset.X_test)
    inference_time = t["elapsed"]
    
    metrics = compute_metrics(
        dataset.y_test,
        predictions,
        training_time=training_time,
        inference_time=inference_time,
    )
    
    print("\n[GA+AL] Results:")
    print(metrics)
    print(f"\n[GA+AL] Active Learning queries: {len(clf.al_history_)}")
    
    return metrics, clf.fitness_history_


def run_ga_al_el_experiment(
    dataset: HiggsDataset,
    config: Config,
    checkpoint_manager: CheckpointManager,
) -> tuple:
    """Run GA + Active Learning + Ensemble Learning experiment.
    
    Returns:
        Tuple of (metrics, fitness_history).
    """
    print("\n" + "=" * 50)
    print("EXPERIMENT 3: GA + Active Learning + Ensemble Learning (GA+AL+EL)")
    print("=" * 50)
    
    clf = GAEnsembleClassifier(
        population_size=config.population_size,
        n_generations=config.n_generations,
        tournament_size=config.tournament_size,
        crossover_prob=config.crossover_prob,
        mutation_prob=config.mutation_prob,
        max_depth=config.max_tree_depth,
        metric="f1",
        al_frequency=config.al_frequency,
        query_batch_size=config.query_batch_size,
        initial_pool_size=config.initial_pool_size,
        sampling_strategy=config.sampling_strategy,
        ensemble_size=config.ensemble_size,
        voting_strategy=config.voting_strategy,
        random_state=config.random_state,
        verbose=True,
        checkpoint_manager=checkpoint_manager,
    )
    
    with timer("GA+AL+EL Training") as t:
        clf.fit(dataset.X_train, dataset.y_train)
    training_time = t["elapsed"]
    
    with timer("GA+AL+EL Inference") as t:
        predictions = clf.predict(dataset.X_test)
    inference_time = t["elapsed"]
    
    metrics = compute_metrics(
        dataset.y_test,
        predictions,
        training_time=training_time,
        inference_time=inference_time,
    )
    
    print("\n[GA+AL+EL] Results:")
    print(metrics)
    
    diversity = clf.get_ensemble_diversity(dataset.X_test)
    print(f"\n[GA+AL+EL] Ensemble diversity: {diversity}")
    
    return metrics, clf.fitness_history_


def save_results(
    results: Dict[str, ClassificationMetrics],
    output_dir: Path,
) -> None:
    """Save results to CSV file."""
    rows = []
    for name, metrics in results.items():
        row = {"approach": name, **metrics.to_dict()}
        rows.append(row)
    
    df = pd.DataFrame(rows)
    filepath = output_dir / "metrics_comparison.csv"
    df.to_csv(filepath, index=False)
    print(f"\n[SAVE] Results saved to {filepath}")


def run_experiments(config: Config) -> Dict[str, ClassificationMetrics]:
    """Run all experiments and return results.
    
    Args:
        config: Experiment configuration.
        
    Returns:
        Dictionary mapping approach names to metrics.
    """
    # Set global seed
    set_global_seed(config.random_state)
    
    # Setup checkpoint manager
    checkpoint_manager = CheckpointManager(str(config.results_dir))
    
    # Load data
    print("\n" + "=" * 50)
    print("LOADING DATA")
    print("=" * 50)
    
    with timer("Data Loading"):
        dataset = load_and_preprocess(
            data_path=config.data_path,
            sample_size=config.sample_size,
            test_size=config.test_size,
            add_missing_indicators=config.add_missing_indicators,
            normalize=config.normalize,
            random_state=config.random_state,
        )
    
    print(f"Dataset: {dataset.n_train_samples} train, {dataset.n_test_samples} test")
    print(f"Features: {dataset.n_features}")
    
    # Run experiments
    results: Dict[str, ClassificationMetrics] = {}
    fitness_histories: Dict[str, List[dict]] = {}
    
    # Experiment 1: GA only
    metrics, history = run_ga_experiment(dataset, config, checkpoint_manager)
    results["GA"] = metrics
    fitness_histories["GA"] = history
    checkpoint_manager.save_metrics(metrics.to_dict(), "ga")
    
    # Experiment 2: GA + AL
    metrics, history = run_ga_al_experiment(dataset, config, checkpoint_manager)
    results["GA+AL"] = metrics
    fitness_histories["GA+AL"] = history
    checkpoint_manager.save_metrics(metrics.to_dict(), "ga_al")
    
    # Experiment 3: GA + AL + EL
    metrics, history = run_ga_al_el_experiment(dataset, config, checkpoint_manager)
    results["GA+AL+EL"] = metrics
    fitness_histories["GA+AL+EL"] = history
    checkpoint_manager.save_metrics(metrics.to_dict(), "ga_al_el")
    
    # Print comparison
    print_comparison(results)
    
    # Save results
    save_results(results, config.results_dir)
    
    # Generate plots
    print("\n[VIZ] Generating visualizations...")
    plots_dir = config.results_dir / "plots"
    generate_all_plots(results, fitness_histories, plots_dir)
    
    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Higgs Boson Hybrid Learning Experiments"
    )
    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="Run quick test with small sample and few generations",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=50000,
        help="Number of samples to use (default: 50000)",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=20,
        help="Number of generations (default: 20)",
    )
    parser.add_argument(
        "--population",
        type=int,
        default=100,
        help="Population size (default: 100)",
    )
    
    args = parser.parse_args()
    
    if args.quick_test:
        config = Config.for_quick_test()
        print("Running in QUICK TEST mode (small sample, few generations)")
    else:
        config = Config(
            sample_size=args.sample_size,
            n_generations=args.generations,
            population_size=args.population,
        )
    
    print("\nConfiguration:")
    print(f"  Sample size: {config.sample_size or 'Full dataset'}")
    print(f"  Population: {config.population_size}")
    print(f"  Generations: {config.n_generations}")
    print(f"  AL frequency: every {config.al_frequency} generations")
    print(f"  Ensemble size: {config.ensemble_size}")
    
    run_experiments(config)
    
    print("\n" + "=" * 50)
    print("EXPERIMENTS COMPLETE")
    print("=" * 50)
    print(f"Results saved to: {config.results_dir}")


if __name__ == "__main__":
    main()
