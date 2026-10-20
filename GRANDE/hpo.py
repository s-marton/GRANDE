"""Default configuration and random-search space of GRANDE, as used on TabArena.

Example
-------
>>> from GRANDE import GRANDEClassifier
>>> from GRANDE.hpo import sample_configs
>>> configs = sample_configs(n_configs=20, seed=1234)
>>> models = [GRANDEClassifier(**cfg) for cfg in configs]
"""

from __future__ import annotations

import numpy as np

# Default configuration (TabArena "c1" config). These are also the defaults of
# GRANDEClassifier / GRANDERegressor.
TABARENA_DEFAULT_PARAMS = {
    "depth": 5,
    "n_estimators": 1024,
    "learning_rate_weights": 0.005,
    "learning_rate_index": 0.01,
    "learning_rate_values": 0.01,
    "learning_rate_leaf": 0.01,
    "use_class_weights": False,
    "dropout": 0.2,
    "selected_variables": 0.8,
    "data_subset_fraction": 1.0,
    "bootstrap": False,
    "verbose": 0,
    "batch_size": 256,
    "early_stopping_epochs": 10,
    "epochs": 200,
    "focal_loss": False,
    "es_metric": False,
    "missing_values": False,
    "swa": False,
    "cosine_decay_restarts": False,
    "learning_rate_embedding": 0.001,
    "use_category_embeddings": False,
    "use_numeric_embeddings": True,
    "embedding_threshold": 1,
    "embedding_dim_cat": 8,
    "embedding_dim_num": 4,
    "label_smoothing": 0.0,
    "use_robust_scale_smoothing": True,
    "loo_cardinality": 10,
    "optimizer": "adam",
    "reduce_on_plateau_scheduler": False,
    "num_emb_n_bins": 8,
}


def _log_uniform(rng: np.random.Generator, low: float, high: float) -> float:
    return np.exp(rng.uniform(np.log(low), np.log(high)))


def sample_config(rng: np.random.Generator) -> dict:
    """Draw one random configuration (parameters not listed keep their defaults).

    ``n_estimators`` is fixed on purpose: all trees are trained jointly, so it is the exact
    ensemble size rather than an upper bound, and larger values mainly add compute.
    """
    params = {
        "depth": rng.choice([3, 5], p=[0.5, 0.5]),
        "n_estimators": 1024,
        "learning_rate_weights": 0.001,
        "learning_rate_index": 0.01,
        "learning_rate_values": _log_uniform(rng, 1e-3, 5e-2),
        "learning_rate_leaf": _log_uniform(rng, 1e-2, 5e-1),
        "use_class_weights": False,
        "dropout": rng.choice([0.0, 0.2]),
        "selected_variables": 0.8,
        "data_subset_fraction": 1.0,
        "bootstrap": False,
        "verbose": 0,
        "batch_size": 256,
        "early_stopping_epochs": 50,
        "epochs": 250,
        "focal_loss": False,
        "es_metric": True,
        "missing_values": False,
        "swa": False,
        "cosine_decay_restarts": rng.choice([True, False], p=[0.5, 0.5]),
        "optimizer": "adam",
        "reduce_on_plateau_scheduler": True,
        "learning_rate_embedding": _log_uniform(rng, 1e-4, 5e-2),
        "use_category_embeddings": rng.choice([True, False], p=[0.5, 0.5]),
        "embedding_dim_cat": 8,
        "use_numeric_embeddings": rng.choice([True, False], p=[0.5, 0.5]),
        "embedding_dim_num": 8,
        "embedding_threshold": 1,
        "label_smoothing": 0.0,
        "use_robust_scale_smoothing": False,
    }
    return {k: (v.item() if isinstance(v, np.generic) else v) for k, v in params.items()}


def sample_configs(n_configs: int = 200, seed: int = 1234) -> list[dict]:
    """Draw ``n_configs`` random configurations with a seeded generator (TabArena uses seed 1234)."""
    rng = np.random.default_rng(seed)
    return [sample_config(rng) for _ in range(n_configs)]
