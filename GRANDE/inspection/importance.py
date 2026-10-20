"""Feature importances that exploit the tree structure of GRANDE.

Two variants, both aggregated over all model inputs derived from a feature (encodings, embedding
dimensions) and computed on the pruned trees, i.e. redundant splits never receive credit:

* ``structure`` (data-free, used for ``feature_importances_``): every tree distributes its expected
  estimator weight over its effective splits in proportion to the probability of reaching each split
  when both branches are equally likely (2^-depth). The expected estimator weight of a tree is the mean of
  exp(leaf weight) over its leaves, again weighted by 2^-depth.
* ``path`` (data-based): for every sample and tree, the sample's instance-wise estimator weight is shared
  equally among the effective splits on the sample's decision path. The importance of a feature is its
  share of all routing decisions of the ensemble on the data, weighted by how much each tree counts for
  the respective sample.
"""

from __future__ import annotations

import numpy as np


def structural_importances(inspector) -> np.ndarray:
    n_features = len(inspector.probe.domains)
    importances = np.zeros(n_features)
    leaf_weights = inspector.params.leaf_weights
    tree_weights, contributions = [], []
    for e in range(inspector.n_estimators):
        a = inspector.analysis(e)
        effective = a["reachable"] & ~a["redundant"] & (a["feature"] >= 0)
        leaves = a["leaf_pruned_depth"] >= 0
        reach_leaf = 2.0 ** -a["leaf_pruned_depth"][leaves]
        tree_weights.append(np.sum(reach_leaf * np.exp(leaf_weights[e, leaves])) / reach_leaf.sum())
        if not effective.any():
            contributions.append(None)
            continue
        reach = 2.0 ** -a["pruned_depth"][effective]
        contributions.append((a["feature"][effective], reach / reach.sum()))
    tree_weights = np.asarray(tree_weights)
    tree_weights = tree_weights / tree_weights.sum()
    for w, contrib in zip(tree_weights, contributions, strict=True):
        if contrib is not None:
            np.add.at(importances, contrib[0], w * contrib[1])
    total = importances.sum()
    return importances / total if total > 0 else importances


def path_importances(inspector, Z: np.ndarray, batch_size: int = 512) -> np.ndarray:
    n_features = len(inspector.probe.domains)
    analyses = inspector.analyses()
    effective = np.stack([a["reachable"] & ~a["redundant"] & (a["feature"] >= 0) for a in analyses])  # (E, I)
    feature = np.stack([a["feature"] for a in analyses])  # (E, I)
    path_nodes = inspector.params.path_nodes  # (L, D)
    importances = np.zeros(n_features)
    E = effective.shape[0]
    tree_idx = np.arange(E)[None, :, None]
    for start in range(0, len(Z), batch_size):
        leaves, weights = inspector.routing(Z[start:start + batch_size])  # (b, E)
        nodes = path_nodes[leaves]  # (b, E, D)
        eff = effective[tree_idx, nodes]
        n_eff = eff.sum(axis=-1, keepdims=True)
        credit = np.where(eff, weights[..., None] / np.maximum(n_eff, 1), 0.0)
        importances += np.bincount(feature[tree_idx, nodes][eff], weights=credit[eff], minlength=n_features)
    total = importances.sum()
    return importances / total if total > 0 else importances
