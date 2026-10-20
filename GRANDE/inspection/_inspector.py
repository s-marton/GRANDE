"""Cached inspection state of a fitted estimator: input probe, tree parameters, per-tree analyses."""

from __future__ import annotations

import numpy as np
import torch

from ._probe import ModelProbe
from .tree import TreeParams, analyze_tree


class Inspector:
    def __init__(self, estimator):
        self.estimator = estimator
        self.module = getattr(estimator.module_, "_orig_mod", estimator.module_)
        self.params = TreeParams.from_module(self.module)
        self.probe = ModelProbe(estimator)
        self._analyses: dict = {}

    @property
    def n_estimators(self) -> int:
        return self.params.columns.shape[0]

    def analysis(self, index: int) -> dict:
        if index not in self._analyses:
            self._analyses[index] = analyze_tree(self, index)
        return self._analyses[index]

    def analyses(self) -> list:
        return [self.analysis(e) for e in range(self.n_estimators)]

    @torch.no_grad()
    def routing(self, Z: np.ndarray):
        """Leaf indices (n, E) and normalized estimator weights (n, E) for model inputs ``Z``."""
        est = self.estimator
        leaves, weights = [], []
        bs = max(1, int(est.predict_batch_size))
        self.module.eval()
        for start in range(0, len(Z), bs):
            zb = torch.from_numpy(np.ascontiguousarray(Z[start:start + bs])).to(est.device_)
            p, w = self.module.routing(zb)
            leaves.append(p.argmax(dim=-1).cpu().numpy())
            weights.append(w.cpu().numpy())
        return np.concatenate(leaves), np.concatenate(weights)

    def leaf_indices(self, Z: np.ndarray, estimators=None) -> np.ndarray:
        leaves, _ = self.routing(Z)
        return leaves if estimators is None else leaves[:, list(estimators)]
