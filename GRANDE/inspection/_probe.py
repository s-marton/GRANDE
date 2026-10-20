"""Map GRANDE's model inputs back to the original features.

GRANDE's trees split on preprocessed inputs: scaled numeric values, one-hot / ordinal / leave-one-out /
frequency encodings and numeric or categorical embedding dimensions. Every such input depends on exactly
one original feature. :class:`ModelProbe` pushes the training domain of each feature through the fitted
preprocessing pipeline (holding all other features fixed) and records, for every model input column, its
value as a function of that one feature. Split conditions can then be expressed on the original features.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

#: Maximum number of probe values of a continuous feature (quantiles of the training data).
MAX_PROBES = 512
#: Rows per preprocessing call while probing.
_CHUNK_ROWS = 16384


@dataclass
class FeatureDomain:
    """Training domain of one original feature."""

    name: str
    kind: str  # "categorical" | "discrete" (numeric, <= 2 values) | "numeric"
    values: np.ndarray  # sorted unique values (numeric: quantiles if there are many), categories in category order
    has_nan: bool
    integer: bool = False  # numeric feature whose training values are all integers


@dataclass
class FeatureDomains:
    features: list
    reference: pd.DataFrame  # one training row; the other features are held at its values while probing


def capture_feature_domains(X: pd.DataFrame, max_probes: int = MAX_PROBES) -> FeatureDomains:
    """Record the training domain of every column of ``X`` (called during ``fit``)."""
    features = []
    for col in X.columns:
        s = X[col]
        has_nan = bool(s.isna().any())
        integer = False
        if isinstance(s.dtype, pd.CategoricalDtype):
            present = set(s.dropna().unique())
            values = np.array([c for c in s.cat.categories if c in present], dtype=object)
            kind = "categorical"
        elif s.dtype == object or pd.api.types.is_string_dtype(s.dtype):
            values = np.array(sorted(s.dropna().unique(), key=str), dtype=object)
            kind = "categorical"
        else:
            unique = np.sort(s.dropna().unique())
            if len(unique) <= 2:
                values, kind = unique, "discrete"
            else:
                numeric = s.dropna().to_numpy(dtype=float)
                values = unique.astype(float)
                integer = bool(np.all(values == np.round(values)))
                if len(values) > max_probes:
                    values = np.unique(np.quantile(numeric, np.linspace(0.0, 1.0, max_probes)))
                kind = "numeric"
        features.append(FeatureDomain(name=str(col), kind=kind, values=values, has_nan=has_nan, integer=integer))
    return FeatureDomains(features=features, reference=X.iloc[[0]].copy())


def _probe_frame(reference: pd.DataFrame, assignments) -> pd.DataFrame:
    """Reference row repeated once per (column, value) assignment, with that single cell replaced."""
    frame = reference.loc[reference.index.repeat(len(assignments))].reset_index(drop=True)
    by_col: dict = {}
    for row, (col, value) in enumerate(assignments):
        by_col.setdefault(col, []).append((row, value))
    for col, items in by_col.items():
        rows = [r for r, _ in items]
        values = [v for _, v in items]
        if isinstance(frame[col].dtype, pd.CategoricalDtype):
            frame[col] = frame[col].astype(object)
            frame.loc[rows, col] = values
            frame[col] = pd.Categorical(frame[col], categories=reference[col].cat.categories)
        else:
            if frame[col].dtype != object and any(isinstance(v, float) for v in values):
                frame[col] = frame[col].astype(float)
            frame.loc[rows, col] = values
    return frame


class ModelProbe:
    """Model input columns as functions of the original features (built lazily, cached on the estimator)."""

    def __init__(self, estimator):
        self.estimator = estimator
        domains = estimator.feature_domains_
        self.domains = domains.features
        self.reference = domains.reference
        # With missing_values=True missing values are routed inside the trees, not mapped to a value.
        include_nan = not estimator.missing_values
        self.probe_values = []
        for f in self.domains:
            values = list(f.values)
            if f.has_nan and include_nan:
                values.append(np.nan)
            self.probe_values.append(values)
        self.has_nan_probe = [f.has_nan and include_nan for f in self.domains]

        ref_inputs = estimator._model_inputs(self.reference)[0]
        self.n_columns = ref_inputs.shape[0]
        self.const_value = ref_inputs
        # candidate columns per feature: (columns, values over the feature's probes, variation of each column)
        self._candidates: list = [None] * len(self.domains)

        chunk, rows = [], 0
        for j, values in enumerate(self.probe_values):
            chunk.append(j)
            rows += len(values)
            if rows >= _CHUNK_ROWS:
                self._probe_chunk(chunk)
                chunk, rows = [], 0
        if chunk:
            self._probe_chunk(chunk)
        self._assign_columns()

    def _probe_chunk(self, features):
        assignments = []
        for j in features:
            name = self.domains[j].name
            assignments.extend((name, v) for v in self.probe_values[j])
        # the reference row is evaluated in the same batch to compare like with like
        frame = pd.concat([self.reference.reset_index(drop=True), _probe_frame(self.reference, assignments)], ignore_index=True)
        Z = self.estimator._model_inputs(frame)
        ref, Z = Z[0], Z[1:]
        start = 0
        for j in features:
            n = len(self.probe_values[j])
            Zj = Z[start:start + n]
            start += n
            diff = np.abs(Zj - ref[None, :])
            both_nan = np.isnan(Zj) & np.isnan(ref[None, :])
            variation = np.max(np.where(both_nan, 0.0, np.nan_to_num(diff, nan=np.inf)), axis=0)
            cols = np.flatnonzero(variation > 1e-6 * (1.0 + np.abs(np.nan_to_num(ref))))
            self._candidates[j] = (cols, Zj[:, cols], variation[cols])

    def _assign_columns(self):
        """Attribute every model input to the feature that changes it the most.

        Each input depends on exactly one feature by construction; tiny changes caused by other features are
        numerical noise of batched GPU arithmetic (e.g. TF32 matrix products in the embeddings) and are ignored.
        """
        best = np.zeros(self.n_columns)
        second = np.zeros(self.n_columns)
        self.col_feature = np.full(self.n_columns, -1, dtype=np.int64)
        for j, (cols, _, variation) in enumerate(self._candidates):
            better = variation > best[cols]
            second[cols] = np.where(better, best[cols], np.maximum(second[cols], variation))
            best[cols] = np.where(better, variation, best[cols])
            self.col_feature[cols[better]] = j
        ambiguous = (self.col_feature >= 0) & (second > 1e-2 * best)
        if ambiguous.any():
            warnings.warn(f"Model inputs {np.flatnonzero(ambiguous).tolist()} depend on several features; they are "
                          "attributed to the feature with the largest effect.", RuntimeWarning, stacklevel=3)
        self.col_local = np.full(self.n_columns, -1, dtype=np.int64)
        self.block_values = [None] * len(self.domains)
        self.block_columns = [None] * len(self.domains)
        for j, (cols, values, _) in enumerate(self._candidates):
            keep = self.col_feature[cols] == j
            self.block_columns[j] = cols[keep]
            self.block_values[j] = values[:, keep]
            self.col_local[cols[keep]] = np.arange(int(keep.sum()))
        self._candidates = None

    # ----------------------------------------------------------------------------------------------
    def column_values(self, column: int) -> np.ndarray:
        """Values of a model input column over the probe values of its feature."""
        j = self.col_feature[column]
        return self.block_values[j][:, self.col_local[column]]

    def left_mask(self, column: int, threshold: float) -> np.ndarray:
        """Probe values of the column's feature that go to the left ("yes") child of a split."""
        values = self.column_values(column)
        with np.errstate(invalid="ignore"):
            return values < threshold

    def column_at(self, requests) -> np.ndarray:
        """Evaluate model input columns at arbitrary raw values: ``requests`` = [(feature, column, raw_value)]."""
        assignments = [(self.domains[j].name, v) for j, _, v in requests]
        Z = self.estimator._model_inputs(_probe_frame(self.reference, assignments))
        return Z[np.arange(len(requests)), [c for _, c, _ in requests]]

    def refine_thresholds(self, requests, n_iter: int = 40) -> np.ndarray:
        """Raw value where a split flips between two probe values of a numeric feature (bisection).

        ``requests`` = [(feature, column, threshold, lo_value, hi_value, lo_goes_left)]; returns the smallest
        value (up to float precision) whose branch differs from ``lo_value``'s.
        """
        if not requests:
            return np.zeros(0)
        lo = np.array([r[3] for r in requests], dtype=float)
        hi = np.array([r[4] for r in requests], dtype=float)
        thr = np.array([r[2] for r in requests], dtype=float)
        lo_left = np.array([r[5] for r in requests], dtype=bool)
        for _ in range(n_iter):
            mid = 0.5 * (lo + hi)
            vals = self.column_at([(j, c, m) for (j, c, *_), m in zip(requests, mid, strict=True)])
            same = (vals < thr) == lo_left
            lo = np.where(same, mid, lo)
            hi = np.where(same, hi, mid)
        return hi
