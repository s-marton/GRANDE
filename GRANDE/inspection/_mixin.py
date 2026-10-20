"""Inspection methods of the GRANDE estimators (trees, pruning, routing)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.utils.validation import check_is_fitted

from ._inspector import Inspector
from .attribution import shapley_values
from .importance import path_importances, structural_importances
from .instance import explain_instance
from .rules import extract_rules
from .tree import GRANDETree, build_tree


class InspectionMixin:
    """Methods to look inside a fitted GRANDE ensemble."""

    # ------------------------------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------------------------------
    def _inspector(self) -> Inspector:
        check_is_fitted(self, "module_")
        if getattr(self, "_inspector_cache", None) is None:
            self._inspector_cache = Inspector(self)
        return self._inspector_cache

    def _feature_names(self, feature_names=None) -> list:
        if feature_names is None:
            return [f"x_{i + 1}" for i in range(self.n_features_in_)]
        names = [str(n) for n in feature_names]
        if len(names) != self.n_features_in_:
            raise ValueError(f"feature_names has {len(names)} entries, but the model has {self.n_features_in_} features.")
        return names

    def _inputs_of(self, X) -> np.ndarray:
        """Model inputs (preprocessed and embedded) of raw data ``X``."""
        return self._model_inputs(self._apply_cat_features(self._to_frame(X)))

    def _single_row(self, x):
        """A single sample as a one-row frame (accepts a one-row DataFrame, a Series or a 1-D array)."""
        if isinstance(x, pd.Series):
            x = x.to_frame().T.reset_index(drop=True)
            x = x.infer_objects()
        elif not isinstance(x, pd.DataFrame):
            x = np.asarray(x, dtype=object if np.asarray(x).dtype == object else None).reshape(1, -1)
            x = pd.DataFrame(x, columns=list(self.feature_names_in_)).infer_objects()
        if len(x) != 1:
            raise ValueError(f"Expected a single sample, got {len(x)} rows.")
        return x

    def _check_estimator_index(self, index: int) -> int:
        n = self._inspector().n_estimators
        if not -n <= index < n:
            raise IndexError(f"estimator index {index} out of range for an ensemble of {n} trees.")
        return index % n

    # ------------------------------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------------------------------
    def get_tree(self, index: int = 0, prune: bool = True, feature_names=None, X=None,
                 exact_thresholds: bool = True) -> GRANDETree:
        """Return tree ``index`` of the ensemble with conditions on the original features.

        Parameters
        ----------
        index : int
            Index of the estimator (tree) in the ensemble.
        prune : bool, default=True
            Remove redundant splits: splits whose outcome is the same for all training-range values that
            reach them (thresholds outside the feature range, contradictions and tautologies with respect
            to earlier splits on the path). Predictions within the training range are unchanged.
        feature_names : list of str, optional
            Names of the input features (default ``x_1, x_2, ...``).
        X : array-like, optional
            Data used to count the samples that reach each node.
        exact_thresholds : bool, default=True
            Compute the exact raw-value thresholds of numeric splits (else midpoints between data values).
        """
        index = self._check_estimator_index(index)
        X_inputs = self._inputs_of(X) if X is not None else None
        return build_tree(self._inspector(), index, prune=prune, feature_names=self._feature_names(feature_names),
                          exact_thresholds=exact_thresholds, X_inputs=X_inputs)

    def export_text(self, index: int = 0, prune: bool = True, feature_names=None, X=None, decimals: int = 4) -> str:
        """Text representation of tree ``index`` (see :meth:`get_tree` for the parameters)."""
        return self.get_tree(index, prune=prune, feature_names=feature_names, X=X).to_text(decimals=decimals)

    def plot_tree(self, index: int = 0, prune: bool = True, feature_names=None, X=None, highlight=None, ax=None,
                  figsize=None, decimals: int = 3, max_classes: int = 4, title=None):
        """Plot tree ``index`` of the ensemble (requires matplotlib).

        Split nodes show the condition of the "yes" branch on the original features; leaves show the class
        distribution (or the regression value) and, in a separate line, the leaf weight.

        Parameters
        ----------
        index : int
            Index of the estimator (tree) in the ensemble.
        prune : bool, default=True
            Remove redundant splits (see :meth:`get_tree`). With ``prune=False`` the complete tree is drawn;
            redundant splits are dashed and unreachable nodes faded.
        feature_names : list of str, optional
            Names of the input features (default ``x_1, x_2, ...``).
        X : array-like, optional
            Data used to show the number of samples reaching each node.
        highlight : sample, optional
            A single sample (one-row DataFrame, Series or 1-D array) whose decision path is highlighted.
        ax : matplotlib Axes, optional
        figsize, decimals, max_classes, title :
            Layout options (``max_classes``: classes shown per leaf, the most probable first).

        Returns
        -------
        matplotlib Axes
        """
        from .plotting import plot_tree

        tree = self.get_tree(index, prune=prune, feature_names=feature_names, X=X)
        highlight_leaf = None
        if highlight is not None:
            highlight_leaf = int(self.apply(self._single_row(highlight))[0, tree.index])
        return plot_tree(tree, ax=ax, decimals=decimals, max_classes=max_classes, highlight_leaf=highlight_leaf,
                         title=title, figsize=figsize)

    def apply(self, X) -> np.ndarray:
        """Index of the leaf that each sample reaches in each tree, shape (n_samples, n_estimators)."""
        leaves, _ = self._inspector().routing(self._inputs_of(X))
        return leaves

    @property
    def feature_importances_(self) -> np.ndarray:
        """Data-free importance of each input feature (sums to 1).

        Every tree distributes its expected estimator weight over its effective (non-redundant) splits,
        in proportion to the probability of reaching each split. See :meth:`compute_feature_importances`
        for the data-based variant.
        """
        return structural_importances(self._inspector())

    def compute_feature_importances(self, X=None, feature_names=None) -> pd.Series:
        """Importance of each input feature, sorted in decreasing order (sums to 1).

        With ``X``, the importance is data-based: for every sample and tree, the sample's instance-wise
        estimator weight is shared among the effective splits on its decision path, so a feature's
        importance is its share of all routing decisions of the ensemble on ``X``. Without ``X``, the
        data-free :attr:`feature_importances_` is returned. Redundant splits (see :meth:`get_tree`) never
        receive credit; encodings and embedding dimensions are aggregated to their original feature.
        """
        insp = self._inspector()
        values = path_importances(insp, self._inputs_of(X)) if X is not None else structural_importances(insp)
        return pd.Series(values, index=self._feature_names(feature_names), name="importance").sort_values(ascending=False)

    def plot_feature_importances(self, X=None, feature_names=None, top_k: int = 20, ax=None, title=None):
        """Bar chart of :meth:`compute_feature_importances` (the ``top_k`` most important features)."""
        from .plotting import plot_feature_importances

        importances = self.compute_feature_importances(X=X, feature_names=feature_names)
        if title is None:
            title = "Feature importance (share of routing decisions on X)" if X is not None else "Feature importance (structure)"
        return plot_feature_importances(importances, top_k=top_k, ax=ax, title=title)

    def explain_instance(self, x) -> pd.DataFrame:
        """Instance-wise estimator weights of one sample, one row per tree, sorted by weight.

        Columns: ``estimator`` (index), ``leaf`` reached, ``weight`` (softmax-normalized over the trees,
        sums to 1), ``weight_logit`` (the leaf weight), ``leaf_prediction`` (the leaf's prediction:
        probability of the positive / predicted class, or the regression value) and ``contribution`` to the
        ensemble output in an additive unit (binary: log-odds of ``classes_[1]``; multiclass: logit of the
        predicted class relative to the mean logit; regression: target units, prediction = mean + sum).
        ``DataFrame.attrs`` holds the prediction, the sum of contributions and the effective number of trees
        (1 / sum of squared weights).
        """
        return explain_instance(self, self._inspector(), self._apply_cat_features(self._to_frame(self._single_row(x))))

    def plot_estimator_weights(self, x, top_k: int = 10, figsize=None, title=None):
        """Plot the estimator weights of one sample: their distribution over all trees, the ``top_k`` trees
        by weight (with their index) and the contributions of these trees (see :meth:`explain_instance`)."""
        from .plotting import plot_estimator_weights

        return plot_estimator_weights(self.explain_instance(x), top_k=top_k, figsize=figsize, title=title)

    def plot_top_estimators(self, x, k: int = 3, prune: bool = True, feature_names=None, figsize=None,
                            decimals: int = 3, max_classes: int = 4):
        """Plot the ``k`` trees with the highest weight for one sample side by side, with the sample's decision
        path highlighted and each tree's weight and contribution in its title."""
        from .plotting import plot_top_estimators

        explanation = self.explain_instance(x)
        rows = explanation.iloc[:k]
        names = self._feature_names(feature_names)
        trees = [self.get_tree(int(e), prune=prune, feature_names=names) for e in rows["estimator"]]
        return plot_top_estimators(trees, rows, unit=explanation.attrs.get("contribution_unit", ""), figsize=figsize,
                                   decimals=decimals, max_classes=max_classes)

    def explain_features(self, x, background, feature_names=None, max_exact_features: int = 12,
                         n_permutations: int = 64, random_state: int = 0) -> pd.DataFrame:
        """Shapley-value attribution of one prediction to the input features, sorted by magnitude.

        Interventional Shapley values with respect to the rows of ``background`` (e.g. 50-200 training rows):
        the attributions add up to the model output for ``x`` minus the mean output over ``background``
        (``attrs["score"] - attrs["base_value"]``), in an additive unit (binary: log-odds of ``classes_[1]``;
        multiclass: logit of the predicted class relative to the mean logit; regression: target units).

        For every background row, only features that some split routes differently for ``x`` and that row can
        change the output; the Shapley values over these features are computed exactly if there are at most
        ``max_exact_features`` of them, and otherwise estimated with ``n_permutations`` antithetic permutations.
        ``attrs`` reports how many background rows were handled exactly / by sampling.
        """
        x_frame = self._apply_cat_features(self._to_frame(self._single_row(x)))
        background = self._apply_cat_features(self._to_frame(background))
        result = shapley_values(self, self._inspector(), x_frame, background, max_exact_features=max_exact_features,
                                n_permutations=n_permutations, random_state=random_state)
        names = dict(zip(x_frame.columns, self._feature_names(feature_names), strict=True))
        result["feature"] = result["feature"].map(names)
        return result

    def plot_feature_attributions(self, x=None, background=None, feature_names=None, top_k: int = 10, ax=None,
                                  title=None, attributions=None, **kwargs):
        """Waterfall chart of :meth:`explain_features`: from the mean output over ``background`` to the prediction.

        Pass the result of :meth:`explain_features` as ``attributions`` to plot it without recomputing.
        """
        from .plotting import plot_feature_attributions

        if attributions is None:
            if x is None or background is None:
                raise ValueError("Pass x and background, or the result of explain_features as attributions.")
            attributions = self.explain_features(x, background, feature_names=feature_names, **kwargs)
        return plot_feature_attributions(attributions, top_k=top_k, ax=ax, title=title)

    def extract_rules(self, X, y=None, top_k: int = 20, sort_by: str = "impact", min_coverage: float = 0.01,
                      merge_similarity: float = 0.9, feature_names=None, decimals: int = 4) -> pd.DataFrame:
        """Readable rules from the leaves the ensemble relies on most for ``X``.

        Every reachable leaf of a pruned tree is a rule: the conditions on its path, merged per feature. For each
        leaf, ``weight_share`` is the instance-wise estimator weight the ensemble puts on it, summed over the covered
        samples of ``X`` and divided by ``len(X)``, and ``lift`` is the difference between the ensemble's mean
        prediction on the covered samples and on all of ``X`` (classification: the rule's class is the class whose
        predicted probability increases most on the covered samples). Rules are ranked by ``impact`` =
        weight share x |lift| (``sort_by="weight_share"`` or ``"coverage"`` are alternatives); rules covering less
        than ``min_coverage`` or all of ``X`` are skipped, and rules with the same class whose coverage on ``X``
        overlaps by at least ``merge_similarity`` (Jaccard) are merged into the one with the largest weight share
        (``n_trees``).

        Columns: ``rule``; classification: ``prediction`` (class of the rule), ``mean_probability`` (the ensemble's
        mean probability of that class on the covered samples) and, with ``y``, ``precision``; regression:
        ``mean_prediction`` and, with ``y``, ``mean_target`` of the covered samples; then ``lift``, ``coverage``,
        ``weight_share``, ``impact``, ``n_trees`` and the ``tree`` / ``leaf`` of the rule. ``attrs["base"]`` holds
        the mean prediction on ``X``.
        """
        X_frame = self._apply_cat_features(self._to_frame(X))
        return extract_rules(self, self._inspector(), X_frame, y=y, top_k=top_k, sort_by=sort_by,
                             min_coverage=min_coverage, merge_similarity=merge_similarity,
                             feature_names=feature_names, decimals=decimals)

    def pruning_statistics(self) -> pd.DataFrame:
        """Per tree: splits of the complete tree, effective splits after pruning, redundant and unreachable splits."""
        insp = self._inspector()
        rows = []
        for e, a in enumerate(insp.analyses()):
            effective = a["reachable"] & ~a["redundant"]
            rows.append(dict(
                estimator=e,
                splits=len(a["redundant"]),
                effective_splits=int(effective.sum()),
                redundant_splits=int((a["reachable"] & a["redundant"]).sum()),
                unreachable_splits=int((~a["reachable"]).sum()),
                reachable_leaves=int(a["leaf_reachable"].sum()),
                depth_after_pruning=int(a["pruned_depth"].max() + 1) if (a["pruned_depth"] >= 0).any() else 0,
            ))
        return pd.DataFrame(rows).set_index("estimator")
