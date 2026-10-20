"""Readable rules from the leaves of the ensemble.

Every reachable leaf of a (pruned) tree is a rule: the conjunction of the split conditions on the path,
merged per feature into raw-value intervals or category sets. On data X, a leaf has

* a *weight share*: the instance-wise estimator weight the ensemble puts on the leaf, summed over the
  samples that reach it and divided by the number of samples (the shares of all leaves add up to 1), and
* a *lift*: how much the ensemble's mean prediction on the covered samples differs from its mean
  prediction on all of X. For classification, the rule's class is the class whose predicted probability
  increases most on the covered samples, and the lift is that increase.

Rules are ranked by their *impact*, weight share times |lift|: how strongly the ensemble relies on a leaf that
singles out samples with a different prediction. Broad rules that hold for almost all samples (e.g. constant
trees, majority-class leaves) have little lift and rank low. Leaves of different trees with the same
predicted class whose coverage on X overlaps almost completely are merged into one rule.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .tree import _INF, _fmt_num, _fmt_value

_FULL = [(-_INF, _INF)]


def _intersect(a, b):
    out = []
    for lo1, hi1 in a:
        for lo2, hi2 in b:
            lo, hi = max(lo1, lo2), min(hi1, hi2)
            if lo < hi:
                out.append((lo, hi))
    return sorted(out)


def _complement(intervals):
    out, start = [], -_INF
    for lo, hi in sorted(intervals):
        if lo > start:
            out.append((start, lo))
        start = max(start, hi)
    if start < _INF:
        out.append((start, _INF))
    return out


def _leaf_path(tree, leaf: int):
    """[(node, went_left)] from the root to ``leaf`` in a (pruned) tree."""
    path = []

    def rec(node):
        if node.is_leaf:
            return node.leaf == leaf
        for child, left in ((node.left, True), (node.right, False)):
            path.append((node, left))
            if rec(child):
                return True
            path.pop()
        return False

    return path if rec(tree.root) else None


def rule_conditions(tree, leaf: int, domains) -> list:
    """Per feature on the path: (feature, kind, region) with region = intervals or a value list, plus missing."""
    regions: dict = {}
    for node, left in _leaf_path(tree, leaf) or []:
        cond = node.condition
        if cond is None or cond.kind == "constant" or cond.feature < 0:
            continue
        j = cond.feature
        domain = domains[j]
        region = regions.setdefault(j, {"kind": cond.kind, "intervals": list(_FULL), "values": None,
                                        "missing": domain.has_nan})
        if cond.kind == "numeric":
            part = cond.intervals or []
            region["intervals"] = _intersect(region["intervals"], part if left else _complement(part))
        else:
            values = set(cond.left_values if left else cond.right_values)
            region["values"] = values if region["values"] is None else region["values"] & values
        if cond.nan_left is not None:
            region["missing"] = region["missing"] and (cond.nan_left if left else not cond.nan_left)
        else:
            region["missing"] = False
    return [(j, r) for j, r in sorted(regions.items())]


def _fmt_interval(lo: float, hi: float, decimals: int) -> str:
    return f"{'(' if lo == -_INF else '['}{_fmt_num(lo, decimals)}, {_fmt_num(hi, decimals)})"


def format_rule(conditions, feature_names, domains, decimals: int = 4, max_values: int = 4) -> str:
    parts = []
    for j, r in conditions:
        name = feature_names[j]
        if r["kind"] == "numeric":
            iv = r["intervals"]
            if not iv:
                text = f"{name} missing" if r["missing"] else f"{name}: (empty)"
                parts.append(text)
                continue
            if iv == _FULL:
                text = f"{name} not missing"
            elif len(iv) == 1:
                lo, hi = iv[0]
                if lo == -_INF:
                    text = f"{name} < {_fmt_num(hi, decimals)}"
                elif hi == _INF:
                    text = f"{name} ≥ {_fmt_num(lo, decimals)}"
                else:
                    text = f"{_fmt_num(lo, decimals)} ≤ {name} < {_fmt_num(hi, decimals)}"
            else:
                text = f"{name} ∈ " + " ∪ ".join(_fmt_interval(lo, hi, decimals) for lo, hi in iv)
            if r["missing"] and iv != _FULL:
                text += " or missing"
        else:
            all_values = list(domains[j].values)
            inside = [v for v in all_values if v in (r["values"] or set())]
            outside = [v for v in all_values if v not in (r["values"] or set())]
            use_in = len(inside) <= len(outside) or not outside
            vals = inside if use_in else outside
            shown = ", ".join(_fmt_value(v, decimals) for v in vals[:max_values])
            if len(vals) > max_values:
                shown += f", … (+{len(vals) - max_values})"
            if len(vals) == 1:
                text = f"{name} {'=' if use_in else '≠'} {shown}"
            else:
                text = f"{name} {'∈' if use_in else '∉'} {{{shown}}}"
            if r["missing"]:
                text += " or missing"
        parts.append(text)
    return " AND ".join(parts) if parts else "(all samples)"


def _jaccard(kept: np.ndarray, mask: np.ndarray) -> np.ndarray:
    inter = kept @ mask.astype(np.float64)
    union = kept.sum(axis=1) + mask.sum() - inter
    return np.where(union > 0, inter / np.maximum(union, 1), 1.0)


def extract_rules(estimator, inspector, X_frame: pd.DataFrame, y=None, top_k: int = 20, sort_by: str = "impact",
                  min_coverage: float = 0.01, merge_similarity: float = 0.9, n_candidates: int = 2000,
                  feature_names=None, decimals: int = 4) -> pd.DataFrame:
    if sort_by not in ("impact", "weight_share", "coverage"):
        raise ValueError(f"sort_by must be 'impact', 'weight_share' or 'coverage', got {sort_by!r}.")
    Z = estimator._model_inputs(X_frame)
    leaves, weights = inspector.routing(Z)
    n, E = leaves.shape
    L = inspector.params.n_leaves
    flat = leaves + np.arange(E)[None, :] * L
    share = np.bincount(flat.ravel(), weights=weights.ravel(), minlength=E * L) / n
    objective = estimator.objective_
    classification = objective in ("binary", "multiclass")
    if classification:
        proba = estimator.predict_proba(X_frame)
        base = proba.mean(axis=0)
        scale = 1.0
    else:
        pred = estimator.predict(X_frame)
        base = float(pred.mean())
        scale = float(pred.std()) or 1.0
    y_arr = None if y is None else np.asarray(y.to_numpy() if hasattr(y, "to_numpy") else y).reshape(-1)

    order = np.argsort(-share)
    order = order[share[order] > 0][:n_candidates]
    groups = []  # dicts with members, share, covered, label_idx
    kept_masks = {}  # label -> (matrix of kept masks, list of group indices)
    for gid in order:
        e, leaf = divmod(int(gid), L)
        covered = leaves[:, e] == leaf
        coverage = covered.mean()
        if coverage < min_coverage or coverage == 1.0:
            continue
        # classification: the class whose predicted probability rises most on the covered samples
        label_idx = int(np.argmax(proba[covered].mean(axis=0) - base)) if classification else None
        km = kept_masks.get(label_idx)
        if km is not None:
            sim = _jaccard(km[0], covered)
            best = int(np.argmax(sim))
            if sim[best] >= merge_similarity:
                group = groups[km[1][best]]
                group["members"].append((e, leaf))
                group["share"] += share[gid]
                continue
        groups.append(dict(members=[(e, leaf)], share=share[gid], covered=covered, label_idx=label_idx))
        row = covered[None, :].astype(np.float64)
        kept_masks[label_idx] = (row, [len(groups) - 1]) if km is None else (np.vstack([km[0], row]), km[1] + [len(groups) - 1])

    for g in groups:
        c = g["covered"]
        if classification:
            g["mean_probability"] = float(proba[c, g["label_idx"]].mean())
            g["lift"] = g["mean_probability"] - float(base[g["label_idx"]])
        else:
            g["mean_prediction"] = float(pred[c].mean())
            g["lift"] = g["mean_prediction"] - base
        g["impact"] = g["share"] * abs(g["lift"]) / scale
        g["coverage"] = float(c.mean())
    key = {"impact": "impact", "weight_share": "share", "coverage": "coverage"}[sort_by]
    ranked = sorted(groups, key=lambda g: -g[key])[:top_k]

    names = estimator._feature_names(feature_names)
    domains = inspector.probe.domains
    trees, rows = {}, []
    for g in ranked:
        e, leaf = g["members"][0]  # the member with the largest weight share
        if e not in trees:
            trees[e] = estimator.get_tree(e, prune=True, feature_names=names)
        covered = g["covered"]
        row = {"rule": format_rule(rule_conditions(trees[e], leaf, domains), names, domains, decimals=decimals)}
        if classification:
            label = estimator.classes_[g["label_idx"]]
            row["prediction"] = label.item() if isinstance(label, np.generic) else label
            row["mean_probability"] = g["mean_probability"]
            if y_arr is not None:
                row["precision"] = float(np.mean(y_arr[covered] == label))
        else:
            row["mean_prediction"] = g["mean_prediction"]
            if y_arr is not None:
                row["mean_target"] = float(np.mean(y_arr[covered]))
        row.update(lift=g["lift"], coverage=g["coverage"], weight_share=g["share"], impact=g["impact"],
                   n_trees=len(g["members"]), tree=e, leaf=leaf)
        rows.append(row)
    out = pd.DataFrame(rows)
    out.attrs["base"] = {str(c): float(b) for c, b in zip(estimator.classes_, base, strict=True)} if classification else base
    return out
