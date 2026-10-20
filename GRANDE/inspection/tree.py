"""Extraction, pruning and text export of the individual trees of a fitted GRANDE ensemble.

Each internal node of a GRANDE tree compares one model input with a learned threshold and sends a
sample to the left ("yes") child if ``input < threshold``. Model inputs are preprocessed features
(see :mod:`GRANDE.inspection._probe`), so every split is translated back to a condition on one
original feature: a set of categories, or intervals of the raw feature values.

Pruning removes *redundant* splits, whose outcome is fixed for all training-range values that can
reach the node: thresholds outside the range of the feature, conditions that contradict earlier
splits on the path (the branch is unreachable), and conditions that earlier splits already imply
(tautologies). Such a split is replaced by the child that is always taken; predictions are unchanged
for all values within the training range.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

_INF = float("inf")


# --------------------------------------------------------------------------------------------------
# Model parameters of the trees
# --------------------------------------------------------------------------------------------------
@dataclass
class TreeParams:
    depth: int
    n_internal: int
    n_leaves: int
    columns: np.ndarray  # (E, I) model input column of each split
    thresholds: np.ndarray  # (E, I)
    leaf_values: np.ndarray  # (E, L) or (E, L, C): raw leaf values (logits for classification)
    leaf_weights: np.ndarray  # (E, L): estimator weight logits per leaf
    path_nodes: np.ndarray  # (L, depth): internal nodes on the path to each leaf
    path_dirs: np.ndarray  # (L, depth): 0 = left ("yes"), 1 = right

    @classmethod
    def from_module(cls, module) -> TreeParams:
        sia = module.split_index_array.detach().float().cpu().numpy()
        selected = sia.argmax(axis=-1)  # argmax of the softmax == argmax of the logits
        features = module.features_by_estimator.detach().cpu().numpy()
        split_values = module.split_values.detach().float().cpu().numpy()
        E, I = selected.shape
        rows = np.arange(E)[:, None]
        return cls(
            depth=int(module.depth),
            n_internal=I,
            n_leaves=I + 1,
            columns=features[rows, selected],
            thresholds=split_values[rows, np.arange(I)[None, :], selected],
            leaf_values=module.leaf_classes_array.detach().float().cpu().numpy(),
            leaf_weights=module.estimator_weights.detach().float().cpu().numpy(),
            path_nodes=module.internal_node_index_list.detach().cpu().numpy(),
            path_dirs=module.path_identifier_list.detach().cpu().numpy(),
        )


def leaf_prediction(objective: str, value, y_mean=0.0, y_std=1.0):
    """Prediction of a single leaf: class probabilities, or the value on the target scale (regression)."""
    if objective == "binary":
        p1 = 1.0 / (1.0 + np.exp(-2.0 * float(value)))
        return np.array([1.0 - p1, p1])
    if objective == "multiclass":
        v = np.asarray(value, dtype=float)
        e = np.exp(v - v.max())
        return e / e.sum()
    return float(value) * float(y_std) + float(y_mean)


# --------------------------------------------------------------------------------------------------
# Conditions and nodes
# --------------------------------------------------------------------------------------------------
@dataclass
class Condition:
    """Condition for the left ("yes") branch of a split, on one original feature.

    In pruned trees only values that can reach the node are taken into account (earlier splits on the
    path restrict the feasible values), so the condition is as simple as possible in the context of the
    path. In complete (unpruned) trees the condition covers the full training range.
    """

    feature: int  # original feature index; -1 for a model input that is constant on the training data
    kind: str  # "numeric" | "discrete" | "categorical" | "constant"
    intervals: list | None = None  # numeric: [(lo, hi)] raw-value intervals [lo, hi) that go left
    left_values: list | None = None  # discrete/categorical: feasible values that go left
    right_values: list | None = None  # discrete/categorical: feasible values that go right
    nan_left: bool | None = None  # where missing values go (if the training data contained missing values)
    always: str | None = None  # "yes" / "no" if the outcome is fixed (redundant split)
    _pending: list = field(default_factory=list, repr=False)

    def text(self, name: str, decimals: int = 4, max_values: int = 4) -> str:
        if self.kind == "constant":
            core = f"{name} (constant input)"
        elif self.kind == "numeric":
            core = _format_intervals(name, self.intervals or [], decimals)
        else:
            core = _format_set(name, self.left_values or [], self.right_values or [], self.kind, decimals, max_values)
        if self.nan_left is not None:
            core += f"  [missing: {'yes' if self.nan_left else 'no'}]"
        if self.always is not None:
            core += f"  (always {self.always})"
        return core

    def evaluate(self, values) -> np.ndarray:
        """Whether raw feature values go to the left branch (numeric / discrete / categorical conditions)."""
        values = np.asarray(values, dtype=object)
        isnan = np.array([_is_nan(v) for v in values])
        out = np.zeros(len(values), dtype=bool)
        if self.kind == "numeric":
            x = np.array([np.nan if n else float(v) for v, n in zip(values, isnan, strict=True)])
            for lo, hi in self.intervals or []:
                with np.errstate(invalid="ignore"):
                    out |= (x >= lo) & (x < hi)
        elif self.kind in ("discrete", "categorical"):
            left = set(self.left_values or [])
            out = np.array([(not n) and (v in left) for v, n in zip(values, isnan, strict=True)])
        if self.nan_left is not None:
            out[isnan] = self.nan_left
        return out


@dataclass
class TreeNode:
    node_id: int  # heap index in the complete tree (root 0, children 2n+1 / 2n+2, leaves after the internal nodes)
    depth: int  # depth in the (possibly pruned) displayed tree
    # split
    column: int | None = None
    threshold: float | None = None
    feature: int | None = None
    condition: Condition | None = None
    left: TreeNode | None = None  # condition true ("yes")
    right: TreeNode | None = None  # condition false ("no")
    redundant: bool = False  # outcome fixed for all feasible values (only kept when not pruned)
    # leaf
    leaf: int | None = None
    value: object = None  # class probabilities (classification) or prediction (regression)
    raw_value: object = None  # raw leaf value (logit(s) for classification)
    weight: float | None = None  # estimator weight logit of the leaf
    # status
    reachable: bool = True
    n_samples: int | None = None

    @property
    def is_leaf(self) -> bool:
        return self.leaf is not None

    def children(self):
        return [c for c in (self.left, self.right) if c is not None]


@dataclass
class GRANDETree:
    """One tree of a fitted GRANDE ensemble (see :meth:`GRANDEClassifier.get_tree`)."""

    index: int
    root: TreeNode
    objective: str
    classes: list | None
    feature_names: list
    depth: int  # depth of the complete tree
    pruned: bool

    def nodes(self):
        stack = [self.root]
        while stack:
            node = stack.pop()
            yield node
            stack.extend(reversed(node.children()))

    def leaves(self):
        return [n for n in self.nodes() if n.is_leaf]

    def splits(self):
        return [n for n in self.nodes() if not n.is_leaf]

    @property
    def n_splits(self) -> int:
        return sum(1 for n in self.splits() if n.reachable and not n.redundant)

    @property
    def max_depth(self) -> int:
        return max(n.depth for n in self.nodes())

    def node_text(self, node: TreeNode, decimals: int = 4) -> str:
        name = self.feature_names[node.feature] if node.feature is not None and node.feature >= 0 else f"input {node.column}"
        return node.condition.text(name, decimals) if node.condition is not None else f"{name} < {node.threshold:.{decimals}g}"

    def leaf_text(self, node: TreeNode, decimals: int = 3, max_classes: int = 4) -> str:
        if self.objective == "regression":
            pred = f"value={format_number(node.value, decimals + 1)}"
        else:
            probs = np.asarray(node.value)
            order = np.argsort(-probs)[:max_classes] if len(probs) > max_classes else range(len(probs))
            pred = "  ".join(f"P({self.classes[k]})={probs[k]:.{decimals}f}" for k in order)
        return f"leaf {node.leaf}: {pred}  | weight={node.weight:.{decimals}f}"

    def to_text(self, decimals: int = 4) -> str:
        n_total = 2 ** self.depth - 1
        lines = [f"Tree {self.index}: {self.n_splits} splits ({n_total} before pruning), "
                 f"{len(self.leaves())} leaves, depth {self.max_depth}" if self.pruned else
                 f"Tree {self.index}: {n_total} splits, {self.n_splits} of them effective, depth {self.depth}"]

        def rec(node, prefix, label):
            status = "" if node.reachable else "  (unreachable)"
            count = f"  [n={node.n_samples}]" if node.n_samples is not None else ""
            body = self.leaf_text(node) if node.is_leaf else self.node_text(node, decimals)
            lines.append(f"{prefix}{label}{body}{count}{status}")
            if node.is_leaf:
                return
            child_prefix = prefix.replace("├─ ", "│  ").replace("└─ ", "   ")
            rec(node.left, child_prefix + "├─ ", "yes: ")
            rec(node.right, child_prefix + "└─ ", "no:  ")

        rec(self.root, "", "")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.to_text()


# --------------------------------------------------------------------------------------------------
# Building / pruning
# --------------------------------------------------------------------------------------------------
def build_tree(inspector, index: int, prune: bool = True, feature_names=None, exact_thresholds: bool = True,
               X_inputs: np.ndarray | None = None) -> GRANDETree:
    """Build the (pruned) tree ``index`` with conditions on the original features."""
    est = inspector.estimator
    root = _traverse(inspector, index, prune=prune, with_conditions=True)
    conditions = [n.condition for n in _iter_nodes(root) if n.condition is not None]
    _resolve_numeric_conditions(inspector, conditions, exact_thresholds)
    tree = GRANDETree(
        index=index,
        root=root,
        objective=est.objective_,
        classes=[str(c) for c in est.classes_] if hasattr(est, "classes_") else None,
        feature_names=list(feature_names),
        depth=inspector.params.depth,
        pruned=prune,
    )
    if X_inputs is not None:
        _count_samples(inspector, tree, X_inputs)
    return tree


def analyze_tree(inspector, index: int) -> dict:
    """Redundancy / reachability of every split of tree ``index`` (fast, without display conditions)."""
    I = inspector.params.n_internal
    redundant = np.zeros(I, dtype=bool)
    reachable = np.zeros(I, dtype=bool)
    leaf_reachable = np.zeros(I + 1, dtype=bool)
    feature = np.full(I, -1, dtype=np.int64)
    pruned_depth = np.full(I, -1, dtype=np.int64)  # depth of each effective split in the pruned tree
    leaf_pruned_depth = np.full(I + 1, -1, dtype=np.int64)  # depth of each reachable leaf in the pruned tree
    for node in _iter_nodes(_traverse(inspector, index, prune=False, with_conditions=False)):
        if node.is_leaf:
            leaf_reachable[node.leaf] = node.reachable
            continue
        redundant[node.node_id] = node.redundant
        reachable[node.node_id] = node.reachable
        feature[node.node_id] = node.feature
    for node in _iter_nodes(_traverse(inspector, index, prune=True, with_conditions=False)):
        if node.is_leaf:
            leaf_pruned_depth[node.leaf] = node.depth
        else:
            pruned_depth[node.node_id] = node.depth
    return dict(redundant=redundant, reachable=reachable, leaf_reachable=leaf_reachable, feature=feature,
                pruned_depth=pruned_depth, leaf_pruned_depth=leaf_pruned_depth)


def _iter_nodes(root):
    stack = [root]
    while stack:
        node = stack.pop()
        yield node
        stack.extend(node.children())


def _traverse(inspector, e: int, prune: bool, with_conditions: bool) -> TreeNode:
    probe, params, est = inspector.probe, inspector.params, inspector.estimator
    I = params.n_internal
    objective = est.objective_
    y_mean, y_std = getattr(est, "y_mean_", 0.0), getattr(est, "y_std_", 1.0)
    full = {}

    def feasible_of(fd, j):
        if j in fd:
            return fd[j]
        if j not in full:
            full[j] = np.ones(len(probe.probe_values[j]), dtype=bool)
        return full[j]

    def rec(node, fd, depth, reachable):
        if node >= I:
            leaf = node - I
            raw = params.leaf_values[e, leaf]
            return TreeNode(node_id=node, depth=depth, leaf=leaf, raw_value=raw,
                            value=leaf_prediction(objective, raw, y_mean, y_std),
                            weight=float(params.leaf_weights[e, leaf]), reachable=reachable)
        c = int(params.columns[e, node])
        t = float(params.thresholds[e, node])
        j = int(probe.col_feature[c])
        if j < 0:
            goes_left = bool(probe.const_value[c] < t)
            cur = mask = None
            can_left, can_right = goes_left, not goes_left
        else:
            cur = feasible_of(fd, j)
            mask = probe.left_mask(c, t)
            fl, fr = cur & mask, cur & ~mask
            can_left, can_right = bool(fl.any()), bool(fr.any())
        if prune and reachable and not (can_left and can_right) and (can_left or can_right):
            return rec(2 * node + 1 if can_left else 2 * node + 2, fd, depth, reachable)

        out = TreeNode(node_id=node, depth=depth, column=c, threshold=t, feature=j,
                       redundant=not (can_left and can_right), reachable=reachable)
        if with_conditions:
            # pruned trees state conditions in the context of the path (only values that can reach the node);
            # the complete tree shows every split as learned, on the full training range
            context = cur if (prune or j < 0) else feasible_of({}, j)
            out.condition = _make_condition(probe, j, c, t, context, mask, can_left, can_right, reachable)
        fdl, fdr = dict(fd), dict(fd)
        if j >= 0:
            fdl[j], fdr[j] = fl, fr
        out.left = rec(2 * node + 1, fdl, depth + 1, reachable and can_left)
        out.right = rec(2 * node + 2, fdr, depth + 1, reachable and can_right)
        return out

    return rec(0, {}, 0, True)


def _make_condition(probe, j, c, t, cur, mask, can_left, can_right, reachable=True) -> Condition:
    always = None if (can_left and can_right) or not reachable else ("yes" if can_left else "no")
    if j < 0:
        return Condition(feature=-1, kind="constant", always=always)
    domain = probe.domains[j]
    n_values = len(domain.values)
    nan_left = None
    if probe.has_nan_probe[j] and cur[n_values]:
        nan_left = bool(mask[n_values])
    if not cur[:n_values].any():  # only missing values (or nothing) reach the node
        return Condition(feature=j, kind=domain.kind if domain.kind != "numeric" else "numeric", intervals=[],
                         left_values=[], right_values=[], nan_left=nan_left, always=always)
    idx = np.flatnonzero(cur[:n_values])
    if domain.kind == "numeric":
        cond = Condition(feature=j, kind="numeric", nan_left=nan_left, always=always)
        m = mask[idx]
        values = domain.values[idx]
        # runs of consecutive feasible probe values that go left -> raw intervals; boundaries resolved later
        runs, start = [], None
        for pos, left in enumerate(m):
            if left and start is None:
                start = pos
            if not left and start is not None:
                runs.append((start, pos - 1))
                start = None
        if start is not None:
            runs.append((start, len(m) - 1))
        intervals = []
        for a, b in runs:
            lo = -_INF if a == 0 else ("pending", len(cond._pending))
            if a > 0:
                cond._pending.append((j, c, t, float(values[a - 1]), float(values[a]), False))
            hi = _INF if b == len(m) - 1 else ("pending", len(cond._pending))
            if b < len(m) - 1:
                cond._pending.append((j, c, t, float(values[b]), float(values[b + 1]), True))
            intervals.append([lo, hi])
        cond.intervals = intervals
        return cond
    left_values = [domain.values[k] for k in idx if mask[k]]
    right_values = [domain.values[k] for k in idx if not mask[k]]
    return Condition(feature=j, kind=domain.kind, left_values=left_values, right_values=right_values,
                     nan_left=nan_left, always=always)


def _resolve_numeric_conditions(inspector, conditions, exact: bool):
    requests, owners = [], []
    for cond in conditions:
        for k, req in enumerate(cond._pending):
            requests.append(req)
            owners.append((cond, k))
    if exact and requests:
        boundaries = inspector.probe.refine_thresholds(requests)
    else:
        boundaries = np.array([0.5 * (r[3] + r[4]) for r in requests])
    domains = inspector.probe.domains
    resolved = {}
    for (cond, k), b, req in zip(owners, boundaries, requests, strict=True):
        if domains[req[0]].integer:
            # integer-valued feature: x < b <=> x < ceil(b) for every integer x (tolerance for the bisection)
            b = np.ceil(b - 1e-9 * max(1.0, abs(b)))
        resolved[(id(cond), k)] = float(b)
    for cond in conditions:
        if cond.intervals is None:
            continue
        cond.intervals = [
            (resolved[(id(cond), lo[1])] if isinstance(lo, tuple) else lo,
             resolved[(id(cond), hi[1])] if isinstance(hi, tuple) else hi)
            for lo, hi in cond.intervals
        ]
        cond._pending = []


def _count_samples(inspector, tree: GRANDETree, X_inputs: np.ndarray):
    """Number of samples (given as model inputs) that reach each node of the displayed tree."""
    params = inspector.params
    leaves = inspector.leaf_indices(X_inputs, estimators=[tree.index])[:, 0]
    # internal nodes on each sample's path in the complete tree
    path = params.path_nodes[leaves]  # (n, depth)
    counts_internal = np.bincount(path.ravel(), minlength=params.n_internal)
    counts_leaf = np.bincount(leaves, minlength=params.n_leaves)

    def rec(node):
        if node.is_leaf:
            node.n_samples = int(counts_leaf[node.leaf])
            return node.n_samples
        node.n_samples = int(counts_internal[node.node_id])
        for child in node.children():
            rec(child)
        return node.n_samples

    rec(tree.root)


# --------------------------------------------------------------------------------------------------
# Formatting
# --------------------------------------------------------------------------------------------------
def _is_nan(v) -> bool:
    try:
        return v is None or bool(np.isnan(v))
    except (TypeError, ValueError):
        return False


def _fmt_num(v: float, decimals: int) -> str:
    if v == _INF:
        return "∞"
    if v == -_INF:
        return "-∞"
    return format_number(v, decimals)


def _format_intervals(name: str, intervals, decimals: int) -> str:
    if not intervals:
        return f"{name}: threshold outside data range"
    if len(intervals) == 1:
        lo, hi = intervals[0]
        if lo == -_INF and hi == _INF:
            return f"{name}: threshold outside data range"
        if lo == -_INF:
            return f"{name} < {_fmt_num(hi, decimals)}"
        if hi == _INF:
            return f"{name} ≥ {_fmt_num(lo, decimals)}"
        return f"{_fmt_num(lo, decimals)} ≤ {name} < {_fmt_num(hi, decimals)}"
    parts = " ∪ ".join(f"{'(' if lo == -_INF else '['}{_fmt_num(lo, decimals)}, {_fmt_num(hi, decimals)})" for lo, hi in intervals)
    return f"{name} ∈ {parts}"


def format_number(v: float, decimals: int = 4) -> str:
    """Compact number: thousands separators for large values, significant digits otherwise."""
    v = float(v)
    if abs(v) >= 10 ** decimals and abs(v) < 1e15:
        return f"{v:,.0f}"
    return f"{v:.{decimals}g}"


def _fmt_value(v, decimals: int) -> str:
    if isinstance(v, (float, np.floating)):
        return f"{v:.{decimals}g}"
    return str(v)


def _format_set(name, left, right, kind, decimals, max_values) -> str:
    if not left or not right:
        return f"{name}: threshold outside data range"
    use_left = len(left) <= len(right)
    values = left if use_left else right
    if kind == "discrete" and len(values) == 1:
        return f"{name} {'=' if use_left else '≠'} {_fmt_value(values[0], decimals)}"
    shown = ", ".join(_fmt_value(v, decimals) for v in values[:max_values])
    if len(values) > max_values:
        shown += f", … (+{len(values) - max_values})"
    return f"{name} {'∈' if use_left else '∉'} {{{shown}}}"
