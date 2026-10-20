"""Matplotlib plots for fitted GRANDE ensembles (requires ``pip install "GRANDE[plot]"``)."""

from __future__ import annotations

import textwrap

import numpy as np

from .tree import GRANDETree, TreeNode, format_number

_SPLIT_COLOR = "#f4f4f4"
_EDGE_COLOR = "#7f7f7f"
_HIGHLIGHT = "#d62728"


def _plt():
    try:
        import matplotlib.pyplot as plt
    except ImportError as err:  # pragma: no cover - depends on the environment
        raise ImportError('Plotting requires matplotlib: pip install "GRANDE[plot]"') from err
    return plt


def _class_colors(n_classes: int):
    plt = _plt()
    cmap = plt.get_cmap("tab10" if n_classes <= 10 else "tab20")
    return [cmap(k % cmap.N) for k in range(n_classes)]


def _blend(color, alpha: float):
    """Mix an RGBA color with white (alpha = share of the color)."""
    r, g, b = color[:3]
    return (1 - alpha + alpha * r, 1 - alpha + alpha * g, 1 - alpha + alpha * b)


def _layout(tree: GRANDETree):
    """x/y positions: leaves left to right, internal nodes centered above their children."""
    pos, order = {}, [0]

    def rec(node: TreeNode):
        if node.is_leaf:
            pos[id(node)] = (order[0], -node.depth)
            order[0] += 1
            return pos[id(node)][0]
        xs = [rec(c) for c in node.children()]
        pos[id(node)] = (float(np.mean(xs)), -node.depth)
        return pos[id(node)][0]

    rec(tree.root)
    return pos, order[0]


def _path_to_leaf(tree: GRANDETree, leaf: int):
    path = []

    def rec(node):
        path.append(node)
        if node.is_leaf:
            if node.leaf == leaf:
                return True
        else:
            for child in node.children():
                if rec(child):
                    return True
        path.pop()
        return False

    return path if rec(tree.root) else []


def _leaf_label(tree: GRANDETree, node: TreeNode, decimals: int, max_classes: int, bar_width: int = 8) -> str:
    lines = []
    if tree.objective == "regression":
        lines.append(f"value  {format_number(node.value, decimals + 1)}")
    else:
        probs = np.asarray(node.value)
        order = np.argsort(-probs)[:max_classes] if len(probs) > max_classes else np.arange(len(probs))
        names = [tree.classes[k] for k in order]
        width = min(max(len(n) for n in names), 10)
        for k, name in zip(order, names, strict=True):
            filled = int(round(probs[k] * bar_width))
            bar = "█" * filled + "░" * (bar_width - filled)
            lines.append(f"{name[:width]:<{width}} {bar} {probs[k]:.{decimals - 1}f}")
        if len(probs) > max_classes:
            lines.append(f"(+{len(probs) - max_classes} classes)")
    lines.append(f"weight {node.weight:.{decimals}f}")
    if node.n_samples is not None:
        lines.append(f"n = {node.n_samples}")
    return "\n".join(lines)


def plot_tree(tree: GRANDETree, ax=None, decimals: int = 3, max_classes: int = 4, highlight_leaf: int | None = None,
              title: str | None = None, fontsize: float | None = None, figsize=None, wrap: int = 26):
    """Draw a :class:`~GRANDE.inspection.GRANDETree` (see :meth:`GRANDEClassifier.plot_tree`)."""
    plt = _plt()
    pos, n_leaves = _layout(tree)
    max_depth = tree.max_depth
    if ax is None:
        if figsize is None:
            figsize = (max(6.0, min(1.9 * n_leaves, 1.2 * n_leaves + 8.0)), 1.55 * (max_depth + 1) + 0.6)
        _, ax = plt.subplots(figsize=figsize)
    if fontsize is None:
        fontsize = 9 if n_leaves <= 12 else 8 if n_leaves <= 24 else 7

    highlight = {id(n) for n in _path_to_leaf(tree, highlight_leaf)} if highlight_leaf is not None else set()

    leaves = tree.leaves()
    if tree.objective == "regression":
        values = np.array([leaf.value for leaf in leaves], dtype=float)
        vmin, vmax = float(values.min()), float(values.max())
        cmap = plt.get_cmap("coolwarm")

        def leaf_color(node):
            frac = 0.5 if vmax == vmin else (node.value - vmin) / (vmax - vmin)
            return _blend(cmap(frac), 0.55)
    else:
        colors = _class_colors(len(tree.classes))

        def leaf_color(node):
            probs = np.asarray(node.value)
            k = int(np.argmax(probs))
            strength = (probs[k] - 1.0 / len(probs)) / (1.0 - 1.0 / len(probs))
            return _blend(colors[k], 0.15 + 0.55 * float(np.clip(strength, 0.0, 1.0)))

    # edges
    for node in tree.nodes():
        if node.is_leaf:
            continue
        x0, y0 = pos[id(node)]
        for child, label in ((node.left, "yes"), (node.right, "no")):
            x1, y1 = pos[id(child)]
            on_path = id(node) in highlight and id(child) in highlight
            ax.plot([x0, x1], [y0, y1], color=_HIGHLIGHT if on_path else _EDGE_COLOR,
                    lw=2.8 if on_path else 1.1, ls="-" if child.reachable else ":", zorder=1, alpha=1.0 if child.reachable else 0.5)
            ax.text(x0 + 0.42 * (x1 - x0), y0 + 0.42 * (y1 - y0), label, fontsize=fontsize - 1, color="#555555",
                    ha="center", va="center", style="italic", zorder=3,
                    bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.8))

    # nodes
    for node in tree.nodes():
        x, y = pos[id(node)]
        on_path = id(node) in highlight
        alpha = 1.0 if node.reachable else 0.45
        if node.is_leaf:
            text = _leaf_label(tree, node, decimals, max_classes)
            box = dict(boxstyle="round,pad=0.45", fc=leaf_color(node), ec=_HIGHLIGHT if on_path else "#555555",
                       lw=2.2 if on_path else 0.8, alpha=alpha)
            ax.text(x, y, text, ha="center", va="center", fontsize=fontsize, family="DejaVu Sans Mono", bbox=box,
                    zorder=4, alpha=alpha)
        else:
            parts = tree.node_text(node, decimals + 1).replace("  [", "\n[").replace("  (", "\n(").split("\n")
            text = "\n".join(line for part in parts for line in (textwrap.wrap(part, wrap) or [part])) or "?"
            if node.n_samples is not None:
                text += f"\nn = {node.n_samples}"
            box = dict(boxstyle="round,pad=0.45", fc=_SPLIT_COLOR, ec=_HIGHLIGHT if on_path else "#555555",
                       lw=2.2 if on_path else 0.8, ls="--" if node.redundant else "-", alpha=alpha)
            ax.text(x, y, text, ha="center", va="center", fontsize=fontsize, bbox=box, zorder=4, alpha=alpha)

    ax.set_xlim(-0.65, n_leaves - 0.35)
    ax.set_ylim(-max_depth - 0.75, 0.65)
    ax.axis("off")
    if title is None:
        title = (f"Tree {tree.index}: {tree.n_splits} splits, {len(leaves)} leaves"
                 + ("" if tree.pruned else " (unpruned: dashed = redundant split, faded = unreachable)"))
    if title:
        ax.set_title(title, fontsize=fontsize + 2)
    return ax


def plot_feature_importances(importances, top_k: int = 20, ax=None, title: str | None = None):
    """Horizontal bar chart of a feature-importance Series (index = feature names)."""
    plt = _plt()
    imp = importances.sort_values(ascending=False).iloc[:top_k][::-1]
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 0.35 * len(imp) + 1.2))
    ax.barh(range(len(imp)), imp.values, color="#4c72b0")
    ax.set_yticks(range(len(imp)), [str(i) for i in imp.index])
    for y, v in enumerate(imp.values):
        ax.text(v, y, f" {v:.1%}", va="center", fontsize=8)
    ax.set_xlabel("importance")
    ax.set_xlim(0, imp.values.max() * 1.18 if len(imp) else 1)
    ax.spines[["top", "right"]].set_visible(False)
    if title:
        ax.set_title(title)
    return ax


def plot_estimator_weights(explanation, top_k: int = 10, figsize=None, title: str | None = None):
    """Distribution of the instance-wise estimator weights of one sample, its top-k trees and their contributions."""
    plt = _plt()
    attrs = explanation.attrs
    w = np.sort(explanation["weight"].to_numpy())[::-1]
    n = len(w)
    cumulative = np.cumsum(w)
    k50 = int(np.searchsorted(cumulative, 0.5) + 1)
    k90 = int(np.searchsorted(cumulative, 0.9) + 1)
    top = explanation.iloc[:top_k]
    if figsize is None:
        figsize = (17, max(4.0, 0.42 * len(top) + 1.8))
    fig, axes = plt.subplots(1, 3, figsize=figsize, gridspec_kw=dict(width_ratios=[1.3, 1, 1]))

    ax = axes[0]
    ranks = np.arange(1, n + 1)
    ax.fill_between(ranks, w * 100, step="mid", alpha=0.35, color="#4c72b0")
    ax.plot(ranks, w * 100, color="#4c72b0", lw=1, drawstyle="steps-mid")
    ax.axhline(100.0 / n, ls="--", color="gray", lw=1)
    ax.text(1, 100.0 / n, " uniform 1/n", va="bottom", fontsize=8, color="gray")
    ax.set_xscale("log")
    ax.set_xlabel("trees, ranked by weight")
    ax.set_ylabel("weight (%)")
    twin = ax.twinx()
    twin.plot(ranks, cumulative * 100, color="#dd8452", lw=1.6)
    twin.set_ylabel("cumulative weight (%)", color="#dd8452")
    twin.set_ylim(0, 102)
    ax.set_title(f"Distribution of the estimator weights\neffective number of trees: "
                 f"{attrs.get('effective_n_estimators', np.nan):.0f} of {n}  ·  50% of the weight: {k50} trees  ·  "
                 f"90%: {k90}", fontsize=10)

    labels = [f"tree #{e}" for e in top["estimator"]]
    y = np.arange(len(top))
    ax = axes[1]
    ax.barh(y, top["weight"] * 100, color="#4c72b0")
    for yi, v in zip(y, top["weight"], strict=True):
        ax.text(v * 100, yi, f" {v:.1%}", va="center", fontsize=8)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("weight (%)")
    ax.set_xlim(0, top["weight"].max() * 100 * 1.25)
    ax.set_title(f"Top {len(top)} trees by weight")

    ax = axes[2]
    contrib = top["contribution"].to_numpy()
    ax.barh(y, contrib, color=np.where(contrib >= 0, "#c44e52", "#4c72b0"))
    ax.axvline(0, color="black", lw=0.8)
    for yi, v in zip(y, contrib, strict=True):
        ax.text(v, yi, f" {v:+.3f} " if v >= 0 else f" {v:+.3f} ", va="center", ha="left" if v >= 0 else "right", fontsize=8)
    ax.set_yticks(y, [""] * len(y))
    ax.invert_yaxis()
    span = np.abs(contrib).max() if len(contrib) else 1.0
    ax.set_xlim(-1.35 * span, 1.35 * span)
    ax.xaxis.set_major_locator(plt.MaxNLocator(5, symmetric=True))
    ax.set_xlabel(f"contribution ({attrs.get('contribution_unit', '')})")
    ax.set_title(f"Their contributions (top {len(top)}: {contrib.sum():+.3f} of {attrs.get('score', np.nan):+.3f})")
    for a in axes[1:]:
        a.spines[["top", "right"]].set_visible(False)

    if title is None:
        title = (f"{attrs.get('prediction_label', 'prediction')} = {attrs.get('prediction', np.nan):.4g}   |   "
                 f"sum of all {n} contributions: {attrs.get('score', np.nan):+.4g} ({attrs.get('contribution_unit', '')})")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    return fig


def plot_top_estimators(trees, rows, unit: str, figsize=None, decimals: int = 3, max_classes: int = 4):
    """Trees side by side with the decision path of one sample highlighted (rows: explanation rows)."""
    plt = _plt()
    widths = [max(2, _layout(t)[1]) for t in trees]
    depth = max(t.max_depth for t in trees)
    if figsize is None:
        figsize = (max(8.0, 1.75 * sum(widths)), 1.55 * (depth + 1) + 1.0)
    fig, axes = plt.subplots(1, len(trees), figsize=figsize, gridspec_kw=dict(width_ratios=widths), squeeze=False)
    for ax, tree, (_, row) in zip(axes[0], trees, rows.iterrows(), strict=True):
        title = f"tree #{tree.index}\nweight {row['weight']:.1%}  |  contribution {row['contribution']:+.3f}"
        plot_tree(tree, ax=ax, decimals=decimals, max_classes=max_classes, highlight_leaf=int(row["leaf"]), title=title)
    fig.suptitle(f"Top {len(trees)} trees for this sample  |  contributions in {unit}", fontsize=12)
    fig.tight_layout()
    return fig


def plot_feature_attributions(attributions, top_k: int = 10, ax=None, title: str | None = None, decimals: int = 3):
    """Waterfall chart of feature attributions: from the mean background output to the prediction."""
    plt = _plt()
    attrs = attributions.attrs
    top = attributions.iloc[:top_k]
    rest = attributions.iloc[top_k:]
    labels = [f"{f} = {_fmt_attr_value(v)}" for f, v in zip(top["feature"], top["value"], strict=True)]
    values = list(top["attribution"])
    if len(rest):
        labels.append(f"{len(rest)} other features")
        values.append(float(rest["attribution"].sum()))
    # bottom-up ("other features" first, then increasing magnitude): the largest attribution ends at the prediction
    labels, values = labels[::-1], values[::-1]
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 0.45 * len(values) + 1.6))
    base = attrs.get("base_value", 0.0)
    pos = base
    span = max(abs(sum(values)), max((abs(v) for v in values), default=1e-12))
    for y, v in enumerate(values):
        color = "#c44e52" if v >= 0 else "#4c72b0"
        ax.barh(y, v, left=pos, color=color, height=0.6)
        ax.text(pos + v + (0.01 * span if v >= 0 else -0.01 * span), y, f"{v:+.{decimals}g}", va="center",
                ha="left" if v >= 0 else "right", fontsize=8)
        pos += v
    ax.set_yticks(range(len(labels)), labels)
    ax.axvline(base, color="gray", ls="--", lw=1)
    ax.axvline(pos, color="black", ls="-", lw=1)
    ymax = len(labels) - 0.4
    ax.text(base, ymax, f" mean over background: {base:.{decimals + 1}g}", color="gray", fontsize=8, va="bottom")
    ax.text(pos, -0.9, f" this sample: {pos:.{decimals + 1}g}", fontsize=8, va="top")
    ax.set_ylim(-1.3, len(labels) + 0.2)
    ax.set_xlabel(f"model output ({attrs.get('unit', '')})")
    lo, hi = min(base, pos), max(base, pos)
    cum = np.cumsum([base] + values)
    lo, hi = min(lo, cum.min()), max(hi, cum.max())
    pad = 0.18 * (hi - lo if hi > lo else 1.0)
    ax.set_xlim(lo - pad, hi + pad)
    ax.spines[["top", "right"]].set_visible(False)
    if title is None:
        title = f"Feature attributions: {attrs.get('prediction_label', 'prediction')} = {attrs.get('prediction', np.nan):.4g}"
    ax.set_title(title)
    return ax


def _fmt_attr_value(v) -> str:
    if isinstance(v, (float, np.floating)):
        return "missing" if np.isnan(v) else format_number(v, 4)
    return str(v)
