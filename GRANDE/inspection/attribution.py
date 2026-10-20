"""Shapley-value attributions of a single prediction to the input features.

Interventional Shapley values with a background sample: the value of a coalition S of features is the
model output for the hybrid input that takes the features in S from the explained sample x and all other
features from a background row z, averaged over the background rows. The attributions add up to the
difference between the output for x and the mean output over the background (efficiency).

GRANDE's output is a softmax-weighted average over all trees, i.e. not a sum over trees, so per-tree
TreeSHAP values cannot be added up. The computation here exploits two properties instead:

* every model input depends on exactly one original feature, so hybrid inputs are assembled directly in
  model-input space (no preprocessing) and evaluated in large batches;
* a feature can only change the output for a pair (x, z) if at least one split of the ensemble routes x
  and z differently on it. All other features are dummy players (Shapley value 0) and are removed.

For every background row the Shapley values are computed exactly by enumerating all coalitions of the
remaining features if there are at most ``max_exact_features`` of them, and otherwise estimated with
antithetic permutation sampling; both satisfy efficiency exactly.

The values are computed with full float32 matrix-multiplication precision (no TF32) and a float64 final
sum, so that coalition values evaluated in different batches are consistent; the reported prediction can
therefore differ from ``predict_proba`` on GPUs in the third decimal.
"""

from __future__ import annotations

from math import factorial

import numpy as np
import pandas as pd
import torch

from .._utils import torch_runtime


def _score_function(estimator, inspector):
    """Model output in an additive unit for model inputs (torch, on the model device)."""
    module = inspector.module
    leaf = module.leaf_classes_array.detach()
    objective = estimator.objective_
    y_mean, y_std = float(getattr(estimator, "y_mean_", 0.0)), float(getattr(estimator, "y_std_", 1.0))

    def score(Z: torch.Tensor, class_index: int | None = None, all_classes: bool = False) -> torch.Tensor:
        p, w = module.routing(Z)
        p, w, lv = p.double(), w.double(), leaf.double()
        if objective == "binary":
            return 2.0 * torch.einsum("be,bel,el->b", w, p, lv)
        if objective == "multiclass":
            z = torch.einsum("be,bel,elc->bc", w, p, lv)
            return z if all_classes else z[:, class_index] - z.mean(dim=1)
        return torch.einsum("be,bel,el->b", w, p, lv) * y_std + y_mean

    return score


def _subset_weights(n: int):
    sizes = np.array([bin(i).count("1") for i in range(2 ** n)])
    w = np.array([factorial(s) * factorial(n - s - 1) / factorial(n) for s in range(n)])
    return sizes, w


def shapley_values(estimator, inspector, x_frame: pd.DataFrame, background: pd.DataFrame, max_exact_features: int = 12,
                   n_permutations: int = 64, random_state: int = 0, batch_size: int = 8192,
                   eval_batch_size: int = 512) -> pd.DataFrame:
    probe, params = inspector.probe, inspector.params
    n_features = len(probe.domains)
    Zx = estimator._model_inputs(x_frame)[0]
    Zb = estimator._model_inputs(background)
    device = estimator.device_
    score_rows = _score_function(estimator, inspector)
    objective = estimator.objective_

    def score(Z: torch.Tensor, class_index: int | None = None) -> torch.Tensor:
        # routing materializes (rows x trees x leaves x depth) tensors: evaluate in small chunks
        return torch.cat([score_rows(Z[s:s + eval_batch_size], class_index) for s in range(0, len(Z), eval_batch_size)])

    with torch.no_grad(), torch_runtime("highest", estimator.n_threads):
        inspector.module.eval()
        class_index = None
        if objective == "multiclass":
            logits_x = score_rows(torch.from_numpy(Zx[None]).to(device), all_classes=True)[0].cpu().numpy()
            class_index = int(np.argmax(logits_x))
        f_x = float(score(torch.from_numpy(Zx[None]).to(device), class_index)[0])
        f_b = np.concatenate([score(torch.from_numpy(Zb[s:s + batch_size]).to(device), class_index).cpu().numpy()
                              for s in range(0, len(Zb), batch_size)])

        # features whose splits route x and the reference differently (all others are dummy players)
        columns, thresholds = params.columns, params.thresholds
        node_feature = probe.col_feature[columns]
        decision_x = Zx[columns] < thresholds
        cols_of = [np.flatnonzero(probe.col_feature == j) for j in range(n_features)]
        rng = np.random.default_rng(random_state)

        jobs = []  # (reference index, relevant features, coalition masks (n, R) bool, mode)
        n_exact = n_sampled = 0
        for b in range(len(Zb)):
            differs = (Zb[b][columns] < thresholds) != decision_x
            relevant = np.unique(node_feature[differs & (node_feature >= 0)])
            R = len(relevant)
            if R == 0:
                continue
            if R <= max_exact_features:
                masks = ((np.arange(2 ** R)[:, None] >> np.arange(R)[None, :]) & 1).astype(bool)
                jobs.append((b, relevant, masks, "exact", None))
                n_exact += 1
            else:
                half = max(1, n_permutations // 2)
                perms = [rng.permutation(R) for _ in range(half)]
                perms += [p[::-1] for p in perms]
                masks = np.zeros((len(perms) * (R + 1), R), dtype=bool)
                for k, perm in enumerate(perms):
                    for pos in range(1, R + 1):
                        masks[k * (R + 1) + pos, perm[:pos]] = True
                jobs.append((b, relevant, masks, "sampled", perms))
                n_sampled += 1

        # evaluate all hybrid inputs in batches
        Zx_t = torch.from_numpy(Zx).to(device)
        Zb_t = torch.from_numpy(Zb).to(device)
        values = [None] * len(jobs)
        pending, rows = [], 0

        def flush():
            hybrids, sizes = [], []
            for k in pending:
                b, relevant, masks, _, _ = jobs[k]
                colmask = torch.zeros((len(masks), Zx.shape[0]), dtype=torch.bool, device=device)
                masks_t = torch.from_numpy(masks).to(device)
                for r, j in enumerate(relevant):
                    colmask[:, torch.from_numpy(cols_of[j]).to(device)] = masks_t[:, r:r + 1]
                hybrids.append(torch.where(colmask, Zx_t[None, :], Zb_t[b][None, :]))
                sizes.append(len(masks))
            out = torch.cat([score(h, class_index) for h in hybrids]).cpu().numpy()
            start = 0
            for k, size in zip(pending, sizes, strict=True):
                values[k] = out[start:start + size]
                start += size
            pending.clear()

        for k, job in enumerate(jobs):
            pending.append(k)
            rows += len(job[2])
            if rows >= batch_size:
                flush()
                rows = 0
        if pending:
            flush()

    phi = np.zeros(n_features)
    weight_cache = {}
    for (_, relevant, _, mode, perms), v in zip(jobs, values, strict=True):
        R = len(relevant)
        if mode == "exact":
            if R not in weight_cache:
                weight_cache[R] = _subset_weights(R)
            sizes, w = weight_cache[R]
            idx = np.arange(2 ** R)
            for r in range(R):
                without = idx[((idx >> r) & 1) == 0]
                phi[relevant[r]] += np.sum(w[sizes[without]] * (v[without | (1 << r)] - v[without]))
        else:
            contrib = np.zeros(R)
            for k, perm in enumerate(perms):
                vals = v[k * (R + 1):(k + 1) * (R + 1)]
                contrib[perm] += np.diff(vals)
            phi[relevant] += contrib / len(perms)
    phi /= len(Zb)

    names = list(x_frame.columns)
    row = x_frame.iloc[0]
    df = pd.DataFrame({"feature": names, "value": [row[c] for c in names], "attribution": phi})
    df = df.reindex(df["attribution"].abs().sort_values(ascending=False).index).reset_index(drop=True)
    base = float(np.mean(f_b))
    attrs = dict(base_value=base, score=f_x, n_references=len(Zb), n_exact=n_exact, n_sampled=n_sampled,
                 n_dummy_references=len(Zb) - n_exact - n_sampled)
    if objective == "binary":
        target = estimator.classes_[1]
        attrs.update(unit=f"log-odds of {target}", prediction=float(1.0 / (1.0 + np.exp(-f_x))), prediction_label=f"P({target})")
    elif objective == "multiclass":
        target = estimator.classes_[class_index]
        e = np.exp(logits_x - logits_x.max())
        attrs.update(unit=f"logit of {target} vs. mean logit", prediction_label=f"P({target})",
                     prediction=float(e[class_index] / e.sum()))
    else:
        attrs.update(unit="target units", prediction=f_x, prediction_label="prediction")
    attrs["target"] = target.item() if objective != "regression" and isinstance(target, np.generic) else (target if objective != "regression" else None)
    df.attrs.update(attrs)
    return df
