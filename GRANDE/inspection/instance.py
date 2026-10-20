"""Per-sample analysis of GRANDE's instance-wise estimator weights.

For a sample x, every tree e reaches one leaf l_e(x) with a raw leaf value v_e and a weight logit
W_e(l_e). The estimator weights are w_e(x) = softmax_e(W_e(l_e(x))) and the ensemble output is
sum_e w_e(x) * v_e. The contribution of a tree is its share of that output, in an additive unit:

* binary:     log-odds of the positive class (``classes_[1]``): 2 * w_e * v_e; P = sigmoid(sum)
* multiclass: logit of the predicted class relative to the mean logit: w_e * (v_e[k] - mean(v_e))
* regression: target units: y_std * w_e * v_e; prediction = y_mean + sum
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _py(value):
    """numpy scalars -> plain Python values (for readable DataFrame.attrs)."""
    return value.item() if isinstance(value, np.generic) else value


def explain_instance(estimator, inspector, x_frame: pd.DataFrame) -> pd.DataFrame:
    Z = estimator._model_inputs(x_frame)
    leaves, weights = inspector.routing(Z)
    leaves, w = leaves[0], weights[0].astype(float)
    E = len(w)
    params = inspector.params
    v = params.leaf_values[np.arange(E), leaves].astype(float)
    objective = estimator.objective_
    attrs = {"effective_n_estimators": float(1.0 / np.sum(w ** 2)), "n_estimators": E}
    if objective == "binary":
        positive = estimator.classes_[1]
        contribution = 2.0 * w * v
        leaf_prediction = 1.0 / (1.0 + np.exp(-2.0 * v))
        score = float(contribution.sum())
        attrs.update(target=_py(positive), score=score, prediction=float(1.0 / (1.0 + np.exp(-score))),
                     prediction_label=f"P({positive})", contribution_unit=f"log-odds of {positive}")
    elif objective == "multiclass":
        z = (w[:, None] * v).sum(axis=0)
        k = int(np.argmax(z))
        contribution = w * (v[:, k] - v.mean(axis=1))
        ev = np.exp(v - v.max(axis=1, keepdims=True))
        leaf_prediction = ev[:, k] / ev.sum(axis=1)
        ez = np.exp(z - z.max())
        target = estimator.classes_[k]
        attrs.update(target=_py(target), score=float(z[k] - z.mean()), prediction=float(ez[k] / ez.sum()),
                     prediction_label=f"P({target})", contribution_unit=f"logit of {target} vs. mean logit")
    else:
        contribution = float(estimator.y_std_) * w * v
        leaf_prediction = v * float(estimator.y_std_) + float(estimator.y_mean_)
        score = float(contribution.sum())
        attrs.update(target=None, score=score, prediction=float(estimator.y_mean_) + score, base=float(estimator.y_mean_),
                     prediction_label="prediction", contribution_unit="target units")
    df = pd.DataFrame({
        "estimator": np.arange(E),
        "leaf": leaves,
        "weight": w,
        "weight_logit": params.leaf_weights[np.arange(E), leaves],
        "leaf_prediction": leaf_prediction,
        "contribution": contribution,
    }).sort_values("weight", ascending=False, kind="stable").reset_index(drop=True)
    df.attrs.update(attrs)
    return df
