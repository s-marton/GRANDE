"""Training of the GRANDE network: embeddings, optimizers, schedulers, the epoch loop and early stopping."""

from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rtdl_num_embeddings import PiecewiseLinearEmbeddings, compute_bins
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, TensorDataset

from ._utils import compile_module, set_seed, vprint
from .embeddings import embed_data
from .losses import FocalLoss
from .model import GRANDE_Module


@dataclass
class TrainingResult:
    model: nn.Module
    numeric_embeddings: nn.Module | None
    category_embeddings: nn.Module | None
    categorical_features_raw_indices: list
    best_val_score: float
    best_epoch: int | None
    n_epochs: int


class _EarlyStopping:
    """Stop when the validation score (lower is better) has not improved by more than ``min_delta`` for
    ``patience`` epochs."""

    def __init__(self, patience: int, min_delta: float, verbose: bool):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.best = None
        self.count = 0

    def step(self, value: float) -> bool:
        if self.best is None or value < self.best - self.min_delta:
            self.best = value
            self.count = 0
            return False
        self.count += 1
        if self.verbose:
            print(f"[EarlyStop] no improvement ({self.count}/{self.patience}). best={self.best:.6f}, "
                  f"current={value:.6f}", flush=True)
        if self.count >= self.patience:
            if self.verbose:
                print("[EarlyStop] stopping.", flush=True)
            return True
        return False


def _loader(X: np.ndarray, y: np.ndarray, objective: str, batch_size: int, shuffle: bool, drop_last: bool) -> DataLoader:
    y = torch.from_numpy(y)
    y = y.long() if objective in ("binary", "multiclass") else y.float()
    return DataLoader(TensorDataset(torch.from_numpy(X), y), batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)


def compute_loss(objective, logits, yb, *, class_weights, focal_loss, label_smoothing, number_of_classes, mean, std, reduction):
    """Training/validation loss: (focal) cross-entropy for classification, MSE on the standardized target."""
    if objective in ("binary", "multiclass"):
        if focal_loss:
            if objective == "binary":
                alpha = (class_weights[1] / (class_weights[0] + class_weights[1]) if class_weights is not None else 0.5)
                criterion = FocalLoss(gamma=2, alpha=alpha, task_type="binary", reduction=reduction)
                return criterion(logits[:, 1], yb)
            alpha = class_weights / class_weights.sum()
            criterion = FocalLoss(gamma=2, alpha=alpha, task_type="multi-class", reduction=reduction, num_classes=number_of_classes)
            return criterion(logits, yb)
        return F.cross_entropy(logits, yb, reduction=reduction, weight=class_weights, label_smoothing=label_smoothing)
    true = (yb.float() - mean) / std
    return F.mse_loss(logits, true, reduction=reduction)


def _validation_metrics(objective, outputs: np.ndarray, y_true: np.ndarray, mean, std) -> dict:
    """Task metrics on the validation data from class probabilities (classification) or raw outputs (regression)."""
    if objective == "regression":
        y_pred = outputs * std + mean
        return {"MSE": mean_squared_error(y_true, y_pred), "MAE": mean_absolute_error(y_true, y_pred),
                "R2": r2_score(y_true, y_pred)}
    y_pred = np.argmax(outputs, axis=1)
    try:
        auc = roc_auc_score(y_true, outputs[:, 1]) if outputs.shape[1] == 2 else roc_auc_score(y_true, outputs, multi_class="ovo")
    except ValueError:
        auc = float("nan")
    return {"ValAcc": accuracy_score(y_true, y_pred), "ValAUC": auc, "ValF1": f1_score(y_true, y_pred, average="macro")}


def _class_weights(objective, y_train, use_class_weights):
    """Number of output classes for the module (1 for binary/regression) and the loss class weights."""
    if objective == "regression":
        use_class_weights = False
    if use_class_weights:
        number_of_classes = 1 if objective == "binary" else len(np.unique(y_train))
        counts = np.bincount(y_train.astype(int), minlength=number_of_classes).astype(np.float64)
        counts[counts == 0] = 1.0
        inv = 1.0 / counts
        class_weights = inv * (number_of_classes / inv.sum())
    else:
        number_of_classes = 1 if objective in ("binary", "regression") else len(np.unique(y_train))
        class_weights = np.ones_like(np.unique(y_train))
    return number_of_classes, class_weights


def _build_embeddings(hp, objective, X_train, X_val, y_train, y_val, num_columns_indices, categorical_features_raw_indices):
    numeric_embeddings = None
    category_embeddings = None
    if hp["use_numeric_embeddings"] and len(num_columns_indices) > 0:
        # The bins are computed on training plus validation data (as in the TabArena benchmark).
        X_bins = np.concatenate([X_train, X_val], axis=0)[:, num_columns_indices]
        if np.isnan(X_bins).any():
            raise ValueError("Numeric embeddings need complete numeric features: use missing_values=False "
                             "(imputation) or use_numeric_embeddings=False for data with missing values.")
        bins = compute_bins(
            torch.from_numpy(X_bins),
            n_bins=hp["num_emb_n_bins"],
            tree_kwargs={"min_samples_leaf": 64, "min_impurity_decrease": 1e-4},
            y=torch.from_numpy(np.concatenate([y_train, y_val], axis=0)),
            regression=objective == "regression",
        )
        set_seed(hp["random_seed"])
        numeric_embeddings = PiecewiseLinearEmbeddings(bins=bins, d_embedding=hp["embedding_dim_num"], activation=False, version="B")
        set_seed(hp["random_seed"])

    if hp["use_category_embeddings"] and len(categorical_features_raw_indices) > 0:
        categories = [len(np.unique(X_train[:, idx])) + 1 for idx in categorical_features_raw_indices]
        set_seed(hp["random_seed"])
        category_embeddings = nn.Embedding(sum(categories), hp["embedding_dim_cat"])
        set_seed(hp["random_seed"])
        nn.init.kaiming_uniform_(category_embeddings.weight, a=math.sqrt(5))
        set_seed(hp["random_seed"])
    return numeric_embeddings, category_embeddings


def _build_optimizers(hp, model, numeric_embeddings, category_embeddings):
    """One optimizer for the tree parameters (two for 'adamw') and one for the embeddings."""
    optimizer = hp["optimizer"].lower()
    betas = (0.9, 0.95)
    optimizers = []
    if optimizer == "adamw":
        optimizers.append(torch.optim.Adam([
            {"params": model.split_values, "lr": hp["learning_rate_values"], "betas": betas},
            {"params": model.estimator_weights, "lr": hp["learning_rate_weights"], "betas": betas},
        ]))
        optimizers.append(torch.optim.AdamW([
            {"params": model.split_index_array, "lr": hp["learning_rate_index"], "betas": betas},
            {"params": model.leaf_classes_array, "lr": hp["learning_rate_leaf"], "betas": betas},
        ]))
    else:
        param_group = [
            {"params": model.split_values, "lr": hp["learning_rate_values"], "betas": betas},
            {"params": model.split_index_array, "lr": hp["learning_rate_index"], "betas": betas},
            {"params": model.estimator_weights, "lr": hp["learning_rate_weights"], "betas": betas},
            {"params": model.leaf_classes_array, "lr": hp["learning_rate_leaf"], "betas": betas},
        ]
        optimizer_cls = {"nadam": torch.optim.NAdam, "radam": torch.optim.RAdam}.get(optimizer, torch.optim.Adam)
        optimizers.append(optimizer_cls(param_group))

    param_group_emb = [
        {"params": emb.parameters(), "lr": hp["learning_rate_embedding"], "betas": betas}
        for emb in (numeric_embeddings, category_embeddings) if emb is not None
    ]
    if param_group_emb:
        optimizer_cls = {"adamw": torch.optim.AdamW, "nadam": torch.optim.NAdam, "radam": torch.optim.RAdam}.get(optimizer, torch.optim.Adam)
        optimizers.append(optimizer_cls(param_group_emb))
    return optimizers


def train_grande(
    *,
    hp: dict,
    objective: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    num_columns_indices: list,
    encoded_columns_indices: list,
    y_mean,
    y_std,
    device: str,
    compile_model: bool,
    start_time: float,
    time_limit: float | None,
) -> TrainingResult:
    """Build and train the GRANDE network (plus optional embeddings) on preprocessed float32 arrays.

    ``hp`` holds the estimator hyperparameters with the seed under ``random_seed``. Returns the
    trained modules with the weights of the best validation epoch restored.
    """
    verbose = hp["verbose"]

    number_of_variables = X_train.shape[1]
    number_of_classes, class_weights = _class_weights(objective, y_train, hp["use_class_weights"])
    class_weights = torch.tensor(class_weights, dtype=torch.float32, device=device)

    if hp["use_category_embeddings"] or hp["use_numeric_embeddings"]:
        categorical_features_raw_indices = [i for i in range(number_of_variables)
                                            if i not in num_columns_indices and i not in encoded_columns_indices]
        numeric_embeddings, category_embeddings = _build_embeddings(
            hp, objective, X_train, X_val, y_train, y_val, num_columns_indices, categorical_features_raw_indices
        )
        n_cat, n_num = len(categorical_features_raw_indices), len(num_columns_indices)
        number_of_variables = (
            (hp["embedding_dim_cat"] * n_cat if hp["use_category_embeddings"] else n_cat)
            + (hp["embedding_dim_num"] * n_num if hp["use_numeric_embeddings"] else n_num)
            + len(encoded_columns_indices)
        )
    else:
        numeric_embeddings = None
        category_embeddings = None
        categorical_features_raw_indices = []

    # effective batch size (the per-estimator data subsets of the module index into the batch)
    batch_size = max(1, min(hp["batch_size"], X_train.shape[0] // 2))

    model = GRANDE_Module(
        **{key: hp[key] for key in ("depth", "n_estimators", "dropout", "selected_variables", "data_subset_fraction",
                                    "bootstrap", "missing_values", "random_seed", "verbose")},
        batch_size=batch_size, device=device, objective=objective,
        number_of_variables=number_of_variables, number_of_classes=number_of_classes,
    ).to(device)
    if compile_model:
        model = compile_module(model, device)

    if numeric_embeddings is not None:
        numeric_embeddings = numeric_embeddings.to(device)
        vprint(verbose, 1, "Use Numeric Embeddings:", numeric_embeddings)
    if category_embeddings is not None:
        category_embeddings = category_embeddings.to(device)
        vprint(verbose, 1, "Use Category Embeddings:", category_embeddings)

    swa = hp["swa"]
    if swa:
        swa_model = torch.optim.swa_utils.AveragedModel(model)
        swa_numeric_embeddings = torch.optim.swa_utils.AveragedModel(numeric_embeddings) if numeric_embeddings is not None else None
        swa_category_embeddings = torch.optim.swa_utils.AveragedModel(category_embeddings) if category_embeddings is not None else None

    train_loader = _loader(X_train, y_train, objective, batch_size,
                           shuffle=hp["data_subset_fraction"] == 1.0 and not hp["bootstrap"], drop_last=True)
    val_loader = _loader(X_val, y_val, objective, batch_size, shuffle=False, drop_last=False)
    early_stopping = _EarlyStopping(patience=hp["early_stopping_epochs"], min_delta=0.001, verbose=verbose >= 2)

    optimizers = _build_optimizers(hp, model, numeric_embeddings, category_embeddings)
    plateau_schedulers = []
    if hp["reduce_on_plateau_scheduler"]:
        plateau_schedulers = [
            torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.2, patience=15, threshold=0.001)
            for opt in optimizers
        ]
    cosine_schedulers = []
    if hp["cosine_decay_restarts"]:
        vprint(verbose, 1, "Using CosineAnnealingLR")
        cosine_schedulers = [
            torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=10, T_mult=2, eta_min=1e-5)
            for opt in optimizers
        ]

    loss_kwargs = dict(class_weights=class_weights, focal_loss=hp["focal_loss"], label_smoothing=float(hp["label_smoothing"]),
                       number_of_classes=number_of_classes, mean=y_mean, std=y_std)
    embed_kwargs = dict(num_features=num_columns_indices, categorical_features_raw_indices=categorical_features_raw_indices,
                        encoded_columns_indices=encoded_columns_indices)

    best_val_score = float("inf")
    best_epoch = None
    best_states = None
    epoch = 0

    set_seed(hp["random_seed"])
    for epoch in range(1, hp["epochs"] + 1):
        t0 = time.time()
        total, n = 0.0, 0
        model.train()
        if numeric_embeddings is not None:
            numeric_embeddings.train()
        if category_embeddings is not None:
            category_embeddings.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            xb = embed_data(xb, numeric_embeddings=numeric_embeddings, category_embeddings=category_embeddings, **embed_kwargs)
            loss = compute_loss(objective, model(xb), yb, reduction="mean", **loss_kwargs)

            for opt in optimizers:
                opt.zero_grad(set_to_none=True)
            loss.backward()
            for opt in optimizers:
                opt.step()

            total += loss.item() * xb.size(0)
            n += xb.size(0)

        # ------------------------------------------------------------------------------------------
        # validation
        # ------------------------------------------------------------------------------------------
        use_swa_now = swa and epoch > 10
        if use_swa_now:
            swa_model.update_parameters(model)
            if numeric_embeddings is not None:
                swa_numeric_embeddings.update_parameters(numeric_embeddings)
            if category_embeddings is not None:
                swa_category_embeddings.update_parameters(category_embeddings)
            model_eval, numeric_embeddings_eval, category_embeddings_eval = swa_model, swa_numeric_embeddings, swa_category_embeddings
            # The averaged model is switched to eval mode only if the task metric is computed, so that without
            # es_metric its validation loss is computed in train mode (as in the TabArena benchmark).
            if hp["es_metric"]:
                model_eval.eval()
        else:
            model_eval, numeric_embeddings_eval, category_embeddings_eval = model, numeric_embeddings, category_embeddings
            model_eval.eval()
            if numeric_embeddings is not None:
                numeric_embeddings_eval.eval()
            if category_embeddings is not None:
                category_embeddings_eval.eval()

        if hp["es_metric"]:
            # The task metric used to be computed in a separate pass over the validation data. Every pass over a
            # DataLoader draws one number from torch's global random number generator; that draw is kept so that
            # training is identical to the TabArena benchmark.
            torch.empty((), dtype=torch.int64).random_()
        vtotal, vn, outputs, targets = 0.0, 0, [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                if hp["es_metric"]:
                    targets.append(yb.numpy())
                xb = xb.to(device)
                yb = yb.to(device)
                xb = embed_data(xb, numeric_embeddings=numeric_embeddings_eval, category_embeddings=category_embeddings_eval,
                                **embed_kwargs)
                logits = model_eval(xb)
                vtotal += float(compute_loss(objective, logits, yb, reduction="sum", **loss_kwargs).item())
                vn += xb.size(0)
                if hp["es_metric"]:
                    outputs.append((logits if objective == "regression" else F.softmax(logits, dim=-1)).cpu().numpy())
        val_loss = vtotal / max(vn, 1)

        # es_metric: ROC-AUC for binary, MSE for regression, validation loss for multiclass
        metrics = {}
        val_score = val_loss
        if hp["es_metric"]:
            metrics = _validation_metrics(objective, np.concatenate(outputs), np.concatenate(targets), y_mean, y_std)
            if objective == "binary":
                val_score = -metrics["ValAUC"]
            elif objective == "regression":
                val_score = metrics["MSE"]

        if val_score < best_val_score:
            best_val_score = val_score
            best_epoch = epoch
            best_states = [copy.deepcopy(m.module.state_dict() if use_swa_now else m.state_dict()) if m is not None else None
                           for m in (model_eval, numeric_embeddings_eval, category_embeddings_eval)]

        log = " | ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
        vprint(verbose, 1, f"Epoch {epoch:03d} | TrainLoss: {total/n:.4f} | ValLoss: {val_loss:.4f} | "
                           f"{log + ' | ' if log else ''}Time: {time.time() - t0:.2f}s")

        if early_stopping.step(val_score):
            break

        if cosine_schedulers:
            for scheduler in cosine_schedulers:
                scheduler.step()
            # ReduceLROnPlateau is only stepped together with cosine decay (as in the TabArena benchmark).
            for plateau in plateau_schedulers:
                plateau.step(val_score)

        if time_limit is not None and time.monotonic() - start_time >= time_limit:
            vprint(verbose, 1, f"[EarlyStop] time limit of {time_limit:.1f}s reached after epoch {epoch}.")
            break

    # restore the weights of the best epoch
    if best_states is not None:
        vprint(verbose, 1, "Restoring best model from epoch with val score:", best_val_score)
        for module, state in zip((model, numeric_embeddings, category_embeddings), best_states, strict=True):
            if module is not None:
                module.load_state_dict(state)

    return TrainingResult(
        model=model,
        numeric_embeddings=numeric_embeddings,
        category_embeddings=category_embeddings,
        categorical_features_raw_indices=categorical_features_raw_indices,
        best_val_score=best_val_score,
        best_epoch=best_epoch,
        n_epochs=epoch,
    )
