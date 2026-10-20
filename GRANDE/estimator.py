"""scikit-learn style estimators: :class:`GRANDEClassifier` and :class:`GRANDERegressor`."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_is_fitted

from ._utils import compile_module, resolve_compile, resolve_device, torch_runtime
from .embeddings import embed_data
from .inspection import InspectionMixin
from .inspection._probe import capture_feature_domains
from .preprocessing import GRANDEPreprocessor
from .training import train_grande

_OPTIMIZERS = ("adam", "adamw", "nadam", "radam")


class _GRANDEBase(InspectionMixin, BaseEstimator):
    """Shared implementation of the GRANDE classifier and regressor.

    Parameters
    ----------
    depth : int, default=5
        Depth of each decision tree.
    n_estimators : int, default=1024
        Number of trees. All trees are trained jointly, so this is the exact ensemble size.
    selected_variables : float or int, default=0.8
        Features per tree: a fraction (<= 1, clipped to [10, 50] features) or an absolute number (> 1).
    data_subset_fraction : float, default=1.0
        Fraction of each batch every tree is trained on (< 1 enables per-tree subsampling).
    bootstrap : bool, default=False
        Draw the per-tree subsets with replacement.
    dropout : float, default=0.2
        Dropout on the instance-wise estimator weights during training.
    learning_rate_weights, learning_rate_index, learning_rate_values, learning_rate_leaf : float
        Learning rates of the estimator weights, split feature indices, split thresholds and leaf values.
    learning_rate_embedding : float, default=0.001
        Learning rate of the numeric/categorical embeddings.
    optimizer : {"adam", "adamw", "nadam", "radam"}, default="adam"
    cosine_decay_restarts : bool, default=False
        Use cosine annealing with warm restarts (T_0=10, T_mult=2).
    reduce_on_plateau_scheduler : bool, default=False
        Reduce learning rates on plateaus of the validation score (only active together with
        ``cosine_decay_restarts``, as in the TabArena benchmark).
    epochs : int, default=200
        Maximum number of training epochs.
    batch_size : int, default=256
        Training batch size (capped at half the training set size).
    early_stopping_epochs : int, default=10
        Patience (in epochs) of early stopping on the validation score.
    es_metric : bool, default=False
        If True, select/stop on ROC-AUC (binary), MSE (regression) or validation loss (multiclass);
        if False, on the validation loss.
    swa : bool, default=False
        Stochastic weight averaging after epoch 10.
    use_class_weights : bool, default=False
        Balanced class weights in the loss (classification only).
    focal_loss : bool, default=False
        Use the focal loss instead of cross-entropy (classification only).
    label_smoothing : float, default=0.0
        Label smoothing of the cross-entropy loss (classification only).
    use_numeric_embeddings : bool, default=True
        Learnable piecewise-linear embeddings of numeric features.
    embedding_dim_num : int, default=4
        Dimension of the numeric embeddings.
    num_emb_n_bins : int, default=8
        Number of (tree-based) bins of the piecewise-linear numeric embeddings.
    use_category_embeddings : bool, default=False
        Learnable embeddings of categorical features.
    embedding_dim_cat : int, default=8
        Dimension of the categorical embeddings.
    embedding_threshold : int, default=1
        With category embeddings, categoricals with fewer levels are one-hot encoded instead of embedded.
    loo_cardinality : int, default=10
        With numeric embeddings only, categoricals with at least this many levels are leave-one-out
        encoded, the others one-hot encoded.
    use_robust_scale_smoothing : bool, default=True
        Normalize with a robust scale + smooth clip transform instead of a quantile transform.
    use_freq_enc : bool, default=False
        Add frequency (rank) encodings of categorical features.
    missing_values : bool, default=False
        Route missing values inside the trees instead of imputing them.
    validation_fraction : float, default=0.2
        Fraction of the training data held out for early stopping if no ``X_val`` is passed to ``fit``.
    device : str or None, default=None
        ``"cpu"``, ``"cuda"`` / ``"cuda:N"``, or None to use the first GPU if available.
    compile_model : bool or "auto", default="auto"
        Compile the network with ``torch.compile`` (much faster training, after a one-off compile of
        typically 10-30 s). ``"auto"`` compiles on CUDA, and on CPU if a C++ compiler is available.
    matmul_precision : {"highest", "high", "medium"} or None, default="high"
        ``torch.set_float32_matmul_precision`` used during fit/predict (restored afterwards).
    n_threads : int or None, default=None
        Number of torch CPU threads during fit/predict (restored afterwards). None keeps the current setting.
    predict_batch_size : int, default=256
        Batch size used for prediction.
    random_state : int or None, default=42
        Seed of the run. None draws a random seed.
    verbose : int, default=0
        0: silent, 1: per-epoch logs, 2: also early-stopping logs.
    """

    def __init__(
        self,
        *,
        depth: int = 5,
        n_estimators: int = 1024,
        selected_variables: float = 0.8,
        data_subset_fraction: float = 1.0,
        bootstrap: bool = False,
        dropout: float = 0.2,
        learning_rate_weights: float = 0.005,
        learning_rate_index: float = 0.01,
        learning_rate_values: float = 0.01,
        learning_rate_leaf: float = 0.01,
        learning_rate_embedding: float = 0.001,
        optimizer: str = "adam",
        cosine_decay_restarts: bool = False,
        reduce_on_plateau_scheduler: bool = False,
        epochs: int = 200,
        batch_size: int = 256,
        early_stopping_epochs: int = 10,
        es_metric: bool = False,
        swa: bool = False,
        use_class_weights: bool = False,
        focal_loss: bool = False,
        label_smoothing: float = 0.0,
        use_numeric_embeddings: bool = True,
        embedding_dim_num: int = 4,
        num_emb_n_bins: int = 8,
        use_category_embeddings: bool = False,
        embedding_dim_cat: int = 8,
        embedding_threshold: int = 1,
        loo_cardinality: int = 10,
        use_robust_scale_smoothing: bool = True,
        use_freq_enc: bool = False,
        missing_values: bool = False,
        validation_fraction: float = 0.2,
        device: str | None = None,
        compile_model: bool | str = "auto",
        matmul_precision: str | None = "high",
        n_threads: int | None = None,
        predict_batch_size: int = 256,
        random_state: int | None = 42,
        verbose: int = 0,
    ):
        self.depth = depth
        self.n_estimators = n_estimators
        self.selected_variables = selected_variables
        self.data_subset_fraction = data_subset_fraction
        self.bootstrap = bootstrap
        self.dropout = dropout
        self.learning_rate_weights = learning_rate_weights
        self.learning_rate_index = learning_rate_index
        self.learning_rate_values = learning_rate_values
        self.learning_rate_leaf = learning_rate_leaf
        self.learning_rate_embedding = learning_rate_embedding
        self.optimizer = optimizer
        self.cosine_decay_restarts = cosine_decay_restarts
        self.reduce_on_plateau_scheduler = reduce_on_plateau_scheduler
        self.epochs = epochs
        self.batch_size = batch_size
        self.early_stopping_epochs = early_stopping_epochs
        self.es_metric = es_metric
        self.swa = swa
        self.use_class_weights = use_class_weights
        self.focal_loss = focal_loss
        self.label_smoothing = label_smoothing
        self.use_numeric_embeddings = use_numeric_embeddings
        self.embedding_dim_num = embedding_dim_num
        self.num_emb_n_bins = num_emb_n_bins
        self.use_category_embeddings = use_category_embeddings
        self.embedding_dim_cat = embedding_dim_cat
        self.embedding_threshold = embedding_threshold
        self.loo_cardinality = loo_cardinality
        self.use_robust_scale_smoothing = use_robust_scale_smoothing
        self.use_freq_enc = use_freq_enc
        self.missing_values = missing_values
        self.validation_fraction = validation_fraction
        self.device = device
        self.compile_model = compile_model
        self.matmul_precision = matmul_precision
        self.n_threads = n_threads
        self.predict_batch_size = predict_batch_size
        self.random_state = random_state
        self.verbose = verbose

    # ------------------------------------------------------------------------------------------
    # fitting
    # ------------------------------------------------------------------------------------------
    def fit(self, X, y, X_val=None, y_val=None, cat_features=None, time_limit: float | None = None):
        """Fit the ensemble.

        Parameters
        ----------
        X : pandas.DataFrame or array-like of shape (n_samples, n_features)
            Training features. Columns with category / object / string dtype are treated as categorical.
        y : array-like of shape (n_samples,)
            Targets.
        X_val, y_val : optional
            Validation data for early stopping and model selection. If omitted, a stratified
            ``validation_fraction`` of ``X`` is held out.
        cat_features : list of column names or indices, optional
            Additional columns to treat as categorical (e.g. integer-coded categories in a numpy array).
        time_limit : float, optional
            Time budget in seconds; training stops after the epoch in which it is exceeded.

        Returns
        -------
        self
        """
        start_time = time.monotonic()
        self._validate_hyperparameters()
        self.device_ = resolve_device(self.device)
        self.compiled_ = resolve_compile(self.compile_model, self.device_)
        self.random_state_ = (
            int(np.random.SeedSequence().generate_state(1)[0] % (2**31)) if self.random_state is None else int(self.random_state)
        )

        X = self._to_frame(X, fit=True)
        y_enc = self._encode_target(y, fit=True)
        self.objective_ = self._infer_objective()
        self._cat_features_in = self._resolve_cat_features(cat_features, X.columns)
        X = self._apply_cat_features(X)

        if X_val is None:
            X_train, X_val, y_train, y_val_enc = self._train_val_split(X, y_enc)
        else:
            X_train, y_train = X, y_enc
            X_val = self._apply_cat_features(self._to_frame(X_val))
            y_val_enc = self._encode_target(y_val)
        # training domain of every feature (used to inspect / prune the trees; does not affect training)
        self.feature_domains_ = capture_feature_domains(X_train)
        self._inspector_cache = None

        with torch_runtime(self.matmul_precision, self.n_threads):
            self.preprocessor_ = GRANDEPreprocessor(
                problem_type=self.objective_,
                use_category_embeddings=self.use_category_embeddings,
                use_numeric_embeddings=self.use_numeric_embeddings,
                embedding_threshold=self.embedding_threshold,
                loo_cardinality=self.loo_cardinality,
                missing_values=self.missing_values,
                use_freq_enc=self.use_freq_enc,
                use_robust_scale_smoothing=self.use_robust_scale_smoothing,
            )
            X_train_np = self.preprocessor_.fit_transform(X_train, y_train)
            X_val_np = self.preprocessor_.transform(X_val)
            self.y_mean_ = self.preprocessor_.mean
            self.y_std_ = self.preprocessor_.std

            hp = self.get_params()
            hp["random_seed"] = self.random_state_
            result = train_grande(
                hp=hp,
                objective=self.objective_,
                X_train=X_train_np,
                y_train=y_train,
                X_val=X_val_np,
                y_val=y_val_enc,
                num_columns_indices=self.preprocessor_.num_columns_indices,
                encoded_columns_indices=self.preprocessor_.encoded_columns_indices,
                y_mean=self.y_mean_,
                y_std=self.y_std_,
                device=self.device_,
                compile_model=self.compiled_,
                start_time=start_time,
                time_limit=time_limit,
            )

        self.module_ = result.model
        self.numeric_embeddings_ = result.numeric_embeddings
        self.category_embeddings_ = result.category_embeddings
        self.categorical_features_raw_indices_ = result.categorical_features_raw_indices
        self.best_val_score_ = result.best_val_score
        self.best_epoch_ = result.best_epoch
        self.n_epochs_ = result.n_epochs
        self.fit_time_ = time.monotonic() - start_time
        return self

    def _validate_hyperparameters(self):
        if str(self.optimizer).lower() not in _OPTIMIZERS:
            raise ValueError(f"optimizer must be one of {_OPTIMIZERS}, got {self.optimizer!r}.")
        if self.compile_model not in (True, False, "auto"):
            raise ValueError(f"compile_model must be True, False or 'auto', got {self.compile_model!r}.")
        if not 0.0 < self.validation_fraction < 1.0:
            raise ValueError(f"validation_fraction must be in (0, 1), got {self.validation_fraction!r}.")

    def _to_frame(self, X, fit: bool = False) -> pd.DataFrame:
        if hasattr(X, "tocsr") or hasattr(X, "toarray"):
            raise TypeError("Sparse input is not supported by GRANDE; pass a dense array or DataFrame.")
        if isinstance(X, pd.DataFrame):
            X = X.copy()
            X.columns = [str(c) for c in X.columns]
        else:
            X = np.asarray(X)
            if X.ndim != 2:
                raise ValueError(f"Expected 2D input, got array of shape {X.shape}.")
            X = pd.DataFrame(X, columns=[str(i) for i in range(X.shape[1])])
        if fit:
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
            self.n_features_in_ = X.shape[1]
        else:
            if X.shape[1] != self.n_features_in_:
                raise ValueError(f"X has {X.shape[1]} features, but this model was fitted with {self.n_features_in_}.")
            expected = list(self.feature_names_in_)
            if list(X.columns) != expected:
                if set(X.columns) != set(expected):
                    raise ValueError("The feature names of X do not match the ones seen during fit.")
                X = X[expected]
        return X

    def _resolve_cat_features(self, cat_features, columns):
        if cat_features is None:
            return []
        cols = list(columns)
        resolved = []
        for c in cat_features:
            if isinstance(c, (int, np.integer)) and str(c) not in cols:
                resolved.append(cols[int(c)])
            else:
                resolved.append(str(c))
        missing = [c for c in resolved if c not in cols]
        if missing:
            raise ValueError(f"cat_features not found in X: {missing}")
        return resolved

    def _apply_cat_features(self, X: pd.DataFrame) -> pd.DataFrame:
        for col in self._cat_features_in:
            if not isinstance(X[col].dtype, pd.CategoricalDtype):
                X[col] = X[col].astype("category")
        return X

    def _train_val_split(self, X: pd.DataFrame, y: np.ndarray):
        """Hold out ``validation_fraction`` of the data (stratified for classification, seed 0)."""
        stratify = y if self.objective_ in ("binary", "multiclass") else None
        X_split, y_split = X, y
        rare_idx = None
        if stratify is not None:
            values, counts = np.unique(y, return_counts=True)
            rare = values[counts < 2]
            if len(rare) > 0:
                # classes with a single sample stay in the training split
                rare_mask = np.isin(y, rare)
                rare_idx = np.flatnonzero(rare_mask)
                X_split, y_split = X.iloc[~rare_mask], y[~rare_mask]
                stratify = y_split
        try:
            X_tr, X_va, y_tr, y_va = train_test_split(
                X_split, y_split, test_size=self.validation_fraction, shuffle=True, random_state=0, stratify=stratify
            )
        except ValueError:
            if stratify is None:
                raise
            X_tr, X_va, y_tr, y_va = train_test_split(
                X_split, y_split, test_size=self.validation_fraction, shuffle=True, random_state=0, stratify=None
            )
        if rare_idx is not None and len(rare_idx) > 0:
            X_tr = pd.concat([X_tr, X.iloc[rare_idx]])
            y_tr = np.concatenate([y_tr, y[rare_idx]])
        return X_tr, X_va, y_tr, y_va

    # ------------------------------------------------------------------------------------------
    # prediction
    # ------------------------------------------------------------------------------------------
    def _inference_module(self):
        if self.compiled_ and not hasattr(self.module_, "_orig_mod"):
            self.module_ = compile_module(self.module_, self.device_)
        return self.module_

    def _network_inputs(self, X: pd.DataFrame):
        """Inputs of the network (preprocessed + embedded features) of a validated frame, in batches on the device."""
        X_np = self.preprocessor_.transform(X)
        for start in range(0, len(X_np), self.predict_batch_size):
            xb = torch.from_numpy(X_np[start:start + self.predict_batch_size]).to(self.device_)
            yield embed_data(
                xb,
                self.preprocessor_.num_columns_indices,
                self.categorical_features_raw_indices_,
                self.preprocessor_.encoded_columns_indices,
                self.numeric_embeddings_,
                self.category_embeddings_,
            )

    @torch.no_grad()
    def _predict_raw(self, X) -> np.ndarray:
        """Class probabilities (classification) or predictions in the original target scale (regression)."""
        check_is_fitted(self, "module_")
        X = self._apply_cat_features(self._to_frame(X))
        with torch_runtime(self.matmul_precision, self.n_threads):
            model = self._inference_module()
            model.eval()
            outputs = []
            for xb in self._network_inputs(X):
                logits = model(xb)
                if self.objective_ != "regression":
                    outputs.append(F.softmax(logits, dim=-1).cpu().numpy())
                else:
                    outputs.append(logits.cpu().numpy() * self.y_std_ + self.y_mean_)
        return np.concatenate(outputs)

    @torch.no_grad()
    def _model_inputs(self, X: pd.DataFrame) -> np.ndarray:
        """Inputs of the network (preprocessed + embedded features) for an already validated frame."""
        with torch_runtime(self.matmul_precision, self.n_threads):
            return np.concatenate([xb.float().cpu().numpy() for xb in self._network_inputs(X)])

    # ------------------------------------------------------------------------------------------
    # devices, persistence
    # ------------------------------------------------------------------------------------------
    def to(self, device: str):
        """Move a fitted model to ``device`` (e.g. ``"cpu"`` before pickling for a CPU-only machine)."""
        check_is_fitted(self, "module_")
        device = resolve_device(device)
        device_type_changed = device.split(":")[0] != self.device_.split(":")[0]
        self.compiled_ = resolve_compile(self.compile_model, device)
        if (device_type_changed or not self.compiled_) and hasattr(self.module_, "_orig_mod"):
            # compiled for another device type: recompile lazily on first use with the settings for `device`
            self.module_ = self.module_._orig_mod
        self.module_.to(device)
        for emb in (self.numeric_embeddings_, self.category_embeddings_):
            if emb is not None:
                emb.to(device)
        self.device_ = device
        self._inspector_cache = None
        return self

    def save(self, path: str) -> None:
        """Save the fitted estimator with ``torch.save``."""
        torch.save(self, path)

    @classmethod
    def load(cls, path: str, device: str | None = None):
        """Load an estimator saved with :meth:`save`; ``device`` defaults to the training device if available."""
        est = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(est, cls):
            raise TypeError(f"{path} does not contain a {cls.__name__}.")
        target = device
        if target is None:
            target = est.device_ if (not est.device_.startswith("cuda") or torch.cuda.is_available()) else "cpu"
        return est.to(target)

    def __getstate__(self):
        state = dict(super().__getstate__())
        state.pop("_inspector_cache", None)  # rebuilt on demand
        module = state.get("module_")
        if module is not None and hasattr(module, "_orig_mod"):
            # store the plain module; it is compiled again on first use
            state["module_"] = module._orig_mod
        return state

    def _more_tags(self):
        return {"allow_nan": True, "non_deterministic": True}

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        tags.non_deterministic = True
        return tags


def _with_parameters_doc(summary: str) -> str:
    base = _GRANDEBase.__doc__ or ""
    params = base[base.find("Parameters"):] if "Parameters" in base else ""
    return f"{summary}\n\n    {params}"


class GRANDEClassifier(ClassifierMixin, _GRANDEBase):
    __doc__ = _with_parameters_doc(
        "GRANDE classifier: gradient-based ensemble of hard, axis-aligned decision trees.\n\n"
        "    Binary and multiclass problems are detected from the number of classes in ``y``."
    )

    def _encode_target(self, y, fit: bool = False) -> np.ndarray:
        y = np.asarray(y.to_numpy() if isinstance(y, (pd.Series, pd.DataFrame)) else y).reshape(-1)
        if fit:
            self._label_encoder = LabelEncoder().fit(y)
            self.classes_ = self._label_encoder.classes_
            self.n_classes_ = len(self.classes_)
        return self._label_encoder.transform(y).astype(np.float32)

    def _infer_objective(self) -> str:
        if self.n_classes_ < 2:
            raise ValueError("GRANDEClassifier needs at least two classes in y.")
        return "binary" if self.n_classes_ == 2 else "multiclass"

    def predict_proba(self, X) -> np.ndarray:
        """Class probabilities of shape (n_samples, n_classes), columns ordered as ``classes_``."""
        return self._predict_raw(X)

    def predict(self, X) -> np.ndarray:
        return self.classes_[np.argmax(self._predict_raw(X), axis=1)]


class GRANDERegressor(RegressorMixin, _GRANDEBase):
    __doc__ = _with_parameters_doc(
        "GRANDE regressor: gradient-based ensemble of hard, axis-aligned decision trees.\n\n"
        "    The target is standardized internally; classification-only parameters are ignored."
    )

    def _encode_target(self, y, fit: bool = False) -> np.ndarray:
        y = np.asarray(y.to_numpy() if isinstance(y, (pd.Series, pd.DataFrame)) else y).reshape(-1)
        return y.astype(np.float32)

    def _infer_objective(self) -> str:
        return "regression"

    def predict(self, X) -> np.ndarray:
        return self._predict_raw(X)
