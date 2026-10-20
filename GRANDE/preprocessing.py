"""Feature preprocessing for GRANDE.

:class:`GRANDEPreprocessor` turns a pandas DataFrame into the float32 matrix GRANDE trains on.
Depending on the embedding configuration it

* label-encodes categorical columns (category / object / string dtypes),
* imputes missing values (unless ``missing_values=True``),
* encodes categoricals: ordinal (binary), one-hot (low cardinality), leave-one-out target
  encoding (high cardinality) or keeps raw codes for learnable category embeddings,
* optionally adds frequency-encoded columns, and
* normalizes features with a noisy quantile transform or a robust scale + smooth clip transform.
"""

from __future__ import annotations

import category_encoders as ce
import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer

#: Without embeddings, categoricals with fewer levels are one-hot encoded, the others leave-one-out encoded.
_MAX_ONE_HOT_LEVELS = 10


class CategoricalCodeEncoder:
    """Replace categorical columns by their integer category codes (NaN and unseen values -> -1).

    The categories are stored at fit time, so a column is encoded consistently across calls.
    """

    def fit(self, X: pd.DataFrame) -> CategoricalCodeEncoder:
        self.features_in = [col for col in X.columns if isinstance(X[col].dtype, pd.CategoricalDtype)]
        self.categories_ = {col: X[col].cat.categories for col in self.features_in}
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {col: pd.Categorical(X[col], categories=self.categories_[col]).codes for col in self.features_in},
            index=X.index,
        )


class FrequencyEncoder:
    """Add a ``<col>_freq`` column per categorical column: the rank of the category's training frequency,
    normalized to (0, 1] (NaN for categories not seen during fit)."""

    def __init__(self, cols):
        self.cols = cols

    def fit(self, X: pd.DataFrame) -> FrequencyEncoder:
        self.maps_ = {}
        for c in self.cols:
            freq = X[c].value_counts(dropna=False)
            freq = freq.rank(method="average") / freq.shape[0]
            self.maps_[c] = {k: float(v) for k, v in freq.to_dict().items()}
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for c in self.cols:
            X[c + "_freq"] = pd.to_numeric(X[c].map(self.maps_[c]), errors="coerce").astype(float)
        return X


class RobustScaleSmoothClipTransform:
    """Scale by the median and interquartile range (half the range if that is 0), then clip smoothly to (-3, 3)."""

    def fit(self, X: np.ndarray) -> RobustScaleSmoothClipTransform:
        self._median = np.median(X, axis=-2)
        quant_diff = np.quantile(X, 0.75, axis=-2) - np.quantile(X, 0.25, axis=-2)
        idxs = quant_diff == 0.0
        quant_diff[idxs] = 0.5 * (np.max(X, axis=-2)[idxs] - np.min(X, axis=-2)[idxs])
        factors = 1.0 / (quant_diff + 1e-30)
        # features that are constant on the training data stay constant
        factors[quant_diff == 0.0] = 0.0
        self._factors = factors
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        x_scaled = self._factors[None, :] * (X - self._median[None, :])
        return x_scaled / np.sqrt(1 + (x_scaled / 3) ** 2)


def _fit_noisy_quantile_transformer(values: np.ndarray, *, float32_noise: bool) -> QuantileTransformer:
    """Fit a normal-output QuantileTransformer on ``values`` plus tiny per-column noise.

    The noise breaks ties between duplicated values. It is drawn in float32 for the category-embedding branch and
    in float64 otherwise (as in the TabArena benchmark).
    """
    quantile_noise = 1e-4
    np.random.seed(42)
    quantile_train = np.array(values, dtype=np.float32, copy=True)
    stds = np.nanstd(quantile_train, axis=0, keepdims=True)
    noise_std = quantile_noise / np.maximum(stds, quantile_noise)
    noise = np.random.randn(*quantile_train.shape)
    if float32_noise:
        noise = noise.astype(np.float32)
    quantile_train += noise_std * noise

    normalizer = QuantileTransformer(
        n_quantiles=min(quantile_train.shape[0], 1000),
        output_distribution="normal",
    )
    normalizer.fit(quantile_train.astype(np.float32))
    return normalizer


def _fit_normalizer(values: np.ndarray, use_robust_scale_smoothing: bool, *, float32_noise: bool = False):
    if use_robust_scale_smoothing:
        return RobustScaleSmoothClipTransform().fit(np.asarray(values, dtype=np.float32))
    return _fit_noisy_quantile_transformer(values, float32_noise=float32_noise)


class GRANDEPreprocessor:
    """Fit/transform the GRANDE input pipeline (see module docstring).

    ``fit_transform`` / ``transform`` return the float32 model input. After fitting, ``num_columns_indices`` and
    ``encoded_columns_indices`` give the positions of the numeric and encoded (non-raw) categorical columns in the
    output matrix; the remaining columns are raw category codes for the category embeddings.
    """

    def __init__(
        self,
        problem_type: str,
        use_category_embeddings: bool = False,
        use_numeric_embeddings: bool = False,
        embedding_threshold: int = 1,
        loo_cardinality: int = 10,
        missing_values: bool = False,
        use_freq_enc: bool = False,
        use_robust_scale_smoothing: bool = False,
    ):
        self.problem_type = problem_type
        self.use_category_embeddings = use_category_embeddings
        self.use_numeric_embeddings = use_numeric_embeddings
        self.embedding_threshold = embedding_threshold
        self.loo_cardinality = loo_cardinality
        self.missing_values = missing_values
        self.use_freq_enc = use_freq_enc
        self.use_robust_scale_smoothing = use_robust_scale_smoothing

    def fit_transform(self, X: pd.DataFrame, y: np.ndarray) -> np.ndarray:
        """Fit on training data. ``y`` must be float32 (class codes or regression targets)."""
        self.mean = np.mean(y)
        self.std = np.std(y)
        return self._transform(X, y=y, is_train=True)

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        return self._transform(X)

    def _transform(self, X: pd.DataFrame, y: np.ndarray | None = None, is_train: bool = False) -> np.ndarray:
        X = X.copy()
        X = X.astype({col: "category" for col in X.select_dtypes(include=["object", "string"]).columns})

        if is_train:
            self._category_encoder = CategoricalCodeEncoder().fit(X)
            self._cat_features = list(self._category_encoder.features_in)
        if self._category_encoder.features_in:
            X[self._category_encoder.features_in] = self._category_encoder.transform(X)

        if self.use_category_embeddings:
            return self._transform_category_embeddings(X, is_train)
        if self.use_numeric_embeddings:
            # binary numeric features are embedded like the other numeric features
            return self._transform_encoded(X, y, is_train, treat_binary_as_categorical=False,
                                           max_one_hot_levels=self.loo_cardinality)
        return self._transform_encoded(X, y, is_train, treat_binary_as_categorical=True,
                                       max_one_hot_levels=_MAX_ONE_HOT_LEVELS)

    def _is_categorical(self, column, n_unique: int, treat_binary_as_categorical: bool) -> bool:
        return column in self._cat_features or (treat_binary_as_categorical and n_unique <= 2)

    def _split_columns(self, X: pd.DataFrame, treat_binary_as_categorical: bool, max_one_hot_levels: int):
        """Sort the columns into numeric, binary, low- and high-cardinality categorical columns."""
        self.binary_indices, self.low_cardinality_indices, self.high_cardinality_indices = [], [], []
        self.num_columns, self.cat_columns = [], []
        for column_index, column in enumerate(X.columns):
            n_unique = X.iloc[:, column_index].nunique(dropna=True)
            if self._is_categorical(column, n_unique, treat_binary_as_categorical):
                if n_unique <= 2:
                    self.binary_indices.append(column)
                elif n_unique < max_one_hot_levels:
                    self.low_cardinality_indices.append(column)
                else:
                    self.high_cardinality_indices.append(column)
                self.cat_columns.append(column)
            else:
                self.num_columns.append(column)

    def _fit_imputation(self, X: pd.DataFrame):
        # All numeric columns are filled with the training mean of the *first* numeric column (as in the TabArena
        # benchmark); categorical columns with their training mode.
        if len(self.num_columns) > 0:
            self.mean_train_num = X[self.num_columns].mean(axis=0).iloc[0]
        if len(self.cat_columns) > 0:
            self.mode_train_cat = X[self.cat_columns].mode(axis=0).iloc[0]

    def _impute(self, X: pd.DataFrame) -> pd.DataFrame:
        if len(self.num_columns) > 0:
            X[self.num_columns] = X[self.num_columns].fillna(self.mean_train_num)
        if len(self.cat_columns) > 0:
            X[self.cat_columns] = X[self.cat_columns].fillna(self.mode_train_cat)
        return X

    def _fit_loo(self, X: pd.DataFrame, y: np.ndarray, cols) -> ce.LeaveOneOutEncoder:
        encoder = ce.LeaveOneOutEncoder(cols=cols, handle_missing="return_nan")
        if self.problem_type == "regression":
            encoder.fit(X, (y - self.mean) / self.std)
        else:
            encoder.fit(X, y)
        return encoder

    # ----------------------------------------------------------------------------------------------
    # No embeddings / numeric embeddings: ordinal (binary), one-hot (few levels), leave-one-out (many levels)
    # ----------------------------------------------------------------------------------------------
    def _transform_encoded(self, X: pd.DataFrame, y, is_train: bool, treat_binary_as_categorical: bool,
                           max_one_hot_levels: int) -> np.ndarray:
        if is_train:
            self._split_columns(X, treat_binary_as_categorical, max_one_hot_levels)
            if not self.missing_values:
                self._fit_imputation(X)
                X = self._impute(X)
            if self.use_freq_enc:
                self.freq_enc = FrequencyEncoder(cols=self.cat_columns).fit(X)

        if self.use_freq_enc:
            X = self.freq_enc.transform(X)

        if is_train:
            # fit on the frame with the frequency columns that the encoders are applied to
            self.encoder_ordinal = ce.OrdinalEncoder(cols=self.binary_indices, handle_missing="return_nan").fit(X)
            self.encoder_loo = self._fit_loo(X, y, self.high_cardinality_indices)
            self.encoder_ohe = ce.OneHotEncoder(cols=self.low_cardinality_indices, handle_missing="return_nan").fit(X)

        if not self.missing_values:
            X = self._impute(X)
        X = self.encoder_ohe.transform(self.encoder_loo.transform(self.encoder_ordinal.transform(X)))

        if is_train:
            self.num_columns_indices = [X.columns.get_loc(col) for col in self.num_columns]
            self.encoded_columns_indices = [X.columns.get_loc(col) for col in X.columns if col not in self.num_columns]

        X = X.astype(np.float32)
        if is_train:
            self.normalizer = _fit_normalizer(X.values, self.use_robust_scale_smoothing)
        return self.normalizer.transform(X.values.astype(np.float32))

    # ----------------------------------------------------------------------------------------------
    # Category embeddings: raw codes for categoricals with >= embedding_threshold levels
    # ----------------------------------------------------------------------------------------------
    def _transform_category_embeddings(self, X: pd.DataFrame, is_train: bool) -> np.ndarray:
        input_columns = list(X.columns)
        if is_train:
            # levels below embedding_threshold are one-hot encoded; binary columns ordinal encoded
            self._split_columns(X, treat_binary_as_categorical=True, max_one_hot_levels=self.embedding_threshold)
            self.columns_to_encode = self.binary_indices + self.low_cardinality_indices
            if not self.missing_values:
                self._fit_imputation(X)

        if not self.missing_values:
            X = self._impute(X)

        if is_train:
            self.encoder_ordinal = ce.OrdinalEncoder(cols=self.binary_indices, handle_missing="return_nan").fit(X)
            if self.embedding_threshold > 1:
                if self.use_freq_enc:
                    self.freq_enc = FrequencyEncoder(cols=self.low_cardinality_indices).fit(X)
                self.encoder_ohe = ce.OneHotEncoder(cols=self.low_cardinality_indices, handle_missing="return_nan").fit(X)

        X = self.encoder_ordinal.transform(X)
        if self.embedding_threshold > 1:
            if self.use_freq_enc:
                X = self.freq_enc.transform(X)
            X = self.encoder_ohe.transform(X)

        if is_train:
            self.encoder_ordinal_emb = ce.OrdinalEncoder(cols=self.high_cardinality_indices, handle_missing="return_nan").fit(X)
        X = self.encoder_ordinal_emb.transform(X)

        # replace unknowns (-1) with max+1 per column; the max is taken on the training data so that the
        # encoding of a row does not depend on the other rows it is transformed with
        if is_train:
            self.max_vals = {col: X[col].max(skipna=True) for col in self.cat_columns}
        for col in self.cat_columns:
            X.loc[X[col] == -1, col] = self.max_vals[col] + 1

        X = X.astype(np.float32)

        if is_train:
            # encoded: binary / one-hot / frequency columns (all columns except the raw codes and numeric features)
            raw_codes = [col for col in input_columns if col not in self.columns_to_encode]
            self.encoded_columns_indices = [X.columns.get_loc(col) for col in X.columns
                                            if col not in raw_codes and col not in self.num_columns]
            self.num_columns_indices = [X.columns.get_loc(col) for col in self.num_columns]

        normalize_idx = self.num_columns_indices + self.encoded_columns_indices
        if len(normalize_idx) > 0:
            if is_train:
                self.normalizer = _fit_normalizer(
                    X.iloc[:, normalize_idx].to_numpy(dtype=np.float32), self.use_robust_scale_smoothing, float32_noise=True
                )
            X_num = self.normalizer.transform(X.iloc[:, normalize_idx].to_numpy(dtype=np.float32))
            X = X.copy()
            X.iloc[:, normalize_idx] = X_num

        return X.values.astype(np.float32)
