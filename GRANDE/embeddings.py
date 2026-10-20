"""Combination of preprocessed features with learnable numeric and categorical embeddings."""

from __future__ import annotations

import torch


def embed_data(X, num_features, categorical_features_raw_indices, encoded_columns_indices, numeric_embeddings=None,
               category_embeddings=None):
    """Network input from the preprocessed features ``X``.

    Numeric features (``num_features``) and/or raw category codes (``categorical_features_raw_indices``) are
    replaced by their tanh-activated embeddings, followed by the encoded columns and the features that are not
    embedded. Without (applicable) embeddings, ``X`` is returned unchanged.
    """
    embed_num = numeric_embeddings is not None and len(num_features) > 0
    embed_cat = category_embeddings is not None and len(categorical_features_raw_indices) > 0
    if not embed_num and not embed_cat:
        return X
    parts = []
    if embed_num:
        parts.append(torch.tanh(numeric_embeddings(X[:, num_features]).flatten(start_dim=1)))
    if embed_cat:
        codes = X[:, categorical_features_raw_indices].to(torch.long)
        parts.append(torch.tanh(category_embeddings(codes).flatten(start_dim=1)))
    if len(encoded_columns_indices) > 0:
        parts.append(X[:, encoded_columns_indices])
    if not embed_cat and len(categorical_features_raw_indices) > 0:
        parts.append(X[:, categorical_features_raw_indices])
    if not embed_num and len(num_features) > 0:
        parts.append(X[:, num_features])
    return torch.cat(parts, dim=1).to(torch.float32) if len(parts) > 1 else parts[0]
