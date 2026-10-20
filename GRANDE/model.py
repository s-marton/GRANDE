"""The GRANDE network: an ensemble of hard, axis-aligned decision trees trained with gradient descent."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ._utils import set_seed, vprint


class GRANDE_Module(nn.Module):
    """Ensemble of hard, axis-aligned decision trees learned end-to-end with gradient descent.

    Each estimator selects a random feature subset, learns split thresholds and (straight-through
    one-hot) split feature indices per internal node, leaf values, and instance-wise leaf weights
    that combine the estimators.

    Codebase: https://github.com/s-marton/GRANDE
    Paper: https://openreview.net/forum?id=XEFWBxi075
    License: MIT license
    """

    def __init__(
        self,
        *,
        number_of_variables: int,
        number_of_classes: int,
        objective: str,
        device: str,
        depth: int = 5,
        n_estimators: int = 1024,
        dropout: float = 0.2,
        selected_variables: float = 0.8,
        data_subset_fraction: float = 1.0,
        bootstrap: bool = False,
        missing_values: bool = False,
        batch_size: int = 256,
        random_seed: int = 42,
        verbose: int = 0,
    ):
        super().__init__()
        if n_estimators == 1:  # a single tree uses all features and all data, without dropout
            selected_variables, data_subset_fraction, bootstrap, dropout = 1.0, 1.0, False, 0.0
        self.number_of_variables = number_of_variables
        self.number_of_classes = number_of_classes
        self.objective = objective
        self.device = device
        self.depth = depth
        self.n_estimators = n_estimators
        self.dropout = dropout
        self.selected_variables = selected_variables
        self.data_subset_fraction = data_subset_fraction
        self.bootstrap = bootstrap
        self.missing_values = missing_values
        self.batch_size = batch_size
        self.random_seed = random_seed
        self.verbose = verbose

        self.internal_node_num_ = 2 ** self.depth - 1
        self.leaf_node_num_ = 2 ** self.depth

        set_seed(self.random_seed)

        if self.selected_variables > 1:
            self.selected_variables = min(self.selected_variables, self.number_of_variables)
        else:
            self.selected_variables = int(self.number_of_variables * self.selected_variables)
            self.selected_variables = min(self.selected_variables, 50)
            self.selected_variables = max(self.selected_variables, 10)
            self.selected_variables = min(self.selected_variables, self.number_of_variables)

        if self.n_estimators > 1:
            features_by_estimator = torch.stack([torch.tensor(np.random.choice(self.number_of_variables, size=(self.selected_variables), replace=False, p=None), device=self.device, dtype=torch.int64) for _ in range(self.n_estimators)])
        else:
            self.selected_variables = self.number_of_variables
            features_by_estimator = torch.tensor(np.array([np.random.choice(self.number_of_variables, size=(self.number_of_variables), replace=False, p=None)]), device=self.device, dtype=torch.int64)
        self.register_buffer("features_by_estimator", features_by_estimator)

        path_identifier_list = []
        internal_node_index_list = []
        for leaf_index in range(self.leaf_node_num_):
            for current_depth in range(1, self.depth + 1):
                path_identifier = (leaf_index // (2 ** (self.depth - current_depth))) % 2
                internal_node_index = (2 ** (current_depth - 1)) + (leaf_index // 2 ** (self.depth - (current_depth - 1))) - 1
                path_identifier_list.append(path_identifier)
                internal_node_index_list.append(internal_node_index)

        path_identifier_list = torch.tensor(np.reshape(np.array(path_identifier_list), (-1, self.depth)), dtype=torch.long)
        self.register_buffer("path_identifier_list", path_identifier_list)

        internal_node_index_list = torch.tensor(np.reshape(np.array(internal_node_index_list), (-1, self.depth)), dtype=torch.long)
        self.register_buffer("internal_node_index_list", internal_node_index_list)

        if self.data_subset_fraction < 1.0 or self.bootstrap:
            subset_size = max(4, int(self.batch_size * self.data_subset_fraction))
            vprint(self.verbose, 1, f"[SUBSET] Using data subset size of {subset_size} for each estimator "
                   f"(data_subset_fraction={self.data_subset_fraction}, bootstrap={self.bootstrap})")
            if self.bootstrap:
                data_select = torch.randint(
                    low=0,
                    high=self.batch_size,
                    size=(self.n_estimators, subset_size),
                    device=self.device,
                    dtype=torch.long,
                )
            else:
                data_select = torch.stack(
                    [
                        torch.randperm(self.batch_size, device=self.device, dtype=torch.long)[:subset_size]
                        for _ in range(self.n_estimators)
                    ],
                    dim=0,
                )
            # Number of times each batch row is selected (rows selected by no estimator count as 1)
            counts = torch.bincount(data_select.view(-1), minlength=self.batch_size).to(torch.float32).clamp_min(1.0)
            self.register_buffer("data_select", data_select)
            self.register_buffer("counts", counts)

        set_seed(self.random_seed)
        self.split_values = nn.Parameter(torch.zeros([self.n_estimators, self.internal_node_num_, self.selected_variables], device=self.device, dtype=torch.float32), requires_grad=True)
        nn.init.normal_(self.split_values, mean=0.0, std=0.05)

        self.split_index_array = nn.Parameter(torch.zeros([self.n_estimators, self.internal_node_num_, self.selected_variables], device=self.device, dtype=torch.float32), requires_grad=True)
        nn.init.normal_(self.split_index_array, mean=0.0, std=0.05)

        self.estimator_weights = nn.Parameter(torch.zeros([self.n_estimators, self.leaf_node_num_], device=self.device, dtype=torch.float32), requires_grad=True)
        nn.init.normal_(self.estimator_weights, mean=0.0, std=0.05)

        leaf_classes_array_shape = [self.n_estimators, self.leaf_node_num_] if self.objective in ("binary", "regression") else [self.n_estimators, self.leaf_node_num_, self.number_of_classes]
        self.leaf_classes_array = nn.Parameter(torch.zeros(leaf_classes_array_shape, device=self.device, dtype=torch.float32), requires_grad=True)
        nn.init.normal_(self.leaf_classes_array, mean=0.0, std=0.05)

    def forward(self, inputs):
        # einsum syntax:
        #       - b is the batch size
        #       - e is the number of estimators
        #       - l the number of leaf nodes  (i.e. the number of paths)
        #       - i is the number of internal nodes
        #       - d is the depth (i.e. the length of each path)
        #       - n is the number of variables (one value is stored for each variable)

        features_indices = self.features_by_estimator
        # Adjust data: For each estimator select a subset of the features (output shape: (b, e, n))
        X_estimator = inputs[:, features_indices]
        if self.data_subset_fraction < 1.0 and self.training:
            # Sample selection: each estimator only sees its own subset of the batch -> (N_sub, E, F)
            X_estimator = X_estimator.permute(1, 0, 2)
            estimator_idx = torch.arange(X_estimator.size(0), device=inputs.device).unsqueeze(1)
            X_estimator = X_estimator[estimator_idx, self.data_select]
            X_estimator = X_estimator.transpose(0, 1)

        if self.missing_values:
            nan_mask = (torch.isnan(X_estimator)).float()
            X_estimator = torch.nan_to_num(X_estimator, nan=0.0)

        # straight-through one-hot selection of the split feature per internal node
        split_index_array = F.softmax(self.split_index_array, dim=-1)
        adjust_constant = split_index_array - F.one_hot(torch.argmax(split_index_array, dim=-1), num_classes=split_index_array.shape[-1])
        split_index_array = split_index_array - adjust_constant.detach()

        # as split_index_array is one-hot-encoded, taking the sum over the last axis after multiplication
        # results in selecting the desired value at the index
        s1_sum = torch.einsum("ein,ein->ei", self.split_values, split_index_array)
        s2_sum = torch.einsum("ben,ein->bei", X_estimator, split_index_array)

        # calculate the split (output shape: (b, e, i))
        node_result = (F.softsign(s1_sum - s2_sum) + 1) / 2
        # use round operation with ST operator to get hard decision for each node
        adjust_constant = node_result - torch.round(node_result)
        node_result_corrected = node_result - adjust_constant.detach()

        # Select internal nodes along each path (output shape: (b, e, l, d))
        internal_node_index_list = self.internal_node_index_list
        node_result_extended_selected = node_result_corrected[..., internal_node_index_list]

        # Separate left/right; for missing values route by the fraction of samples going left
        left = node_result_extended_selected
        right = 1.0 - node_result_extended_selected
        if self.missing_values:
            masked_selected = torch.einsum("ben,ein->bei", nan_mask, split_index_array)
            masked_ext = masked_selected[..., internal_node_index_list].to(dtype=torch.bool)

            smaller_count = torch.einsum("bei->ei", node_result)
            smaller_count = smaller_count.unsqueeze(0).expand(X_estimator.shape[0], -1, -1)  # [n, e, i]
            smaller_prob = smaller_count / X_estimator.shape[0]

            smaller_prob_ext = smaller_prob[..., internal_node_index_list]

            left = torch.where(masked_ext, smaller_prob_ext, left)
            right = torch.where(masked_ext, 1 - smaller_prob_ext, right)

        # reduce the path via multiplication to get result for each path (in each estimator) based on the
        # results of the corresponding internal nodes (output shape: (b, e, l))
        p = torch.prod(((1 - self.path_identifier_list) * left + self.path_identifier_list * right), dim=3)

        # calculate instance-wise leaf weights for each estimator by selecting the weight of the selected path
        estimator_weights_leaf = torch.einsum("el,bel->be", self.estimator_weights, p)

        # use softmax over weights for each instance
        estimator_weights_leaf_softmax = F.softmax(estimator_weights_leaf, dim=-1)

        # optional dropout (deactivating random estimators), renormalized per instance
        if self.training and self.dropout > 0.0:
            estimator_weights_leaf_softmax = F.dropout(estimator_weights_leaf_softmax, p=self.dropout, training=True)
            estimator_weights_leaf_softmax = estimator_weights_leaf_softmax / estimator_weights_leaf_softmax.sum(dim=1, keepdim=True).clamp_min(1e-8)

        p_weighted = torch.einsum("bel,be->bel", p, estimator_weights_leaf_softmax)

        if self.objective in ("regression", "binary"):
            layer_output = torch.einsum("el,bel->be", self.leaf_classes_array, p_weighted)
        else:
            layer_output = torch.einsum("elc,bel->bec", self.leaf_classes_array, p_weighted)

        if self.data_subset_fraction < 1.0 and self.training:
            # each estimator saw its own subset of the batch: scatter the outputs back to the full batch and
            # average over the estimators that saw each row
            B = inputs.size(0)
            if self.objective == "multiclass":
                # (S_e, E, C) -> (E, S_e, C), accumulated into (B, C)
                updates = layer_output.permute(1, 0, 2).contiguous()
                C = updates.size(-1)
                idx = self.data_select.unsqueeze(-1).expand(-1, -1, C)
                result = torch.zeros(B, C, dtype=updates.dtype, device=inputs.device)
                result.scatter_add_(0, idx.reshape(-1, C), updates.reshape(-1, C))
                result = (result / self.counts.unsqueeze(-1)) * self.n_estimators
            else:
                base = torch.zeros(B, dtype=layer_output.dtype, device=inputs.device)
                result = torch.scatter_add(base, 0, self.data_select.reshape(-1), layer_output.t().reshape(-1))
                result = (result / self.counts) * self.n_estimators
        elif self.objective == "multiclass":
            result = torch.einsum("bec->bc", layer_output)
        else:
            result = torch.einsum("be->b", layer_output)

        if self.objective == "binary":
            result = torch.stack([-result, result], dim=1)
        return result

    @torch.no_grad()
    def routing(self, inputs):
        """Leaf assignment and instance-wise estimator weights at inference (eval mode, no dropout).

        Mirrors the inference path of :meth:`forward`. Returns ``p`` of shape (b, e, l) with the path
        indicator of every leaf (one-hot unless ``missing_values`` routes a missing value through both
        children) and ``weights`` of shape (b, e), the softmax-normalized estimator weights per sample.
        """
        X_estimator = inputs[:, self.features_by_estimator]
        if self.missing_values:
            nan_mask = (torch.isnan(X_estimator)).float()
            X_estimator = torch.nan_to_num(X_estimator, nan=0.0)

        split_index_array = F.softmax(self.split_index_array, dim=-1)
        adjust_constant = split_index_array - F.one_hot(torch.argmax(split_index_array, dim=-1), num_classes=split_index_array.shape[-1])
        split_index_array = split_index_array - adjust_constant.detach()

        s1_sum = torch.einsum("ein,ein->ei", self.split_values, split_index_array)
        s2_sum = torch.einsum("ben,ein->bei", X_estimator, split_index_array)
        node_result = (F.softsign(s1_sum - s2_sum) + 1) / 2
        node_result_corrected = node_result - (node_result - torch.round(node_result))

        internal_node_index_list = self.internal_node_index_list
        left = node_result_corrected[..., internal_node_index_list]
        right = 1.0 - left
        if self.missing_values:
            masked_selected = torch.einsum("ben,ein->bei", nan_mask, split_index_array)
            masked_ext = masked_selected[..., internal_node_index_list].to(dtype=torch.bool)
            smaller_prob = torch.einsum("bei->ei", node_result).unsqueeze(0).expand(X_estimator.shape[0], -1, -1) / X_estimator.shape[0]
            smaller_prob_ext = smaller_prob[..., internal_node_index_list]
            left = torch.where(masked_ext, smaller_prob_ext, left)
            right = torch.where(masked_ext, 1 - smaller_prob_ext, right)

        p = torch.prod(((1 - self.path_identifier_list) * left + self.path_identifier_list * right), dim=3)
        weights = F.softmax(torch.einsum("el,bel->be", self.estimator_weights, p), dim=-1)
        return p, weights
