from __future__ import annotations

__version__ = '0.0.12'

__all__ = [
    'LinearEmbeddings',
    'LinearReLUEmbeddings',
    'PeriodicEmbeddings',
    'PiecewiseLinearEmbeddings',
    'PiecewiseLinearEncoding',
    'compute_bins',
]


# --- stdlib ---
import math
import os
import random
import time
import warnings
import copy
from pprint import pprint
from typing import Any, Literal, Optional, Union

# --- third-party ---
import numpy as np
import pandas as pd
import sklearn
import category_encoders as ce
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parameter import Parameter
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)
from autogluon.features.generators import LabelEncoderFeatureGenerator

try:
    import sklearn.tree as sklearn_tree
except ImportError:
    sklearn_tree = None

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


torch.set_float32_matmul_precision('high')



def embed_data(X, num_features, categorical_features_raw_indices, encoded_columns_indices, numeric_embeddings=None, category_embeddings=None, preprocessed_data=None):
    """
    Embeds data using numeric and/or categorical embeddings.
    Optionally concatenates preprocessed data if provided.
    
    Args:
        X: Input tensor (normalized data for embeddings)
        num_features: Indices of numeric features
        categorical_features_raw_indices: Indices of categorical features (raw, for embeddings)
        encoded_columns_indices: Indices of encoded columns
        numeric_embeddings: Optional numeric embeddings module
        category_embeddings: Optional category embeddings module
        preprocessed_data: Optional preprocessed data tensor to concatenate
    """
    if (category_embeddings is None and numeric_embeddings is None):
        if preprocessed_data is not None:
            return torch.cat([X, preprocessed_data], dim=1).to(torch.float32)
        return X
    elif (category_embeddings is not None and numeric_embeddings is None and len(categorical_features_raw_indices) == 0):
        if preprocessed_data is not None:
            return torch.cat([X, preprocessed_data], dim=1).to(torch.float32)
        return X
    elif (category_embeddings is None and numeric_embeddings is not None and len(num_features) == 0):
        if preprocessed_data is not None:
            return torch.cat([X, preprocessed_data], dim=1).to(torch.float32)
        return X

    if len(num_features) > 0:
        data_num = X[:, num_features]
    if len(encoded_columns_indices) > 0:
        data_enc = X[:, encoded_columns_indices]      
    if len(categorical_features_raw_indices) > 0:
        data_cat = X[:, categorical_features_raw_indices]


    apply_category_embeddings = (category_embeddings is not None and len(categorical_features_raw_indices) > 0)
    apply_numeric_embeddings = (numeric_embeddings is not None and len(num_features) > 0)

    if apply_category_embeddings and apply_numeric_embeddings:
        data_cat = data_cat.to(torch.long)

        #self.category_embeddings(x_cat + self.category_offsets[None])
        embedded_cat_features = category_embeddings(data_cat)
        embedded_cat_features = embedded_cat_features.flatten(start_dim=1)
        embedded_cat_features = F.tanh(embedded_cat_features)


        embedded_num_features = numeric_embeddings(data_num)
        embedded_num_features = embedded_num_features.flatten(start_dim=1)
        embedded_num_features = F.tanh(embedded_num_features)


        if len(encoded_columns_indices) > 0:
            embedded_combined = torch.cat([embedded_num_features, embedded_cat_features], dim=1).to(torch.float32) 
            embedded_result = torch.cat([embedded_combined, X[:, encoded_columns_indices]], dim=1).to(torch.float32)
        else:
            embedded_result = torch.cat([embedded_num_features, embedded_cat_features], dim=1).to(torch.float32)
        
        if preprocessed_data is not None:
            return torch.cat([embedded_result, preprocessed_data], dim=1).to(torch.float32)
        return embedded_result

    if not apply_category_embeddings and apply_numeric_embeddings:
        embedded_num_features = numeric_embeddings(data_num)
        embedded_num_features = embedded_num_features.flatten(start_dim=1)
        embedded_num_features = F.tanh(embedded_num_features)

        embedded_data = embedded_num_features

        if len(encoded_columns_indices) > 0:
            embedded_data = torch.cat([embedded_data, X[:, encoded_columns_indices]], dim=1).to(torch.float32) 

        if len(categorical_features_raw_indices) > 0:
            embedded_data = torch.cat([embedded_data, X[:, categorical_features_raw_indices]], dim=1).to(torch.float32)

        if preprocessed_data is not None:
            return torch.cat([embedded_data, preprocessed_data], dim=1).to(torch.float32)
        return embedded_data

    if apply_category_embeddings and not apply_numeric_embeddings:
        data_cat = data_cat.to(torch.long)

        #category_embeddings(x_cat + self.category_offsets[None])
        embedded_cat_features = category_embeddings(data_cat)
        embedded_cat_features = embedded_cat_features.flatten(start_dim=1)
        embedded_cat_features = F.tanh(embedded_cat_features)

        embedded_data = embedded_cat_features

        if len(encoded_columns_indices) > 0:
            embedded_data = torch.cat([embedded_data, X[:, encoded_columns_indices]], dim=1).to(torch.float32) 

        if len(num_features) > 0:
            embedded_data = torch.cat([embedded_data, X[:, num_features]], dim=1).to(torch.float32)

        if preprocessed_data is not None:
            return torch.cat([embedded_data, preprocessed_data], dim=1).to(torch.float32)
        return embedded_data


    return None



class GRANDE_Module(nn.Module):

    """ 
    GRANDE is a novel ensemble method for hard, axis-aligned decision trees learned end-to-end with gradient descent.
    
    Codebase: https://github.com/s-marton/GRANDE
    Paper: https://openreview.net/forum?id=XEFWBxi075
    License: MIT license
    """

    def __init__(self, params):
        super().__init__()

        self.config = None

        self.set_params(**params)

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
            for current_depth in range(1, self.depth+1):
                path_identifier = ((leaf_index // (2 ** (self.depth - current_depth))) % 2)
                internal_node_index = ((2 ** (current_depth-1)) + (leaf_index // 2 ** (self.depth-(current_depth-1))) - 1)
                path_identifier_list.append(path_identifier)
                internal_node_index_list.append(internal_node_index)

        #path_identifier_list = nn.Parameter(torch.tensor(np.reshape(np.array(path_identifier_list), (-1,self.depth)), device=self.device, dtype=torch.int64), requires_grad=False)
        path_identifier_list = torch.tensor(np.reshape(np.array(path_identifier_list), (-1,self.depth)), dtype=torch.long)
        self.register_buffer("path_identifier_list", path_identifier_list)  

        #internal_node_index_list = nn.Parameter(torch.tensor(np.reshape(np.array(self.internal_node_index_list), (-1,self.depth)), device=self.device, dtype=torch.int64), requires_grad=False)
        internal_node_index_list = torch.tensor(np.reshape(np.array(internal_node_index_list), (-1,self.depth)), dtype=torch.long)
        self.register_buffer("internal_node_index_list", internal_node_index_list)  # shape [K]

        if (self.data_subset_fraction < 1.0 or self.bootstrap == True):
            subset_size = max(4, int(self.batch_size * self.data_subset_fraction))
            print('[SUBSET] Using data subset size of ', subset_size, ' for each estimator (data_subset_fraction=', self.data_subset_fraction, ', bootstrap=', self.bootstrap, ')')
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
            # Compute counts like np.unique with return_counts=True
            _, counts = torch.unique(data_select.view(-1), return_counts=True)
            counts = counts.to(torch.float32)
            self.register_buffer("data_select", data_select)
            self.register_buffer("counts", counts)

        set_seed(self.random_seed)
        self.initializer = nn.init.normal_

        self.split_values = nn.Parameter(torch.zeros([self.n_estimators,self.internal_node_num_, self.selected_variables], device=self.device, dtype=torch.float32), requires_grad=True)
        self.initializer(self.split_values, mean=0.0, std=0.05)        

        self.split_index_array = nn.Parameter(torch.zeros([self.n_estimators,self.internal_node_num_, self.selected_variables], device=self.device, dtype=torch.float32), requires_grad=True)
        self.initializer(self.split_index_array, mean=0.0, std=0.05) 

        self.estimator_weights = nn.Parameter(torch.zeros([self.n_estimators,self.leaf_node_num_], device=self.device, dtype=torch.float32), requires_grad=True)
        self.initializer(self.estimator_weights, mean=0.0, std=0.05) 

        leaf_classes_array_shape = [self.n_estimators,self.leaf_node_num_,] if self.objective == 'binary' or self.objective == 'regression' else [self.n_estimators, self.leaf_node_num_, self.number_of_classes]
        self.leaf_classes_array = nn.Parameter(torch.zeros(leaf_classes_array_shape, device=self.device, dtype=torch.float32), requires_grad=True)
        self.initializer(self.leaf_classes_array, mean=0.0, std=0.05)     


    def forward(self, inputs, return_per_estimator_logits=False):
        # einsum syntax:         
        #       - b is the batch size
        #       - e is the number of estimators
        #       - l the number of leaf nodes  (i.e. the number of paths)
        #       - i is the number of internal nodes
        #       - d is the depth (i.e. the length of each path)
        #       - n is the number of variables (one value is stored for each variable)        

        features_indices = self.features_by_estimator 
        # Adjust data: For each estimator select a subset of the features (output shape: (b, e, n))
        #with torch.no_grad():
        X_estimator = inputs[:, features_indices] #if not (self.data_subset_fraction < 1.0 and self.training) else self.get_subset(inputs, features_indices)
        if self.data_subset_fraction < 1.0 and self.training:
            if False:
                X_estimator = X_estimator.permute(1, 0, 2)          # (E, B, F_e)
                idx = self.data_select.unsqueeze(-1).expand(-1, -1, X_estimator.size(-1))  # (E, S_e, F_e)
                X_estimator = torch.gather(X_estimator, dim=1, index=idx)             # (E, S_e, F_e)
                X_estimator = X_estimator.permute(1, 0, 2)        

            else:
                X_estimator = X_estimator.permute(1, 0, 2)
                
                # 2. Sample Selection
                # TF equivalent: tf.gather_nd(..., self.data_select, batch_dims=1)
                # We create an index for the estimators [0, 1, ... E] to pair with self.data_select
                estimator_idx = torch.arange(X_estimator.size(0), device=inputs.device).unsqueeze(1)

                # We use advanced indexing to pick specific samples for each estimator
                # Shape becomes: (Estimator, N_sub, Feature)
                X_estimator = X_estimator[estimator_idx, self.data_select]

                # 3. Final Transpose
                # TF equivalent: tf.transpose(..., [1,0,2])
                # Final Shape: (N_sub, Estimator, Feature)
                X_estimator = X_estimator.transpose(0, 1)    

        if self.missing_values:        
            nan_mask = (torch.isnan(X_estimator)).float()
            X_estimator = torch.nan_to_num(X_estimator, nan=0.0)
            num_nans = torch.isnan(X_estimator).sum().item()
            #print(f"Number of NaNs: {num_nans}")

        #entmax transformaton
        split_index_array = self.split_index_array
        #split_index_array = entmax15(split_index_array)
        split_index_array = F.softmax(self.split_index_array, dim=-1)
        adjust_constant = split_index_array - F.one_hot(torch.argmax(split_index_array, dim=-1), num_classes=split_index_array.shape[-1])
        split_index_array = split_index_array - adjust_constant.detach()

        # as split_index_array_selected is one-hot-encoded, taking the sum over the last axis after multiplication results in selecting the desired value at the index
        
        s1_sum = torch.einsum("ein,ein->ei", self.split_values, split_index_array)
        s2_sum = torch.einsum("ben,ein->bei", X_estimator, split_index_array)            

        # calculate the split (output shape: (b, e, l, d))
        node_result = (F.softsign(s1_sum-s2_sum) + 1) / 2
        #use round operation with ST operator to get hard decision for each node
        adjust_constant = node_result - torch.round(node_result)
        node_result_corrected = node_result - adjust_constant.detach()

        # Select internal nodes along each path
        internal_node_index_list = self.internal_node_index_list
        node_result_extended_selected = node_result_corrected[...,internal_node_index_list]

            
        # Separate left/right and apply mask so both paths are taken when masked
        left = node_result_extended_selected
        right = 1.0 - node_result_extended_selected
        if self.missing_values:        
            masked_selected = torch.einsum("ben,ein->bei", nan_mask, split_index_array)
            masked_ext = masked_selected[..., internal_node_index_list].to(dtype=torch.bool)

            smaller_count = torch.einsum('bei->ei', node_result)
            smaller_count = smaller_count.unsqueeze(0).expand(X_estimator.shape[0], -1, -1)  # [n, e, i]
            smaller_prob = smaller_count / X_estimator.shape[0]

            smaller_prob_ext = smaller_prob[..., internal_node_index_list]

            left = torch.where(masked_ext, smaller_prob_ext, left)
            right = torch.where(masked_ext, 1-smaller_prob_ext, right)                

        #reduce the path via multiplication to get result for each path (in each estimator) based on the results of the corresponding internal nodes (output shape: (b, e, l))
        p = torch.prod(((1-self.path_identifier_list)*left + self.path_identifier_list*right), dim=3)

        #calculate instance-wise leaf weights for each estimator by selecting the weight of the selected path for each estimator
        estimator_weights_leaf = torch.einsum("el,bel->be", self.estimator_weights, p)

        #use softmax over weights for each instance
        estimator_weights_leaf_softmax = F.softmax(estimator_weights_leaf, dim=-1)

        if self.training and self.dropout > 0.0:
            estimator_weights_leaf_softmax = F.dropout(estimator_weights_leaf_softmax, p=self.dropout, training=True)# * (1 - self.dropout)
            # Normalize along dim=1 (like tf.reduce_sum over axis=1)
            estimator_weights_leaf_softmax = estimator_weights_leaf_softmax / estimator_weights_leaf_softmax.sum(dim=1, keepdim=True).clamp_min(1e-8)

        #optional dropout (deactivating random estimators)
        #estimator_weights_leaf_softmax = self.apply_dropout_leaf(estimator_weights_leaf_softmax, training=training)                     

        p_weighted = torch.einsum('bel,be->bel', p, estimator_weights_leaf_softmax)

        if self.objective in ['regression', 'binary']:
            #layer_output = torch.einsum('el,bel,be->be', self.leaf_classes_array, p, estimator_weights_leaf_softmax)
            layer_output = torch.einsum('el,bel->be', self.leaf_classes_array, p_weighted)
        else:
            #layer_output = torch.einsum('elc,bel,be->bec', self.leaf_classes_array, p, estimator_weights_leaf_softmax)
            layer_output = torch.einsum('elc,bel->bec', self.leaf_classes_array, p_weighted)

        if self.data_subset_fraction < 1.0 and self.training:
            per_estimator_logits = None
            B = inputs.size(0)
            # --- Subset branch: scatter estimator outputs back to full batch ---
            if layer_output.dim() == 2:
                base = torch.zeros(B, dtype=layer_output.dtype, device=inputs.device)
                result = torch.scatter_add(base, 0, self.data_select.reshape(-1), layer_output.t().reshape(-1))
                result = (result / self.counts.to(result.dtype).to(inputs.device)) * self.n_estimators

            elif layer_output.dim() == 3:
                # Multiclass: (S_e, E, C) -> (E, S_e, C)
                updates = layer_output.permute(1, 0, 2).contiguous()  # (E, S_e, C)
                C = updates.size(-1)
                idx = self.data_select.unsqueeze(-1).expand(-1, -1, C)  # (E, S_e, C)
                # Accumulate into (B, C)
                result = torch.zeros(B, C, dtype=updates.dtype, device=inputs.device)
                result = result.index_add_(0, idx, updates)
                # Broadcast-safe normalization
                denom = self.counts.to(result.dtype).to(inputs.device)
                if denom.dim() == 1:
                    denom = denom.unsqueeze(-1)                        # (B,1)
                result = (result / denom) * self.n_estimators
            else:
                raise ValueError("Unexpected layer_output shape in subset branch.")

            if self.objective == 'binary':
                result = torch.stack([-result, result], dim=1)          

        else:
            if self.objective == 'regression' or self.objective == 'binary':
                per_estimator_logits = layer_output * self.n_estimators
                per_estimator_logits = torch.stack([-per_estimator_logits, per_estimator_logits], dim=2)

                result = torch.einsum('be->b', layer_output)
                if self.objective == 'binary':
                    result = torch.stack([-result, result], dim=1)            
            else:
                per_estimator_logits = layer_output * self.n_estimators
                result = torch.einsum('bec->bc', layer_output)

        if return_per_estimator_logits:
            return result, per_estimator_logits

        return result

    def set_params(self, **kwargs):
        base_defaults = {
            'depth': 5,
            'n_estimators': 1024,
            'learning_rate_weights': 0.001,
            'learning_rate_index': 0.01,
            'learning_rate_values': 0.05,
            'learning_rate_leaf': 0.05,
            # Whether to use multiple learning rates across estimators in the
            # ensemble. If True and n_estimators > 1, the estimators are split
            # into groups with different LRs around the base learning rates.
            'use_multi_estimator_lr': False,
            # LR multipliers for multi-estimator scheme. Default: [0.1, 0.3, 1.0, 3.0, 10.0]
            # Each estimator gets assigned to one group, and its effective LR is base_lr * factor.
            'estimator_lr_factors': None,  # None = use default [0.1, 0.3, 1.0, 3.0, 10.0]
            'temperature': 0.0,
            'dropout': 0.2,
            'selected_variables': 0.8,
            'data_subset_fraction': 1.0,
            'bootstrap': False,
            'random_seed': 42,
            'verbose': 2,

            'use_category_embeddings': False,
            'use_numeric_embeddings': False,
            'attach_preprocessed_data': False,  # If True, attach fully preprocessed data alongside embeddings

        }

        if getattr(self, 'config', None) is None:
            self.config = base_defaults.copy()

        # Merge user kwargs
        self.config.update(kwargs)

        # Single estimator specialization
        if self.config.get('n_estimators', 1) == 1:
            self.config['selected_variables'] = 1.0
            self.config['data_subset_fraction'] = 1.0
            self.config['bootstrap'] = False
            self.config['dropout'] = 0.0

        # Attach as attributes
        for k, v in self.config.items():
            setattr(self, k, v)

        set_seed(self.random_seed)
        return self.config

        def get_params(self):
            return self.config  

class GRANDE:
    def __init__(self, 
                 params):

        params['device'] = "cpu" if not torch.cuda.is_available() else "cuda:0"
        params['objective'] = params['problem_type']
        self.params = params

        for k, v in self.params.items():
            setattr(self, k, v)

        self.attach_preprocessed_data = getattr(self, 'attach_preprocessed_data', False)

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: pd.DataFrame = None,
        y_val: pd.Series = None,
        **kwargs,
    ):
        start_time = time.time()

        if X_val is None:
            from autogluon.core.utils import generate_train_test_split

            X_train, X_val, y_train, y_val = generate_train_test_split(
                X=X,
                y=y,
                problem_type=self.problem_type,
                test_size=0.2,
                random_state=0,
            )
        else:
            X_train, y_train = X, y

        X_train, y_train = self.preprocess(X_train, y=y_train, is_train=True)
        # Store preprocessed data for training if flag is True
        if (self.use_category_embeddings or self.use_numeric_embeddings) and self.attach_preprocessed_data:
            self.X_train_preprocessed = getattr(self, '_stored_preprocessed_data', None)
        else:
            self.X_train_preprocessed = None
            
        if X_val is not None:
            X_val, y_val = self.preprocess(X_val, y=y_val)
            # Store preprocessed data for validation if flag is True
            if (self.use_category_embeddings or self.use_numeric_embeddings) and self.attach_preprocessed_data:
                self.X_val_preprocessed = getattr(self, '_stored_preprocessed_data', None)
            else:
                self.X_val_preprocessed = None
        else:
            self.X_val_preprocessed = None


        self.number_of_variables = X_train.shape[1]
        if self.objective == 'regression':
            self.use_class_weights = False
        if self.use_class_weights:
            if self.objective == 'multiclass' or self.objective == 'binary':
                self.number_of_classes = 1 if self.objective == 'binary' else len(np.unique(y_train))
                #class_weights = sklearn.utils.class_weight.compute_class_weight(class_weight = 'balanced', classes = np.unique(y_train), y = y_train)  
                counts = np.bincount(y_train.astype(int), minlength=self.number_of_classes).astype(np.float64)
                counts[counts == 0] = 1.0
                inv = 1.0 / counts
                class_weights = inv * (self.number_of_classes / inv.sum())

            else:
                self.number_of_classes = 1
                class_weights = np.ones_like(np.unique(y_train))
        else:   
            self.number_of_classes = 1 if (self.objective == 'binary' or self.objective == 'regression') else len(np.unique(y_train))
            class_weights = np.ones_like(np.unique(y_train))
        
        class_weights = torch.tensor(class_weights, dtype=torch.float32, device=self.device)

        if self.use_category_embeddings or self.use_numeric_embeddings:
            all_indices = list(range(self.number_of_variables))
            self.categorical_features_raw_indices = [i for i in all_indices if i not in self.num_columns_indices and i not in self.encoded_columns_indices]
            
            if self.use_numeric_embeddings and self.num_columns_indices is not None and len(self.num_columns_indices) > 0:
                # Compute all other indices (non-numeric features)

                #embeddings = LinearEmbeddings(len(self.num_columns_indices), self.embedding_dim)
                bins = compute_bins(
                    torch.from_numpy(np.concatenate([X_train, X_val], axis=0)[:, self.num_columns_indices]),
                    n_bins=8,#48,
                    # NOTE: requires scikit-learn>=1.0 to be installed.
                    tree_kwargs={'min_samples_leaf': 64, 'min_impurity_decrease': 1e-4},
                    y=torch.from_numpy(np.concatenate([y_train, y_val], axis=0)),
                    regression=self.objective == 'regression',
                )


                set_seed(self.random_seed)
                numeric_embeddings = PiecewiseLinearEmbeddings(bins=bins, d_embedding=self.embedding_dim_num, activation=False, version="B",)
                set_seed(self.random_seed)
                # Get the output shape without batch dimensions:
                assert numeric_embeddings.get_output_shape() == (len(self.num_columns_indices), self.embedding_dim_num)
                # Get the total output size after flattening:
                assert numeric_embeddings.get_output_shape().numel() == len(self.num_columns_indices) * self.embedding_dim_num

            else:
                numeric_embeddings = None
                swa_numeric_embeddings = None

            if self.use_category_embeddings and self.categorical_features_raw_indices is not None and len(self.categorical_features_raw_indices) > 0:

                categories = [len(np.unique(X_train[:, idx])) + 1 for idx in self.categorical_features_raw_indices]

                #category_offsets = torch.tensor([0] + categories[:-1]).cumsum(0)
                #self.register_buffer('category_offsets', category_offsets)
                set_seed(self.random_seed)
                category_embeddings = nn.Embedding(sum(categories), self.embedding_dim_cat)
                set_seed(self.random_seed)
                nn.init.kaiming_uniform_(category_embeddings.weight, a=math.sqrt(5))
                set_seed(self.random_seed)

            else:
                category_embeddings = None
                swa_category_embeddings = None

            if self.use_category_embeddings or self.use_numeric_embeddings:
                embedding_dim_total = 0
                feature_dim_total = 0

                num_features_total = len(self.num_columns_indices)
                cat_features_raw_total = len(self.categorical_features_raw_indices)
                cat_features_encoded_total = len(self.encoded_columns_indices)

                if self.use_category_embeddings:
                    embedding_dim_total += self.embedding_dim_cat * cat_features_raw_total
                else:
                    feature_dim_total += cat_features_raw_total

                if self.use_numeric_embeddings:
                    embedding_dim_total += self.embedding_dim_num * num_features_total
                else:
                    feature_dim_total += num_features_total

                feature_dim_total += cat_features_encoded_total

                self.number_of_variables = feature_dim_total + embedding_dim_total
                
                # Add preprocessed data dimensions if flag is True
                if self.attach_preprocessed_data and self.X_train_preprocessed is not None:
                    self.number_of_variables += self.X_train_preprocessed.shape[1]
        else:
            numeric_embeddings = None
            category_embeddings = None
            swa_numeric_embeddings = None
            swa_category_embeddings = None
            self.categorical_features_raw_indices = []

        self.params['number_of_variables'] = self.number_of_variables
        self.params['number_of_classes'] = self.number_of_classes
        model = GRANDE_Module(params=self.params)
        model = model.to(self.device)
        #self.model = model
        model = torch.compile(model, mode="reduce-overhead", fullgraph=False, dynamic=False) 

        if self.use_numeric_embeddings and self.num_columns_indices is not None and len(self.num_columns_indices) > 0:
            numeric_embeddings = numeric_embeddings.to(self.device)
            #numeric_embeddings = torch.compile(numeric_embeddings)#, mode="reduce-overhead", fullgraph=False, dynamic=True)  

        if self.use_category_embeddings and self.categorical_features_raw_indices is not None and len(self.categorical_features_raw_indices) > 0:
            category_embeddings = category_embeddings.to(self.device)
            #category_embeddings = torch.compile(category_embeddings)#, mode="reduce-overhead", fullgraph=False, dynamic=True)  

        if self.swa:             
            swa_model = torch.optim.swa_utils.AveragedModel(model) #.to(self.device) #, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.999)
            #self.swa_model = torch.compile(self.swa_model, mode="reduce-overhead", fullgraph=False, dynamic=True)     
            if numeric_embeddings is not None:
                swa_numeric_embeddings = torch.optim.swa_utils.AveragedModel(numeric_embeddings)
            if category_embeddings is not None:
                swa_category_embeddings = torch.optim.swa_utils.AveragedModel(category_embeddings)
        else:
            swa_model = None
            swa_numeric_embeddings = None
            swa_category_embeddings = None

        self.model = model
        self.numeric_embeddings = numeric_embeddings
        self.category_embeddings = category_embeddings
        self.swa_model = swa_model
        self.swa_numeric_embeddings = swa_numeric_embeddings
        self.swa_category_embeddings = swa_category_embeddings

        if self.numeric_embeddings is not None:
            print('Use Numeric Embeddings:', self.numeric_embeddings, flush=True)
        if self.category_embeddings is not None:
            print('Use Category Embeddings:', self.category_embeddings, flush=True)

        self.batch_size = min(self.batch_size, X_train.shape[0]//2)
        # Include preprocessed data in datasets if flag is True
        X_train_preprocessed_for_loader = self.X_train_preprocessed if self.attach_preprocessed_data else None
        X_val_preprocessed_for_loader = self.X_val_preprocessed if self.attach_preprocessed_data else None
        train_loader, val_loader = make_datasets_and_loaders(X_train, y_train, X_val, y_val, shuffle_train=(self.data_subset_fraction == 1.0 and self.bootstrap==False), batch_size=self.batch_size, task_type=self.objective, X_train_preprocessed=X_train_preprocessed_for_loader, X_val_preprocessed=X_val_preprocessed_for_loader) #

        if val_loader is not None:
            early_stopper = EarlyStopper(
                patience=self.early_stopping_epochs, min_delta=0.001, mode="min",  # was "max"
                #get_state=lambda: {"clf": self.model.state_dict()},
                #set_state=lambda st: self.model.load_state_dict(st["clf"]),
                name=f"TRAINING",
                verbose=(self.verbose >= 2)
            )
        else:
            early_stopper = None

        # -----------------------------------------------------------------------------
        # Optimizer(s) and per-estimator learning rate scaling
        # -----------------------------------------------------------------------------
        #
        # We still use a single optimizer with one base LR per parameter type, but
        # if `use_multi_estimator_lr` is enabled we rescale the gradients for each
        # estimator so that the *effective* LR per estimator is different.
        #
        # This avoids slicing `nn.Parameter`s (which would create non-leaf tensors)
        # while still achieving different step sizes across estimators.

        n_estimators = getattr(self.model, "n_estimators", None)
        use_multi_estimator_lr = (
            getattr(self, "use_multi_estimator_lr", False)
            and n_estimators is not None
            and n_estimators > 1
        )

        # Define LR factors for multi-estimator LR scheme (used for both estimators and embeddings)
        # Allow user to customize, with sensible default
        if use_multi_estimator_lr:
            lr_factors = getattr(self, 'estimator_lr_factors', None)
            if lr_factors is None:
                lr_factors = [0.1, 0.3, 1.0, 3.0, 10.0]
            elif not isinstance(lr_factors, (list, tuple)) or len(lr_factors) < 2:
                raise ValueError(
                    f"estimator_lr_factors must be a list/tuple with at least 2 values, got: {lr_factors}"
                )
            lr_factors = list(lr_factors)  # Ensure it's a list
            n_groups = len(lr_factors)
            
            if self.verbose >= 1:
                print(f"[MULTI-LR] Using {n_groups} LR groups with factors: {lr_factors}", flush=True)
                if n_estimators < n_groups:
                    print(
                        f"[MULTI-LR] Warning: n_estimators ({n_estimators}) < n_groups ({n_groups}). "
                        f"Some groups will be empty.",
                        flush=True
                    )
        else:
            lr_factors = []
            n_groups = 0

        if use_multi_estimator_lr:

            # Assign each estimator to one of the factors as evenly as possible.
            base_group_size = max(1, n_estimators // n_groups)
            remainder = n_estimators % n_groups
            factors_per_estimator = []
            start = 0
            for g in range(n_groups):
                extra = 1 if g < remainder else 0
                end = min(n_estimators, start + base_group_size + extra)
                if start < end:
                    factors_per_estimator.extend([lr_factors[g]] * (end - start))
                start = end
            # In case of any rounding issues, truncate/extend to exact length
            factors_per_estimator = factors_per_estimator[:n_estimators]
            if len(factors_per_estimator) < n_estimators:
                factors_per_estimator.extend(
                    [lr_factors[-1]] * (n_estimators - len(factors_per_estimator))
                )

            factors_tensor = torch.tensor(
                factors_per_estimator, dtype=torch.float32, device=self.device
            )

            # Register gradient hooks to scale grads along estimator dimension.
            # Shapes:
            #   split_values:        [E, I, V]
            #   split_index_array:   [E, I, V]
            #   estimator_weights:   [E, L]
            #   leaf_classes_array:  [E, L] or [E, L, C]

            factors_eiv = factors_tensor.view(-1, 1, 1)
            factors_el = factors_tensor.view(-1, 1)

            def _scale_grad_split_values(grad):
                return grad * factors_eiv

            def _scale_grad_split_index(grad):
                return grad * factors_eiv

            def _scale_grad_estimator_weights(grad):
                return grad * factors_el

            def _scale_grad_leaf_classes(grad):
                # Works for [E, L] or [E, L, C] via broadcasting.
                view_shape = [factors_el.size(0), factors_el.size(1)] + [1] * (
                    grad.dim() - 2
                )
                return grad * factors_el.view(*view_shape)

            self.model.split_values.register_hook(_scale_grad_split_values)
            self.model.split_index_array.register_hook(_scale_grad_split_index)
            self.model.estimator_weights.register_hook(_scale_grad_estimator_weights)
            self.model.leaf_classes_array.register_hook(_scale_grad_leaf_classes)

        # Standard optimizer construction (original behaviour).
        if self.optimizer.lower() == "adamw":
            param_group = [
                {
                    "params": self.model.split_values,
                    "lr": self.learning_rate_values,
                    "betas": (0.9, 0.95),
                },
                {
                    "params": self.model.estimator_weights,
                    "lr": self.learning_rate_weights,
                    "betas": (0.9, 0.95),
                },
            ]

            opt = torch.optim.Adam(param_group)

            param_group_adamW = [
                {
                    "params": self.model.split_index_array,
                    "lr": self.learning_rate_index,
                    "betas": (0.9, 0.95),
                },
                {
                    "params": self.model.leaf_classes_array,
                    "lr": self.learning_rate_leaf,
                    "betas": (0.9, 0.95),
                },
            ]

            opt_adamW = torch.optim.AdamW(param_group_adamW)  # , weight_decay=5e-4

        else:
            param_group = [
                {
                    "params": self.model.split_values,
                    "lr": self.learning_rate_values,
                    "betas": (0.9, 0.95),
                },
                {
                    "params": self.model.split_index_array,
                    "lr": self.learning_rate_index,
                    "betas": (0.9, 0.95),
                },
                {
                    "params": self.model.estimator_weights,
                    "lr": self.learning_rate_weights,
                    "betas": (0.9, 0.95),
                },
                {
                    "params": self.model.leaf_classes_array,
                    "lr": self.learning_rate_leaf,
                    "betas": (0.9, 0.95),
                },
            ]

            if self.optimizer.lower() == "nadam":
                opt = torch.optim.NAdam(param_group)
            elif self.optimizer.lower() == "radam":
                opt = torch.optim.RAdam(param_group)
            else:
                opt = torch.optim.Adam(param_group)

        if self.numeric_embeddings is not None or self.category_embeddings is not None:
            param_group_emb = []
            
            # Collect all embedding parameters
            emb_params_list = []
            if self.numeric_embeddings is not None:
                emb_params_list.extend(list(self.numeric_embeddings.parameters()))
            if self.category_embeddings is not None:
                emb_params_list.extend(list(self.category_embeddings.parameters()))
            
            if use_multi_estimator_lr and len(emb_params_list) > 0:
                # Apply multi-LR scheme to embeddings: split parameters into 5 groups
                # Use the same lr_factors as for estimators
                n_emb_params = len(emb_params_list)
                base_group_size = max(1, n_emb_params // n_groups)
                remainder = n_emb_params % n_groups
                
                start = 0
                for g in range(n_groups):
                    extra = 1 if g < remainder else 0
                    end = min(n_emb_params, start + base_group_size + extra)
                    if start < end:
                        param_group_emb.append({
                            "params": emb_params_list[start:end],
                            "lr": self.learning_rate_embedding * lr_factors[g],
                            "betas": (0.9, 0.95),
                        })
                    start = end
            else:
                # Original single-LR behaviour for embeddings
                if self.numeric_embeddings is not None:
                    param_group_emb += [{"params": self.numeric_embeddings.parameters(), "lr": self.learning_rate_embedding, "betas": (0.9, 0.95)},]
                if self.category_embeddings is not None:
                    param_group_emb += [{"params": self.category_embeddings.parameters(), "lr": self.learning_rate_embedding, "betas": (0.9, 0.95)},]
            
            if self.optimizer.lower() == "adamw":
                opt_emb = torch.optim.AdamW(param_group_emb) #, weight_decay=5e-4
            elif self.optimizer.lower() == "nadam":
                opt_emb = torch.optim.NAdam(param_group_emb)
            elif self.optimizer.lower() == "radam":
                opt_emb = torch.optim.RAdam(param_group_emb)
            else:
                opt_emb = torch.optim.Adam(param_group_emb)

        if self.reduce_on_plateau_scheduler:
            plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt,
                mode='min',      # 'min' for loss, 'max' for accuracy, etc.
                factor=0.2,      # multiply LR by this factor
                patience=15,     # epochs with no improvement before reducing
                threshold=0.001,  # minimum change to qualify as improvement
            )
            if self.optimizer.lower() == "adamw":
                plateau_adamW = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    opt_adamW,
                    mode='min',      # 'min' for loss, 'max' for accuracy, etc.
                    factor=0.2,      # multiply LR by this factor
                    patience=15,     # epochs with no improvement before reducing
                    threshold=0.001,  # minimum change to qualify as improvement
                )
            if self.numeric_embeddings is not None or self.category_embeddings is not None:
                plateau_emb = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    opt_emb,
                    mode='min',      # 'min' for loss, 'max' for accuracy, etc.
                    factor=0.2,      # multiply LR by this factor
                    patience=15,     # epochs with no improvement before reducing
                    threshold=0.001,  # minimum change to qualify as improvement
                )


        if self.cosine_decay_restarts:
            print("Using CosineAnnealingLR", flush=True)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                opt,
                T_0=10,
                T_mult=2,
                eta_min=1e-5
            )
            if self.optimizer.lower() == "adamw":
                scheduler_adamW = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    opt_adamW,
                    T_0=10,
                    T_mult=2,
                    eta_min=1e-5
                )
            
            if self.numeric_embeddings is not None or self.category_embeddings is not None:
                scheduler_emb = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    opt_emb,
                    T_0=10,
                    T_mult=2,
                    eta_min=1e-5
                )

        best_val_score= float("inf")
        best_state = None #copy.deepcopy(self.model.state_dict())
        best_numeric_embeddings_state = None
        best_category_embeddings_state = None

        use_per_estimator_logits = False
        set_seed(self.random_seed)
        for epoch in range(1, self.epochs + 1):
            t0 = time.time()
            total, n = 0.0, 0
            n_batches = len(train_loader)
            self.model.train()
            if self.numeric_embeddings is not None:
                self.numeric_embeddings.train()
            if self.category_embeddings is not None:
                self.category_embeddings.train()
            for batch_idx, batch_data in enumerate(train_loader):
                if self.attach_preprocessed_data and (self.use_category_embeddings or self.use_numeric_embeddings):
                    xb, yb, xb_preprocessed = batch_data
                    xb_preprocessed = xb_preprocessed.to(self.device)
                else:
                    xb, yb = batch_data
                    xb_preprocessed = None
                xb = xb.to(self.device); yb = yb.to(self.device)
                xb = embed_data(xb, self.num_columns_indices, self.categorical_features_raw_indices, self.encoded_columns_indices, self.numeric_embeddings, self.category_embeddings, preprocessed_data=xb_preprocessed)
                logits, per_estimator_logits = self.model(xb, return_per_estimator_logits=True)
                if (self.objective == 'binary' or self.objective == 'multiclass'):
                    if self.focal_loss:
                        if self.objective == 'binary':
                            alpha = (class_weights[1] / (class_weights[0]+class_weights[1]) if class_weights is not None else 0.5)
                            criterion = FocalLoss(gamma=2, alpha=alpha, task_type='binary', reduction="mean")
                            loss = criterion(logits[:,1], yb)
                        else:
                            alpha = class_weights / class_weights.sum()
                            criterion = FocalLoss(gamma=2, alpha=alpha, task_type='multi-class', reduction="mean", num_classes=self.number_of_classes)
                            loss = criterion(logits, yb)
                    else:
                        loss = F.cross_entropy(
                            logits, yb,
                            reduction="mean",
                            weight=class_weights,
                            label_smoothing=self.label_smoothing,
                        )
                elif self.objective == 'regression':
                    preds = logits
                    #preds = preds * self.std + self.mean
                    true = yb.float()
                    with torch.no_grad():
                        true = (true - self.mean) / self.std
                    loss = F.mse_loss(
                        preds, true,
                        reduction="mean",
                        )
                opt.zero_grad(set_to_none=True)
                if self.optimizer.lower() == "adamw":
                    opt_adamW.zero_grad(set_to_none=True)
                if self.numeric_embeddings is not None or self.category_embeddings is not None:
                    opt_emb.zero_grad(set_to_none=True)

                loss.backward()

                if False and batch_idx == 0 and self.verbose >= 2:
                    if True:
                        print("---- Gradient Summary MODEL ----", flush=True)
                        for name, p in list(self.model.named_parameters()):
                            if p.grad is not None:
                                grad = p.grad.detach()
                                print(f"{name:30s} | "
                                    f"mean: {grad.mean().item():+.4e} | "
                                    f"std: {grad.std().item():.4e} | "
                                    f"min: {grad.min().item():+.4e} | "
                                    f"max: {grad.max().item():+.4e} | "
                                    f"norm: {grad.norm().item():.4e}", flush=True)
                            else:
                                print(f"{name:30s} | no grad", flush=True)   
                    if self.numeric_embeddings is not None:
                        print("\n", flush=True)
                        print("---- Gradient Summary NUMERIC EMBEDDINGS ----", flush=True)
                        for name, p in list(self.numeric_embeddings.named_parameters()):
                            if p.grad is not None:
                                grad = p.grad.detach()
                                print(f"{name:30s} | "
                                    f"mean: {grad.mean().item():+.4e} | "
                                    f"std: {grad.std().item():.4e} | "
                                    f"min: {grad.min().item():+.4e} | "
                                    f"max: {grad.max().item():+.4e} | "
                                    f"norm: {grad.norm().item():.4e}", flush=True)
                            else:
                                print(f"{name:30s} | no grad", flush=True)   
                    if self.category_embeddings is not None:
                        print("\n", flush=True)
                        print("---- Gradient Summary CATEGORY EMBEDDINGS ----", flush=True)
                        for name, p in list(self.category_embeddings.named_parameters()):
                            if p.grad is not None:
                                grad = p.grad.detach()
                                print(f"{name:30s} | "
                                    f"mean: {grad.mean().item():+.4e} | "
                                    f"std: {grad.std().item():.4e} | "
                                    f"min: {grad.min().item():+.4e} | "
                                    f"max: {grad.max().item():+.4e} | "
                                    f"norm: {grad.norm().item():.4e}", flush=True)
                            else:
                                print(f"{name:30s} | no grad", flush=True)   
                opt.step()
                if self.optimizer.lower() == "adamw":
                    opt_adamW.step()

                if self.numeric_embeddings is not None or self.category_embeddings is not None:
                    opt_emb.step()

                total += loss.item() * xb.size(0); n += xb.size(0)
            ###print("train_time", time.time() - t0, flush=True)
            if self.swa and epoch > 10:
                self.swa_model.update_parameters(self.model)
                if self.numeric_embeddings is not None:
                    self.swa_numeric_embeddings.update_parameters(self.numeric_embeddings)
                if self.category_embeddings is not None:
                    self.swa_category_embeddings.update_parameters(self.category_embeddings)
            ###print("train_time with update_parameters", time.time() - t0, flush=True)
            if val_loader is not None:
                # compute val metrics
                if self.swa and epoch > 10:
                    numeric_embeddings_eval = self.swa_numeric_embeddings
                else:
                    numeric_embeddings_eval = self.numeric_embeddings

                if self.swa and epoch > 10:
                    category_embeddings_eval = self.swa_category_embeddings
                else:
                    category_embeddings_eval = self.category_embeddings

                if self.swa and epoch > 10:
                    model_eval = self.swa_model
                else:
                    model_eval = self.model

                if not (self.swa and epoch > 10):
                    model_eval.eval()
                    if self.numeric_embeddings is not None:
                        numeric_embeddings_eval.eval()
                    if self.category_embeddings is not None:
                        category_embeddings_eval.eval()

                if self.es_metric:
                    if self.objective == 'binary' or self.objective == 'multiclass':
                        acc, auc, f1 = evaluate(model=model_eval, loader=val_loader, device=self.device, numeric_embeddings=numeric_embeddings_eval, category_embeddings=category_embeddings_eval, num_features=self.num_columns_indices, categorical_features_raw_indices=self.categorical_features_raw_indices, encoded_columns_indices=self.encoded_columns_indices, attach_preprocessed_data=self.attach_preprocessed_data)
                    elif self.objective == 'regression':
                        mse, mae, r2 = evaluate_regression(model=model_eval, loader=val_loader, device=self.device, std=self.std, mean=self.mean, numeric_embeddings=numeric_embeddings_eval, category_embeddings=category_embeddings_eval, num_features=self.num_columns_indices, categorical_features_raw_indices=self.categorical_features_raw_indices, encoded_columns_indices=self.encoded_columns_indices, attach_preprocessed_data=self.attach_preprocessed_data)

                # compute val loss
                with torch.no_grad():
                    vtotal, vn = 0.0, 0
                    for batch_data in val_loader:
                        if self.attach_preprocessed_data and (self.use_category_embeddings or self.use_numeric_embeddings):
                            xb, yb, xb_preprocessed = batch_data
                            xb_preprocessed = xb_preprocessed.to(self.device)
                        else:
                            xb, yb = batch_data
                            xb_preprocessed = None
                        xb = xb.to(self.device); yb = yb.to(self.device)

                        xb = embed_data(xb, self.num_columns_indices, self.categorical_features_raw_indices, self.encoded_columns_indices, numeric_embeddings_eval, category_embeddings_eval, preprocessed_data=xb_preprocessed)
                        logits = model_eval(xb)
                      
                        if (self.objective == 'binary' or self.objective == 'multiclass'):
                            if self.focal_loss:
                                if self.objective == 'binary':
                                    alpha = (class_weights[1] / (class_weights[0]+class_weights[1]) if class_weights is not None else 0.5)
                                    criterion = FocalLoss(gamma=2, alpha=alpha, task_type='binary', reduction="sum")
                                    vloss = criterion(logits[:,1], yb)
                                else:
                                    alpha = class_weights / class_weights.sum()
                                    criterion = FocalLoss(gamma=2, alpha=alpha, task_type='multi-class', reduction="sum", num_classes=self.number_of_classes)
                                    vloss = criterion(logits, yb)
                                
                            else:
                                vloss = F.cross_entropy(
                                        logits, yb,
                                        reduction="sum",
                                        weight=class_weights,
                                        label_smoothing=self.label_smoothing,
                                    )
                        elif self.objective == 'regression':
                            preds = logits
                            #preds = preds * self.std + self.mean
                            true = yb.float()
                            true = (true - self.mean) / self.std

                            vloss = F.mse_loss(
                                    preds, true,
                                    reduction="sum",
                                )
                        vtotal += float(vloss.item()); vn += xb.size(0)
                    val_loss = vtotal / max(vn, 1)
                ###print("train_time with update_parameters and val loop", time.time() - t0, flush=True)
                #val_score = (val_loss if not self.es_metric else (auc if self.objective in ['binary', 'multiclass'] else r2))
                val_score = (val_loss if not self.es_metric else (-auc if self.objective == 'binary' else val_loss if self.objective == 'multiclass' else mse))
                if val_score < best_val_score:
                    best_val_score = val_score
                    if self.swa and epoch > 10:
                        best_state = copy.deepcopy(self.swa_model.module.state_dict())
                        if self.numeric_embeddings is not None:
                            best_numeric_embeddings_state = copy.deepcopy(self.swa_numeric_embeddings.module.state_dict())
                        if self.category_embeddings is not None:
                            best_category_embeddings_state = copy.deepcopy(self.swa_category_embeddings.module.state_dict())

                    else:
                        best_state = copy.deepcopy(self.model.state_dict())
                        if self.numeric_embeddings is not None:
                            best_numeric_embeddings_state = copy.deepcopy(self.numeric_embeddings.state_dict())
                        if self.category_embeddings is not None:
                            best_category_embeddings_state = copy.deepcopy(self.category_embeddings.state_dict())

                epoch_sec = time.time() - t0
                if self.verbose >= 1:
                    if self.es_metric:
                        if (self.objective == 'binary' or self.objective == 'multiclass'):
                            print(f"Epoch {epoch:03d} | TrainLoss: {total/n:.4f} | ValLoss: {val_loss:.4f} | ValAcc: {acc:.4f} | ValAUC: {auc:.4f} | ValF1: {f1:.4f} | Time: {epoch_sec:.2f}s", flush=True)
                        else:
                            print(f"Epoch {epoch:03d} | TrainLoss: {total/n:.4f} | ValLoss: {val_loss:.4f} | MSE: {mse:.4f} | MAE: {mae:.4f} | R2: {r2:.4f} | Time: {epoch_sec:.2f}s", flush=True)
                    else:
                            print(f"Epoch {epoch:03d} | TrainLoss: {total/n:.4f} | ValLoss: {val_loss:.4f} | Time: {epoch_sec:.2f}s", flush=True)
                # Early stopping on val loss
                if early_stopper is not None and early_stopper.step(val_score):
                    break
                ###print("train_time with update_parameters and val loop and storing state", time.time() - t0, flush=True)
            else:
                epoch_sec = time.time() - t0
                if self.verbose >= 1:
                    print(f"Epoch {epoch:03d} | TrainLoss: {total/n:.4f} | Time: {epoch_sec:.2f}s", flush=True)
            if self.cosine_decay_restarts:
                scheduler.step()
                if self.optimizer.lower() == "adamw":
                    scheduler_adamW.step()
                if self.numeric_embeddings is not None or self.category_embeddings is not None:
                    scheduler_emb.step()

                if val_loader is not None and self.reduce_on_plateau_scheduler:
                    plateau.step(val_score)
                    if self.optimizer.lower() == "adamw":
                        plateau_adamW.step(val_score)
                    if self.numeric_embeddings is not None or self.category_embeddings is not None:
                        plateau_emb.step(val_score)

        # Restore best weights if we tracked validation
        if val_loader is not None and best_state is not None:
            print('Restoring best model from epoch with val score:', best_val_score, flush=True)
            self.model.load_state_dict(best_state)
            del best_state
            if self.numeric_embeddings is not None:
                self.numeric_embeddings.load_state_dict(best_numeric_embeddings_state)
                del best_numeric_embeddings_state
            if self.category_embeddings is not None:
                self.category_embeddings.load_state_dict(best_category_embeddings_state)
                del best_category_embeddings_state
            
        if self.swa and epoch > 10:    
            #self.model = self.swa_model
            del self.swa_model
            del self.swa_numeric_embeddings
            del self.swa_category_embeddings

        return self.model

    @torch.no_grad()
    def predict_proba(self, X, **kwargs):
        self.model.eval()

        X = self.preprocess(X)
        # Get preprocessed data if flag is True
        X_preprocessed = getattr(self, '_stored_preprocessed_data', None) if self.attach_preprocessed_data and (self.use_category_embeddings or self.use_numeric_embeddings) else None
        
        if X_preprocessed is not None:
            ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(X_preprocessed))
        else:
            ds = TensorDataset(torch.from_numpy(X))

        loader = DataLoader(
            ds,
            batch_size=256,
            shuffle=False,
            drop_last=False,
        )

        ps = []
        for batch_data in loader:
            if X_preprocessed is not None:
                xb, xb_preprocessed = batch_data
                xb_preprocessed = xb_preprocessed.to(self.device)
            else:
                xb = batch_data[0]
                xb_preprocessed = None
            xb = xb.to(self.device)

            xb = embed_data(xb, self.num_columns_indices, self.categorical_features_raw_indices, self.encoded_columns_indices, self.numeric_embeddings, self.category_embeddings, preprocessed_data=xb_preprocessed)

            logits = self.model(xb)
            if not self.objective == 'regression':
                probs = F.softmax(logits, dim=-1).detach().cpu().numpy()
            else:
                probs = logits.detach().cpu().numpy() * self.std + self.mean  
            ps.append(probs)
        y_prob = np.concatenate(ps)
        #y_prob = y_prob[:,1] if self.objective == 'binary' else y_prob

        return y_prob
    
    @torch.no_grad()
    def predict(self, X):
        self.model.eval()

        X = self.preprocess(X)
        # Get preprocessed data if flag is True
        X_preprocessed = getattr(self, '_stored_preprocessed_data', None) if self.attach_preprocessed_data and (self.use_category_embeddings or self.use_numeric_embeddings) else None
        
        if X_preprocessed is not None:
            ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(X_preprocessed))
        else:
            ds = TensorDataset(torch.from_numpy(X))

        loader = DataLoader(
            ds,
            batch_size=256,
            shuffle=False,
            drop_last=False,
        )

        ps = []
        for batch_data in loader:
            if X_preprocessed is not None:
                xb, xb_preprocessed = batch_data
                xb_preprocessed = xb_preprocessed.to(self.device)
            else:
                xb = batch_data[0]
                xb_preprocessed = None
            xb = xb.to(self.device)            

            xb = embed_data(xb, self.num_columns_indices, self.categorical_features_raw_indices, self.encoded_columns_indices, self.numeric_embeddings, self.category_embeddings, preprocessed_data=xb_preprocessed)
            logits = self.model(xb)
            if not self.objective == 'regression':
                probs = F.softmax(logits, dim=-1).detach().cpu().numpy()
                probs = np.argmax(probs, axis=-1)
            else:
                probs = logits.detach().cpu().numpy() * self.std + self.mean  
            ps.append(probs)
        y_prob = np.concatenate(ps)
        return y_prob

    def preprocess(
        self,
        X: pd.DataFrame,        
        y: pd.Series = None,
        is_train: bool = False,
        **kwargs,
    ) -> pd.DataFrame:
        """Imputes missing values via the mean and adds indicator columns for numerical features.
        Converts indicator columns to categorical features to avoid them being treated as numerical by RealMLP.
        """
        if y is not None and isinstance(y, pd.Series):
            try:
                y = y.values.codes.astype(np.float32)
            except:
                y = y.values.astype(np.float32)


        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        X = X.astype({col: "category" for col in X.select_dtypes(include="object").columns})

        if is_train and y is not None:
            self.mean = np.mean(y)
            self.std = np.std(y)
        if is_train:               
            self._cat_features = None
            self._feature_generator = LabelEncoderFeatureGenerator(verbosity=0)
            self._feature_generator.fit(X=X)

        if self._feature_generator.features_in:
            X = X.copy()
            X[self._feature_generator.features_in] = self._feature_generator.transform(X=X)
            if self._cat_features is None:
                self._cat_features = self._feature_generator.features_in[:]

        # If embeddings are used AND attach_preprocessed_data is True, compute fully preprocessed data first
        if (self.use_category_embeddings or self.use_numeric_embeddings) and self.attach_preprocessed_data:
            # Save a copy for full preprocessing
            X_for_preprocessing = X.copy()
            
            # Compute fully preprocessed data (as if no embeddings were used)
            # This will be stored and attached later
            if is_train:
                # Initialize preprocessing components if not already done
                if not hasattr(self, '_preprocessed_binary_indices'):
                    self._preprocessed_binary_indices = []
                    self._preprocessed_low_cardinality_indices = []
                    self._preprocessed_high_cardinality_indices = []
                    self._preprocessed_num_columns = []
                
                for column_index, column in enumerate(X_for_preprocessing.columns):
                    col_data = X_for_preprocessing.iloc[:, column_index]
                    n_unique = col_data.nunique(dropna=True)    
                    
                    if self._cat_features is not None and column in self._cat_features or n_unique <= 2:
                        if n_unique <= 2:
                            if column not in self._preprocessed_binary_indices:
                                self._preprocessed_binary_indices.append(column)
                        elif n_unique < 10:
                            if column not in self._preprocessed_low_cardinality_indices:
                                self._preprocessed_low_cardinality_indices.append(column)
                        else:
                            if column not in self._preprocessed_high_cardinality_indices:
                                self._preprocessed_high_cardinality_indices.append(column)
                    else:
                        if column not in self._preprocessed_num_columns:
                            self._preprocessed_num_columns.append(column)
                
                self._preprocessed_cat_columns = [col for col in X_for_preprocessing.columns if col not in self._preprocessed_num_columns]

                if not self.missing_values:
                    if len(self._preprocessed_num_columns) > 0:
                        self._preprocessed_mean_train_num = X_for_preprocessing[self._preprocessed_num_columns].mean(axis=0).iloc[0]
                        X_for_preprocessing[self._preprocessed_num_columns] = X_for_preprocessing[self._preprocessed_num_columns].fillna(self._preprocessed_mean_train_num)
                    if len(self._preprocessed_cat_columns) > 0:
                        self._preprocessed_mode_train_cat = X_for_preprocessing[self._preprocessed_cat_columns].mode(axis=0).iloc[0]
                        X_for_preprocessing[self._preprocessed_cat_columns] = X_for_preprocessing[self._preprocessed_cat_columns].fillna(self._preprocessed_mode_train_cat)

                if self.use_freq_enc: 
                    self._preprocessed_freq_enc = FrequencyEncoder(cols=self._preprocessed_cat_columns, as_rank=True)
                    self._preprocessed_freq_enc.fit(X_for_preprocessing)

            # Apply frequency encoding if it exists
            if self.use_freq_enc and hasattr(self, '_preprocessed_freq_enc'): 
                X_for_preprocessing = self._preprocessed_freq_enc.transform(X_for_preprocessing)

            # Fit encoders on training data, transform on all data
            if is_train:
                self._preprocessed_encoder_ordinal = ce.OrdinalEncoder(cols=self._preprocessed_binary_indices, handle_missing='return_nan')
                self._preprocessed_encoder_ordinal.fit(X_for_preprocessing)

                self._preprocessed_encoder_loo = ce.LeaveOneOutEncoder(cols=self._preprocessed_high_cardinality_indices, handle_missing='return_nan')
                if self.problem_type == 'regression':
                    self._preprocessed_encoder_loo.fit(X_for_preprocessing, (y-self.mean)/self.std)
                else:
                    self._preprocessed_encoder_loo.fit(X_for_preprocessing, y)    

                self._preprocessed_encoder_ohe = ce.OneHotEncoder(cols=self._preprocessed_low_cardinality_indices, handle_missing='return_nan')
                self._preprocessed_encoder_ohe.fit(X_for_preprocessing)

            # Apply missing value imputation for validation/test
            if not self.missing_values:
                if len(self._preprocessed_num_columns) > 0:
                    X_for_preprocessing[self._preprocessed_num_columns] = X_for_preprocessing[self._preprocessed_num_columns].fillna(self._preprocessed_mean_train_num)
                if len(self._preprocessed_cat_columns) > 0:
                    X_for_preprocessing[self._preprocessed_cat_columns] = X_for_preprocessing[self._preprocessed_cat_columns].fillna(self._preprocessed_mode_train_cat)
            
            # Transform using fitted encoders
            X_for_preprocessing = self._preprocessed_encoder_ordinal.transform(X_for_preprocessing)                        
            X_for_preprocessing = self._preprocessed_encoder_loo.transform(X_for_preprocessing)
            X_for_preprocessing = self._preprocessed_encoder_ohe.transform(X_for_preprocessing)

            # Fit normalizer on training data, transform on all data
            if is_train:
                if self.use_robust_scale_smoothing:
                    self._preprocessed_normalizer = RobustScaleSmoothClipTransform()
                    self._preprocessed_normalizer.fit(X_for_preprocessing.values.astype(np.float32))
                else:              
                    quantile_noise = 1e-4
                    quantile_train = np.copy(X_for_preprocessing.values).astype(np.float32)
                    np.random.seed(42)
                    stds = np.nanstd(quantile_train, axis=0, keepdims=True)
                    noise_std = quantile_noise / np.maximum(stds, quantile_noise)
                    quantile_train += noise_std * np.random.randn(*quantile_train.shape)    

                    quantile_train = pd.DataFrame(quantile_train, columns=X_for_preprocessing.columns, index=X_for_preprocessing.index)

                    self._preprocessed_normalizer = sklearn.preprocessing.QuantileTransformer(
                                                                                    n_quantiles=min(quantile_train.shape[0], 1000),
                                                                                    output_distribution='normal',
                                                                                    )

                    self._preprocessed_normalizer.fit(quantile_train.values.astype(np.float32))

            # Transform using fitted normalizer
            X_preprocessed = self._preprocessed_normalizer.transform(X_for_preprocessing.values.astype(np.float32))
            # Store preprocessed data - will be attached during embed_data call
            self._stored_preprocessed_data = X_preprocessed
        else:
            # Clear stored preprocessed data if flag is False
            if hasattr(self, '_stored_preprocessed_data'):
                delattr(self, '_stored_preprocessed_data')

        if not self.use_category_embeddings and not self.use_numeric_embeddings:
            if is_train:               
                self.binary_indices = []
                self.low_cardinality_indices = []
                self.high_cardinality_indices = []
                self.num_columns = []

                n_rows = len(X)

                for column_index, column in enumerate(X.columns):
                    col_data = X.iloc[:, column_index]
                    n_unique = col_data.nunique(dropna=True)    
                    
                    if self._cat_features is not None and column in self._cat_features or n_unique <= 2:
                        if n_unique <= 2:
                            self.binary_indices.append(column)
                        elif n_unique < 10:
                            self.low_cardinality_indices.append(column)
                        else:
                            self.high_cardinality_indices.append(column)
                    else:
                        self.num_columns.append(column)     

                self.cat_columns = [col for col in X.columns if col not in self.num_columns]

                if not self.missing_values:
                    if len(self.num_columns) > 0:
                        self.mean_train_num = X[self.num_columns].mean(axis=0).iloc[0]
                        X[self.num_columns] = X[self.num_columns].fillna(self.mean_train_num)
                    if len(self.cat_columns) > 0:
                        self.mode_train_cat = X[self.cat_columns].mode(axis=0).iloc[0]
                        X[self.cat_columns] = X[self.cat_columns].fillna(self.mode_train_cat)

                if self.use_freq_enc: 
                    # 2) Frequency (rank) encoding (unsupervised)
                    self.freq_enc = FrequencyEncoder(cols=self.cat_columns, as_rank=True)
                    self.freq_enc.fit(X)

            if self.use_freq_enc: 
                X = self.freq_enc.transform(X)

            if is_train:               
                self.encoder_ordinal = ce.OrdinalEncoder(cols=self.binary_indices, handle_missing='return_nan')
                self.encoder_ordinal.fit(X)

                self.encoder_loo = ce.LeaveOneOutEncoder(cols=self.high_cardinality_indices, handle_missing='return_nan')
                if self.problem_type == 'regression':
                    self.encoder_loo.fit(X, (y-self.mean)/self.std)
                else:
                    self.encoder_loo.fit(X, y)    

                self.encoder_ohe = ce.OneHotEncoder(cols=self.low_cardinality_indices, handle_missing='return_nan')
                self.encoder_ohe.fit(X)


            if not self.missing_values:
                if len(self.num_columns) > 0:
                    X[self.num_columns] = X[self.num_columns].fillna(self.mean_train_num)
                if len(self.cat_columns) > 0:
                    X[self.cat_columns] = X[self.cat_columns].fillna(self.mode_train_cat)
            
            X = self.encoder_ordinal.transform(X)                        
            X = self.encoder_loo.transform(X)
            X = self.encoder_ohe.transform(X)

            if is_train:               
                self.num_columns_indices = [X.columns.get_loc(col) for col in self.num_columns]
                self.column_names_dataframe = X.columns.tolist()

                self.encoded_columns = [col for col in X.columns if col not in self.num_columns]
                self.encoded_columns_indices = [X.columns.get_loc(col) for col in self.encoded_columns]

                self.not_encoded_columns = []

            X = X.astype(np.float32)
            if is_train:             

                if self.use_robust_scale_smoothing:
                    self.normalizer = RobustScaleSmoothClipTransform()
                    self.normalizer.fit(X.values.astype(np.float32))
                else:              
                    quantile_noise = 1e-4
                    quantile_train = np.copy(X.values).astype(np.float32)
                    np.random.seed(42)
                    stds = np.nanstd(quantile_train, axis=0, keepdims=True)
                    noise_std = quantile_noise / np.maximum(stds, quantile_noise)
                    quantile_train += noise_std * np.random.randn(*quantile_train.shape)    

                    quantile_train = pd.DataFrame(quantile_train, columns=X.columns, index=X.index)

                    self.normalizer = sklearn.preprocessing.QuantileTransformer(
                                                                                n_quantiles=min(quantile_train.shape[0], 1000),
                                                                                output_distribution='normal',
                                                                                )

                    self.normalizer.fit(quantile_train.values.astype(np.float32))

            X = self.normalizer.transform(X.values.astype(np.float32))         

        elif self.use_category_embeddings:
            if is_train:               
                self.binary_indices = []
                self.low_cardinality_indices = []
                self.high_cardinality_indices = []
                self.columns_to_encode = []
                self.num_columns = []
                self.cat_columns = []
                n_rows = len(X)

                for column_index, column in enumerate(X.columns):
                    col_data = X.iloc[:, column_index]
                    n_unique = col_data.nunique(dropna=True)    
                    
                    if self._cat_features is not None and column in self._cat_features or n_unique <= 2:
                        if n_unique <= 2:
                            self.binary_indices.append(column)
                            self.columns_to_encode.append(column)
                        elif n_unique < self.embedding_threshold:
                            self.low_cardinality_indices.append(column)
                            self.columns_to_encode.append(column)   
                        else:
                            self.high_cardinality_indices.append(column)
                        self.cat_columns.append(column)
                    else:
                        self.num_columns.append(column)     
                
                if not self.missing_values:
                    if len(self.num_columns) > 0:
                        self.mean_train_num = X[self.num_columns].mean(axis=0).iloc[0]
                    if len(self.cat_columns) > 0:
                        self.mode_train_cat = X[self.cat_columns].mode(axis=0).iloc[0]
                

            if not self.missing_values:
                if len(self.num_columns) > 0:
                    X[self.num_columns] = X[self.num_columns].fillna(self.mean_train_num)
                if len(self.cat_columns) > 0:
                    X[self.cat_columns] = X[self.cat_columns].fillna(self.mode_train_cat)

            if is_train:          
                self.not_encoded_columns = [col for col in X.columns if col not in self.columns_to_encode]
                #self.columns_to_encode = [col for col in X.columns if col in self.not_encoded_columns and col not in self.num_columns]
                
                self.encoder_ordinal = ce.OrdinalEncoder(cols=self.binary_indices, handle_missing='return_nan')
                self.encoder_ordinal.fit(X)

                if self.embedding_threshold > 1:

                    if self.use_freq_enc: 
                        # 2) Frequency (rank) encoding (unsupervised)
                        self.freq_enc = FrequencyEncoder(cols=self.low_cardinality_indices, as_rank=True)
                        #TODO: USE ALL CATEGORICAL HERE??
                        self.freq_enc.fit(X)

                    self.encoder_ohe = ce.OneHotEncoder(cols=self.low_cardinality_indices, handle_missing='return_nan')
                    self.encoder_ohe.fit(X)

                #else:
                    #self.columns_to_encode = self.cat_columns[:]

                    #self.encoded_columns = []
                    #self.encoded_columns_indices = []

            X = self.encoder_ordinal.transform(X)
            if self.embedding_threshold > 1:
                if self.use_freq_enc: 
                    X = self.freq_enc.transform(X)
                X = self.encoder_ohe.transform(X)



            if is_train:               
                self.encoder_ordinal_emb = ce.OrdinalEncoder(cols=self.high_cardinality_indices, handle_missing='return_nan')
                self.encoder_ordinal_emb.fit(X)

            X = self.encoder_ordinal_emb.transform(X)

            # replace unknowns (-1) with max+1 per column
            self.max_vals = {}
            for col in self.cat_columns:
                max_val = X[col].max(skipna=True)
                self.max_vals[col] = max_val
                X.loc[X[col] == -1, col] = max_val + 1

            X = X.astype(np.float32)

            if is_train:      
                self.encoded_columns = [col for col in X.columns if col not in self.not_encoded_columns and col not in self.num_columns]
                self.encoded_columns_indices = [X.columns.get_loc(col) for col in self.encoded_columns]

                self.num_columns_indices = [X.columns.get_loc(col) for col in self.num_columns]
                self.column_names_dataframe = X.columns.tolist()

            if len(self.num_columns_indices)+len(self.encoded_columns_indices) > 0:
                if is_train:              
                    if self.use_robust_scale_smoothing:
                        self.normalizer = RobustScaleSmoothClipTransform()
                        self.normalizer.fit(X.iloc[:, self.num_columns_indices+self.encoded_columns_indices].values.astype(np.float32))
                    else:                                    
                        # --- prepare training block for quantile fit (only numeric cols) ---
                        quantile_noise = 1e-4
                        np.random.seed(42)

                        # extract numeric block as float32
                        
                        quantile_train_num = X.iloc[:, self.num_columns_indices+self.encoded_columns_indices].to_numpy(dtype=np.float32, copy=True)

                        # add tiny noise to avoid duplicate values (per column)
                        stds = np.nanstd(quantile_train_num, axis=0, keepdims=True)
                        noise_std = quantile_noise / np.maximum(stds, quantile_noise)
                        quantile_train_num += noise_std * np.random.randn(*quantile_train_num.shape).astype(np.float32)

                        # --- fit on numeric block only ---
                        from sklearn.preprocessing import QuantileTransformer
                        self.normalizer = QuantileTransformer(
                            n_quantiles=min(quantile_train_num.shape[0], 1000),
                            output_distribution='normal',
                        )

                        self.normalizer.fit(quantile_train_num)


                # --- transform train/val numeric blocks ---
                X_num = self.normalizer.transform(X.iloc[:, self.num_columns_indices+self.encoded_columns_indices].to_numpy(dtype=np.float32))

                # --- write back the transformed blocks by position ---
                X = X.copy()
                X.iloc[:, self.num_columns_indices+self.encoded_columns_indices] = X_num

            X = X.values.astype(np.float32)

        elif not self.use_category_embeddings and self.use_numeric_embeddings:
            if is_train:               
                self.binary_indices = []
                self.low_cardinality_indices = []
                self.high_cardinality_indices = []
                self.num_columns = []

                n_rows = len(X)

                for column_index, column in enumerate(X.columns):
                    col_data = X.iloc[:, column_index]
                    n_unique = col_data.nunique(dropna=True)    
                    
                    if self._cat_features is not None and column in self._cat_features:
                        if n_unique <= 2:
                            self.binary_indices.append(column)
                        elif n_unique < self.loo_cardinality:
                            self.low_cardinality_indices.append(column)
                        else:
                            self.high_cardinality_indices.append(column)
                    else:
                        self.num_columns.append(column)     

                self.cat_columns = [col for col in X.columns if col not in self.num_columns]

                if not self.missing_values:
                    if len(self.num_columns) > 0:
                        self.mean_train_num = X[self.num_columns].mean(axis=0).iloc[0]
                        X[self.num_columns] = X[self.num_columns].fillna(self.mean_train_num)
                    if len(self.cat_columns) > 0:
                        self.mode_train_cat = X[self.cat_columns].mode(axis=0).iloc[0]
                        X[self.cat_columns] = X[self.cat_columns].fillna(self.mode_train_cat)

                if self.use_freq_enc: 
                    # 2) Frequency (rank) encoding (unsupervised)
                    self.freq_enc = FrequencyEncoder(cols=self.cat_columns, as_rank=True)
                    self.freq_enc.fit(X)
            
                self.encoder_ordinal = ce.OrdinalEncoder(cols=self.binary_indices, handle_missing='return_nan')
                self.encoder_ordinal.fit(X)

                self.encoder_loo = ce.LeaveOneOutEncoder(cols=self.high_cardinality_indices, handle_missing='return_nan')
                if self.problem_type == 'regression':
                    self.encoder_loo.fit(X, (y-self.mean)/self.std)
                else:
                    self.encoder_loo.fit(X, y)    

                self.encoder_ohe = ce.OneHotEncoder(cols=self.low_cardinality_indices, handle_missing='return_nan')
                self.encoder_ohe.fit(X)

            if not self.missing_values:
                if len(self.num_columns) > 0:
                    X[self.num_columns] = X[self.num_columns].fillna(self.mean_train_num)
                if len(self.cat_columns) > 0:
                    X[self.cat_columns] = X[self.cat_columns].fillna(self.mode_train_cat)
            
            if self.use_freq_enc: 
                X = self.freq_enc.transform(X)
            X = self.encoder_ordinal.transform(X)                        
            X = self.encoder_loo.transform(X)
            X = self.encoder_ohe.transform(X)

            if is_train:               
                self.encoded_columns = [col for col in X.columns if col not in self.num_columns]
                self.encoded_columns_indices = [X.columns.get_loc(col) for col in self.encoded_columns]
                self.num_columns_indices = [X.columns.get_loc(col) for col in self.num_columns]

                self.column_names_dataframe = X.columns.tolist()

                self.not_encoded_columns = []


            X = X.astype(np.float32)
            if is_train:      

                if self.use_robust_scale_smoothing:
                    self.normalizer = RobustScaleSmoothClipTransform()
                    self.normalizer.fit(X.values.astype(np.float32))
                else:                                                  


                    quantile_noise = 1e-4
                    quantile_train = np.copy(X.values).astype(np.float32)
                    np.random.seed(42)
                    stds = np.nanstd(quantile_train, axis=0, keepdims=True)
                    noise_std = quantile_noise / np.maximum(stds, quantile_noise)
                    quantile_train += noise_std * np.random.randn(*quantile_train.shape)    

                    quantile_train = pd.DataFrame(quantile_train, columns=X.columns, index=X.index)

                    self.normalizer = sklearn.preprocessing.QuantileTransformer(
                                                                                n_quantiles=min(quantile_train.shape[0], 1000),
                                                                                output_distribution='normal',
                                                                                )

                    self.normalizer.fit(quantile_train.values.astype(np.float32))

            X = self.normalizer.transform(X.values.astype(np.float32))                
        if y is not None:
            return X, y
        return X

    def convert_to_numpy(self,data):
        """
        Converts input data (Pandas DataFrame, TensorFlow tensor, PyTorch tensor, list, or similar iterable) to a NumPy array.
    
        Args:
        data: Input data to be converted. Can be a Pandas DataFrame, TensorFlow tensor, list, or similar iterable.
    
        Returns:
        numpy_array: A NumPy array representation of the input data.
        """
        # Check if the data is a Pandas DataFrame
        if isinstance(data, (pd.DataFrame, pd.Series)):
            return data.values
        
        # Check if the data is a TensorFlow tensor
        elif isinstance(data, torch.Tensor):
            return data.numpy()
        
        # Check if the data is a list or similar iterable (not including strings)
        elif isinstance(data, (list, tuple, np.ndarray)):
            return np.array(data)
        
        else:
            raise TypeError("The input data type is not supported for conversion.")
        




# IMPLEMENTATION FROM: https://github.com/itakurah/Focal-loss-PyTorch | https://arxiv.org/pdf/1708.02002
class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, reduction='mean', task_type='binary', num_classes=None):
        """
        Unified Focal Loss class for binary, multi-class, and multi-label classification tasks.
        :param gamma: Focusing parameter, controls the strength of the modulating factor (1 - p_t)^gamma
        :param alpha: Balancing factor, can be a scalar or a tensor for class-wise weights. If None, no class balancing is used.
        :param reduction: Specifies the reduction method: 'none' | 'mean' | 'sum'
        :param task_type: Specifies the type of task: 'binary', 'multi-class', or 'multi-label'
        :param num_classes: Number of classes (only required for multi-class classification)
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.task_type = task_type
        self.num_classes = num_classes

        # Handle alpha for class balancing in multi-class tasks
        if task_type == 'multi-class' and alpha is not None and isinstance(alpha, (list, torch.Tensor)):
            assert num_classes is not None, "num_classes must be specified for multi-class classification"
            if isinstance(alpha, list):
                self.alpha = torch.Tensor(alpha)
            else:
                self.alpha = alpha

    def forward(self, inputs, targets):
        """
        Forward pass to compute the Focal Loss based on the specified task type.
        :param inputs: Predictions (logits) from the model.
                       Shape:
                         - binary/multi-label: (batch_size, num_classes)
                         - multi-class: (batch_size, num_classes)
        :param targets: Ground truth labels.
                        Shape:
                         - binary: (batch_size,)
                         - multi-label: (batch_size, num_classes)
                         - multi-class: (batch_size,)
        """
        if self.task_type == 'binary':
            return self.binary_focal_loss(inputs, targets)
        elif self.task_type == 'multi-class':
            return self.multi_class_focal_loss(inputs, targets)
        elif self.task_type == 'multi-label':
            return self.multi_label_focal_loss(inputs, targets)
        else:
            raise ValueError(
                f"Unsupported task_type '{self.task_type}'. Use 'binary', 'multi-class', or 'multi-label'.")

    def binary_focal_loss(self, inputs, targets):
        """ Focal loss for binary classification. """
        probs = torch.sigmoid(inputs)
        targets = targets.float()

        # Compute binary cross entropy
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

        # Compute focal weight
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma

        # Apply alpha if provided
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            bce_loss = alpha_t * bce_loss

        # Apply focal loss weighting
        loss = focal_weight * bce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

    def multi_class_focal_loss(self, inputs, targets):
        """ Focal loss for multi-class classification. """
        if self.alpha is not None:
            alpha = self.alpha.to(inputs.device)

        # Convert logits to probabilities with softmax
        probs = F.softmax(inputs, dim=1)

        # One-hot encode the targets
        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes).float()

        # Compute cross-entropy for each class
        ce_loss = -targets_one_hot * torch.log(probs)

        # Compute focal weight
        p_t = torch.sum(probs * targets_one_hot, dim=1)  # p_t for each sample
        focal_weight = (1 - p_t) ** self.gamma

        # Apply alpha if provided (per-class weighting)
        if self.alpha is not None:
            alpha_t = alpha.gather(0, targets)
            ce_loss = alpha_t.unsqueeze(1) * ce_loss

        # Apply focal loss weight
        loss = focal_weight.unsqueeze(1) * ce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

    def multi_label_focal_loss(self, inputs, targets):
        """ Focal loss for multi-label classification. """
        probs = torch.sigmoid(inputs)

        # Compute binary cross entropy
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

        # Compute focal weight
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma

        # Apply alpha if provided
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            bce_loss = alpha_t * bce_loss

        # Apply focal loss weight
        loss = focal_weight * bce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

class FrequencyEncoder:
    def __init__(self, cols, as_rank=True, handle_missing="return_nan", fill_value=0.0):
        """
        cols : list of categorical columns
        as_rank : if True, use rank-normalized frequency (0..1) instead of raw proportion
        handle_missing : {'value','return_nan'}   # 'indicator' not needed here since this is unsupervised
        fill_value : scalar used when handle_missing='value'
        """
        self.cols = cols
        self.as_rank = as_rank
        self.handle_missing = handle_missing
        self.fill_value = fill_value
        self.maps_ = {}

    def fit(self, X, y=None):
        for c in self.cols:
            # include NaNs in counts for stability, but note: NaN won't match via map (NaN != NaN)
            freq = X[c].value_counts(dropna=False)
            freq = freq / freq.sum()
            if self.as_rank:
                freq = freq.rank(method="average") / freq.shape[0]
            # ensure float mapping
            self.maps_[c] = {k: float(v) for k, v in freq.to_dict().items()}
        return self

    def transform(self, X):
        Xo = X.copy()
        for c in self.cols:
            mapped = Xo[c].map(self.maps_[c])

            # Force to numeric BEFORE fillna so we don't keep categorical dtype
            mapped = pd.to_numeric(mapped, errors="coerce")

            if self.handle_missing == "return_nan":
                # leave NaNs as NaNs
                Xo[c + "_freq"] = mapped.astype(float)
            else:
                # default: replace NaN (unseen/NaN originals) with fill_value, e.g., 0.0
                Xo[c + "_freq"] = mapped.astype(float).fillna(self.fill_value)

        return Xo



def set_seed(seed: int = 42):
    # Python built-in RNG
    random.seed(seed)

    # NumPy RNG
    np.random.seed(seed)

    # PyTorch RNG
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU

    # Make CuDNN deterministic (may impact performance)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = False    

    # Set environment hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)


@torch.no_grad()
def evaluate(model, loader, device, numeric_embeddings=None, category_embeddings=None, num_features=None, categorical_features_raw_indices=None, encoded_columns_indices=None, attach_preprocessed_data=False):
    model.eval()
    ys, ps = [], []
    for batch_data in loader:
        if attach_preprocessed_data and len(batch_data) == 3:
            xb, yb, xb_preprocessed = batch_data
            xb_preprocessed = xb_preprocessed.to(device).clone()
        else:
            xb, yb = batch_data
            xb_preprocessed = None
        xb = xb.to(device).clone()

        xb = embed_data(X=xb, num_features=num_features, categorical_features_raw_indices=categorical_features_raw_indices, encoded_columns_indices=encoded_columns_indices, numeric_embeddings=numeric_embeddings, category_embeddings=category_embeddings, preprocessed_data=xb_preprocessed).clone()

        logits = model(xb)
        probs = F.softmax(logits, dim=-1).detach().cpu().numpy()
        ps.append(probs)
        ys.append(yb.numpy())
    y_true = np.concatenate(ys)
    y_prob = np.concatenate(ps)
    # For accuracy, take argmax of probabilities
    y_pred = np.argmax(y_prob, axis=1)
    acc = accuracy_score(y_true, y_pred)  # fix: was undefined
    # Try multiclass AUC (one-vs-one). If fails, return nan.
    try:
        if y_prob.shape[1] == 2:
            auc = roc_auc_score(y_true, y_prob[:, 1])
        else:
            auc = roc_auc_score(y_true, y_prob, multi_class='ovo')
    except Exception:
        auc = float("nan")
    # NEW: macro F1
    try:
        f1 = f1_score(y_true, y_pred, average='macro')
    except Exception:
        f1 = float("nan")
    return acc, auc, f1

@torch.no_grad()
def evaluate_regression(model, loader, device, std, mean, numeric_embeddings=None, category_embeddings=None, num_features=None, categorical_features_raw_indices=None, encoded_columns_indices=None, attach_preprocessed_data=False):
    model.eval()
    ys, ps = [], []
    for batch_data in loader:
        if attach_preprocessed_data and len(batch_data) == 3:
            xb, yb, xb_preprocessed = batch_data
            xb_preprocessed = xb_preprocessed.to(device).clone()
        else:
            xb, yb = batch_data
            xb_preprocessed = None
        xb = xb.to(device).clone()

        xb = embed_data(X=xb, num_features=num_features, categorical_features_raw_indices=categorical_features_raw_indices, encoded_columns_indices=encoded_columns_indices, numeric_embeddings=numeric_embeddings, category_embeddings=category_embeddings, preprocessed_data=xb_preprocessed).clone()

        yb = yb.to(device)
        preds = model(xb).detach().cpu().numpy()
        preds = preds * std + mean
        ps.append(preds)
        ys.append(yb.cpu().numpy())

    y_true = np.concatenate(ys)
    y_pred = np.concatenate(ps)

    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return mse, mae, r2

class EarlyStopper:
    """
    mode='min' (e.g., loss) or 'max' (e.g., AUC).
    Restores best weights via provided state_dict_getter/setter callables.
    """
    def __init__(self, patience=5, min_delta=0.0, mode='min',
                 get_state=None, set_state=None, verbose=True, name=""):
        assert mode in ("min", "max")
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best = None
        self.count = 0
        self.get_state = get_state
        self.set_state = set_state
        self.best_state = None
        self.verbose = verbose
        self.name = name

    def step(self, value):
        if self.best is None:
            self.best = value
            if self.get_state:
                self.best_state = self.get_state()
            return False  # do not stop

        improved = (value < self.best - self.min_delta) if self.mode == "min" else (value > self.best + self.min_delta)
        if improved:
            self.best = value
            self.count = 0
            if self.get_state:
                self.best_state = self.get_state()
        else:
            self.count += 1
            if self.verbose:
                print(f"[EarlyStop {self.name}] no improve ({self.count}/{self.patience}). best={self.best:.6f}, curr={value:.6f}", flush=True)
            if self.count >= self.patience:
                if self.set_state and self.best_state is not None:
                    self.set_state(self.best_state)
                if self.verbose:
                    print(f"[EarlyStop {self.name}] restoring best weights and stopping.", flush=True)
                return True
        return False

def make_datasets_and_loaders(
    X_train,
    y_train,
    X_val=None,
    y_val=None,
    batch_size=256,
    shuffle_train=True,
    task_type="binary",
    X_train_preprocessed=None,
    X_val_preprocessed=None,
):
    """
    Convert numpy arrays or torch tensors into TensorDatasets + DataLoaders
    for training and validation.

    Args:
        X_train, y_train: training data and labels
        X_val, y_val: optional validation data and labels
        batch_size: loader batch size
        num_workers: DataLoader workers
        pin_memory: defaults to True if CUDA available
        shuffle_train: shuffle training loader (default: True)
        X_train_preprocessed: optional preprocessed training data to include
        X_val_preprocessed: optional preprocessed validation data to include

    Returns:
        train_loader, val_loader (val_loader may be None)
    """

    # --- convert numpy arrays to tensors if needed ---
    def to_tensor(x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        return x

    if isinstance(y_train, pd.Series):
        try:
            y_train = y_train.values.codes.astype(np.float32)
        except:
            y_train = y_train.values.astype(np.float32)

    X_train = to_tensor(X_train)
    y_train = to_tensor(y_train)
    if task_type in ('multiclass', 'binary'):
        y_train = y_train.long()
    else:
        y_train = y_train.float()
    
    # Include preprocessed data if provided
    if X_train_preprocessed is not None:
        X_train_preprocessed = to_tensor(X_train_preprocessed)
        train_ds = TensorDataset(X_train, y_train, X_train_preprocessed)
    else:
        train_ds = TensorDataset(X_train, y_train)
        
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=shuffle_train,
        drop_last=True,
    )

    val_loader = None
    if X_val is not None and y_val is not None:

        if isinstance(y_val, pd.Series):
            try:
                y_val = y_val.values.codes.astype(np.float32)
            except:
                y_val = y_val.values.astype(np.float32)

        X_val = to_tensor(X_val)
        y_val = to_tensor(y_val)
        if task_type in ('multiclass', 'binary'):
            y_val = y_val.long()
        else:
            y_val = y_val.float()

        # Include preprocessed data if provided
        if X_val_preprocessed is not None:
            X_val_preprocessed = to_tensor(X_val_preprocessed)
            val_ds = TensorDataset(X_val, y_val, X_val_preprocessed)
        else:
            val_ds = TensorDataset(X_val, y_val)
            
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
        )

    return train_loader, val_loader


# taken from https://github.com/yandex-research/rtdl-num-embeddings/blob/main/package/rtdl_num_embeddings.py
"""On Embeddings for Numerical Features in Tabular Deep Learning."""

def _check_input_shape(x: Tensor, expected_n_features: int) -> None:
    if x.ndim < 1:
        raise ValueError(
            f'The input must have at least one dimension, however: {x.ndim=}'
        )
    if x.shape[-1] != expected_n_features:
        raise ValueError(
            'The last dimension of the input was expected to be'
            f' {expected_n_features}, however, {x.shape[-1]=}'
        )


class LinearEmbeddings(nn.Module):
    """Linear embeddings for continuous features.

    **Shape**

    - Input: `(*, n_features)`
    - Output: `(*, n_features, d_embedding)`

    **Examples**

    >>> batch_size = 2
    >>> n_cont_features = 3
    >>> x = torch.randn(batch_size, n_cont_features)
    >>> d_embedding = 4
    >>> m = LinearEmbeddings(n_cont_features, d_embedding)
    >>> m.get_output_shape()
    torch.Size([3, 4])
    >>> m(x).shape
    torch.Size([2, 3, 4])
    """

    def __init__(self, n_features: int, d_embedding: int) -> None:
        """
        Args:
            n_features: the number of continuous features.
            d_embedding: the embedding size.
        """
        if n_features <= 0:
            raise ValueError(f'n_features must be positive, however: {n_features=}')
        if d_embedding <= 0:
            raise ValueError(f'd_embedding must be positive, however: {d_embedding=}')

        super().__init__()
        self.weight = Parameter(torch.empty(n_features, d_embedding))
        self.bias = Parameter(torch.empty(n_features, d_embedding))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        d_rqsrt = self.weight.shape[1] ** -0.5
        nn.init.uniform_(self.weight, -d_rqsrt, d_rqsrt)
        nn.init.uniform_(self.bias, -d_rqsrt, d_rqsrt)

    def get_output_shape(self) -> torch.Size:
        """Get the output shape without the batch dimensions."""
        return self.weight.shape

    def forward(self, x: Tensor) -> Tensor:
        """Do the forward pass."""
        _check_input_shape(x, self.weight.shape[0])
        return torch.addcmul(self.bias, self.weight, x[..., None])


class LinearReLUEmbeddings(nn.Module):
    """Simple non-linear embeddings for continuous features.

    **Shape**

    - Input: `(*, n_features)`
    - Output: `(*, n_features, d_embedding)`

    **Examples**

    >>> batch_size = 2
    >>> n_cont_features = 3
    >>> x = torch.randn(batch_size, n_cont_features)
    >>>
    >>> d_embedding = 32
    >>> m = LinearReLUEmbeddings(n_cont_features, d_embedding)
    >>> m.get_output_shape()
    torch.Size([3, 32])
    >>> m(x).shape
    torch.Size([2, 3, 32])
    """

    def __init__(self, n_features: int, d_embedding: int = 32) -> None:
        """
        Args:
            n_features: the number of continuous features.
            d_embedding: the embedding size.
        """
        super().__init__()
        self.linear = LinearEmbeddings(n_features, d_embedding)
        self.activation = nn.ReLU()

    def get_output_shape(self) -> torch.Size:
        """Get the output shape without the batch dimensions."""
        return self.linear.weight.shape

    def forward(self, x: Tensor) -> Tensor:
        """Do the forward pass."""
        x = self.linear(x)
        x = self.activation(x)
        return x


class _Periodic(nn.Module):
    """
    NOTE: THIS MODULE SHOULD NOT BE USED DIRECTLY.

    Technically, this is a linear embedding without bias followed by
    the periodic activations. The scale of the initialization
    (defined by the `sigma` argument) plays an important role.
    """

    def __init__(self, n_features: int, k: int, sigma: float) -> None:
        if sigma <= 0.0:
            raise ValueError(f'sigma must be positive, however: {sigma=}')

        super().__init__()
        self._sigma = sigma
        self.weight = Parameter(torch.empty(n_features, k))
        self.reset_parameters()

    def reset_parameters(self):
        """Reset the parameters."""
        # NOTE[DIFF]
        # Here, extreme values (~0.3% probability) are explicitly avoided just in case.
        # In the paper, there was no protection from extreme values.
        bound = self._sigma * 3
        nn.init.trunc_normal_(self.weight, 0.0, self._sigma, a=-bound, b=bound)

    def forward(self, x: Tensor) -> Tensor:
        """Do the forward pass."""
        _check_input_shape(x, self.weight.shape[0])
        x = 2 * math.pi * self.weight * x[..., None]
        x = torch.cat([torch.cos(x), torch.sin(x)], -1)
        return x


# _NLinear is a simplified copy of delu.nn.NLinear:
# https://yura52.github.io/delu/stable/api/generated/delu.nn.NLinear.html
class _NLinear(nn.Module):
    """N *separate* linear layers for N feature embeddings.

    In other words,
    each feature embedding is transformed by its own dedicated linear layer.
    """

    def __init__(
        self, n: int, in_features: int, out_features: int, bias: bool = True
    ) -> None:
        super().__init__()
        self.weight = Parameter(torch.empty(n, in_features, out_features))
        self.bias = Parameter(torch.empty(n, out_features)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        """Reset the parameters."""
        d_in_rsqrt = self.weight.shape[-2] ** -0.5
        nn.init.uniform_(self.weight, -d_in_rsqrt, d_in_rsqrt)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -d_in_rsqrt, d_in_rsqrt)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Do the forward pass."""
        if x.ndim != 3:
            raise ValueError(
                '_NLinear supports only inputs with exactly one batch dimension,'
                ' so `x` must have a shape like (BATCH_SIZE, N_FEATURES, D_EMBEDDING).'
            )
        assert x.shape[-(self.weight.ndim - 1) :] == self.weight.shape[:-1]

        x = x.transpose(0, 1)
        x = x @ self.weight
        x = x.transpose(0, 1)
        if self.bias is not None:
            x = x + self.bias
        return x


class PeriodicEmbeddings(nn.Module):
    """Embeddings for continuous features based on periodic activations.

    See README for details.

    **Shape**

    - Input: `(*, n_features)`
    - Output: `(*, n_features, d_embedding)`

    **Examples**

    >>> batch_size = 2
    >>> n_cont_features = 3
    >>> x = torch.randn(batch_size, n_cont_features)
    >>>
    >>> d_embedding = 24
    >>> m = PeriodicEmbeddings(n_cont_features, d_embedding, lite=False)
    >>> m.get_output_shape()
    torch.Size([3, 24])
    >>> m(x).shape
    torch.Size([2, 3, 24])
    >>>
    >>> m = PeriodicEmbeddings(n_cont_features, d_embedding, lite=True)
    >>> m.get_output_shape()
    torch.Size([3, 24])
    >>> m(x).shape
    torch.Size([2, 3, 24])
    >>>
    >>> # PL embeddings.
    >>> m = PeriodicEmbeddings(n_cont_features, d_embedding=8, activation=False, lite=False)
    >>> m.get_output_shape()
    torch.Size([3, 8])
    >>> m(x).shape
    torch.Size([2, 3, 8])
    """  # noqa: E501

    def __init__(
        self,
        n_features: int,
        d_embedding: int = 24,
        *,
        n_frequencies: int = 48,
        frequency_init_scale: float = 0.01,
        activation: bool = True,
        lite: bool,
    ) -> None:
        """
        Args:
            n_features: the number of features.
            d_embedding: the embedding size.
            n_frequencies: the number of frequencies for each feature.
                (denoted as "k" in Section 3.3 in the paper).
            frequency_init_scale: the initialization scale for the first linear layer
                (denoted as "sigma" in Section 3.3 in the paper).
                **This is an important hyperparameter**, see README for details.
            activation: if `False`, the ReLU activation is not applied.
                Must be `True` if ``lite=True``.
            lite: if True, the outer linear layer is shared between all features.
                See README for details.
        """
        super().__init__()
        self.periodic = _Periodic(n_features, n_frequencies, frequency_init_scale)
        self.linear: Union[nn.Linear, _NLinear]
        if lite:
            # NOTE[DIFF]
            # The lite variation was introduced in a different paper
            # (about the TabR model).
            if not activation:
                raise ValueError('lite=True is allowed only when activation=True')
            self.linear = nn.Linear(2 * n_frequencies, d_embedding)
        else:
            self.linear = _NLinear(n_features, 2 * n_frequencies, d_embedding)
        self.activation = nn.ReLU() if activation else None

    def get_output_shape(self) -> torch.Size:
        """Get the output shape without the batch dimensions."""
        n_features = self.periodic.weight.shape[0]
        d_embedding = (
            self.linear.weight.shape[0]
            if isinstance(self.linear, nn.Linear)
            else self.linear.weight.shape[-1]
        )
        return torch.Size((n_features, d_embedding))

    def forward(self, x: Tensor) -> Tensor:
        """Do the forward pass."""
        x = self.periodic(x)
        x = self.linear(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


def _check_bins(bins: list[Tensor]) -> None:
    if not bins:
        raise ValueError('The list of bins must not be empty')
    for i, feature_bins in enumerate(bins):
        if not isinstance(feature_bins, Tensor):
            raise ValueError(
                'bins must be a list of PyTorch tensors. '
                f'However, for {i=}: {type(bins[i])=}'
            )
        if feature_bins.ndim != 1:
            raise ValueError(
                'Each item of the bin list must have exactly one dimension.'
                f' However, for {i=}: {bins[i].ndim=}'
            )
        if len(feature_bins) < 2:
            raise ValueError(
                'All features must have at least two bin edges.'
                f' However, for {i=}: {len(bins[i])=}'
            )
        if not feature_bins.isfinite().all():
            raise ValueError(
                'Bin edges must not contain nan/inf/-inf.'
                f' However, this is not true for the {i}-th feature'
            )
        if (feature_bins[:-1] >= feature_bins[1:]).any():
            raise ValueError(
                'Bin edges must be sorted.'
                f' However, the for the {i}-th feature, the bin edges are not sorted'
            )
        # Commented out due to spaming warnings.
        # if len(feature_bins) == 2:
        #     warnings.warn(
        #         f'The {i}-th feature has just two bin edges, which means only one bin.'
        #         ' Strictly speaking, using a single bin for the'
        #         ' piecewise-linear encoding should not break anything,'
        #         ' but it is the same as using sklearn.preprocessing.MinMaxScaler'
        #     )


def compute_bins(
    X: torch.Tensor,
    n_bins: int = 48,
    *,
    tree_kwargs: Optional[dict[str, Any]] = None,
    y: Optional[Tensor] = None,
    regression: Optional[bool] = None,
    verbose: bool = False,
) -> list[Tensor]:
    """Compute the bin boundaries for `PiecewiseLinearEncoding` and `PiecewiseLinearEmbeddings`.

    **Usage**

    Compute bins using quantiles (Section 3.2.1 in the paper):

    >>> X_train = torch.randn(10000, 2)
    >>> bins = compute_bins(X_train)

    Compute bins using decision trees (Section 3.2.2 in the paper):

    >>> X_train = torch.randn(10000, 2)
    >>> y_train = torch.randn(len(X_train))
    >>> bins = compute_bins(
    ...     X_train,
    ...     y=y_train,
    ...     regression=True,
    ...     tree_kwargs={'min_samples_leaf': 64, 'min_impurity_decrease': 1e-4},
    ... )

    Args:
        X: the training features.
        n_bins: the number of bins.
        tree_kwargs: keyword arguments for `sklearn.tree.DecisionTreeRegressor`
            (if ``regression=True``) or `sklearn.tree.DecisionTreeClassifier`
            (if ``regression=False``).
            NOTE: requires ``scikit-learn>=1.0,>2`` to be installed.
        y: the training labels (must be provided if ``tree`` is not None).
        regression: whether the labels are regression labels
            (must be provided if ``tree`` is not None).
        verbose: if True and ``tree_kwargs`` is not None, than ``tqdm``
            (must be installed) will report the progress while fitting trees.

    Returns:
        A list of bin edges for all features. For one feature:

        - the maximum possible number of bin edges is ``n_bins + 1``.
        - the minimum possible number of bin edges is ``1``.
    """  # noqa: E501
    if not isinstance(X, Tensor):
        raise ValueError(f'X must be a PyTorch tensor, however: {type(X)=}')
    if X.ndim != 2:
        raise ValueError(f'X must have exactly two dimensions, however: {X.ndim=}')
    if X.shape[0] < 2:
        raise ValueError(f'X must have at least two rows, however: {X.shape[0]=}')
    if X.shape[1] < 1:
        raise ValueError(f'X must have at least one column, however: {X.shape[1]=}')
    if not X.isfinite().all():
        raise ValueError('X must not contain nan/inf/-inf.')
    if (X == X[0]).all(dim=0).any():
        raise ValueError(
            'All columns of X must have at least two distinct values.'
            ' However, X contains columns with just one distinct value.'
        )
    if n_bins <= 1 or n_bins >= len(X):
        raise ValueError(
            'n_bins must be more than 1, but less than len(X), however:'
            f' {n_bins=}, {len(X)=}'
        )

    if tree_kwargs is None:
        if y is not None or regression is not None or verbose:
            raise ValueError(
                'If tree_kwargs is None, then y must be None, regression must be None'
                ' and verbose must be False'
            )

        _upper = 2**24  # 16_777_216
        if len(X) > _upper:
            warnings.warn(
                f'Computing quantile-based bins for more than {_upper} million objects'
                ' may not be possible due to the limitation of PyTorch'
                ' (for details, see https://github.com/pytorch/pytorch/issues/64947;'
                ' if that issue is successfully resolved, this warning may be irrelevant).'  # noqa
                ' As a workaround, subsample the data, i.e. instead of'
                '\ncompute_bins(X, ...)'
                '\ndo'
                '\ncompute_bins(X[torch.randperm(len(X), device=X.device)[:16_777_216]], ...)'  # noqa
                '\nOn CUDA, the computation can still fail with OOM even after'
                ' subsampling. If this is the case, try passing features by groups:'
                '\nbins = sum('
                '\n    compute_bins(X[:, idx], ...)'
                '\n    for idx in torch.arange(len(X), device=X.device).split(group_size),'  # noqa
                '\n    start=[]'
                '\n)'
                '\nAnother option is to perform the computation on CPU:'
                '\ncompute_bins(X.cpu(), ...)'
            )
        del _upper

        # NOTE[DIFF]
        # The code below is more correct than the original implementation,
        # because the original implementation contains an unintentional divergence
        # from what is written in the paper. That divergence affected only the
        # quantile-based embeddings, but not the tree-based embeddings.
        # For historical reference, here is the original, less correct, implementation:
        # https://github.com/yandex-research/tabular-dl-num-embeddings/blob/c1d9eb63c0685b51d7e1bc081cdce6ffdb8886a8/bin/train4.py#L612C30-L612C30
        # (explanation: limiting the number of quantiles by the number of distinct
        #  values is NOT the same as removing identical quantiles after computing them).
        bins = [
            q.unique()
            for q in torch.quantile(
                X, torch.linspace(0.0, 1.0, n_bins + 1).to(X), dim=0
            ).T
        ]
        _check_bins(bins)
        return bins

    else:
        if sklearn_tree is None:
            raise RuntimeError(
                'The scikit-learn package is missing.'
                ' See README.md for installation instructions'
            )
        if y is None or regression is None:
            raise ValueError(
                'If tree_kwargs is not None, then y and regression must not be None'
            )
        if y.ndim != 1:
            raise ValueError(f'y must have exactly one dimension, however: {y.ndim=}')
        if len(y) != len(X):
            raise ValueError(
                f'len(y) must be equal to len(X), however: {len(y)=}, {len(X)=}'
            )
        if y is None or regression is None:
            raise ValueError(
                'If tree_kwargs is not None, then y and regression must not be None'
            )
        if 'max_leaf_nodes' in tree_kwargs:
            raise ValueError(
                'tree_kwargs must not contain the key "max_leaf_nodes"'
                ' (it will be set to n_bins automatically).'
            )

        if verbose:
            if tqdm is None:
                raise ImportError('If verbose is True, tqdm must be installed')
            tqdm_ = tqdm
        else:
            tqdm_ = lambda x: x  # noqa: E731

        if X.device.type != 'cpu' or y.device.type != 'cpu':
            warnings.warn(
                'Computing tree-based bins involves the conversion of the input PyTorch'
                ' tensors to NumPy arrays. The provided PyTorch tensors are not'
                ' located on CPU, so the conversion has some overhead.',
                UserWarning,
            )
        X_numpy = X.cpu().numpy()
        y_numpy = y.cpu().numpy()
        bins = []
        for column in tqdm_(X_numpy.T):
            feature_bin_edges = [float(column.min()), float(column.max())]
            tree = (
                (
                    sklearn_tree.DecisionTreeRegressor
                    if regression
                    else sklearn_tree.DecisionTreeClassifier
                )(max_leaf_nodes=n_bins, **tree_kwargs)
                .fit(column.reshape(-1, 1), y_numpy)
                .tree_
            )
            for node_id in range(tree.node_count):
                # The following condition is True only for split nodes. Source:
                # https://scikit-learn.org/1.0/auto_examples/tree/plot_unveil_tree_structure.html#tree-structure
                if tree.children_left[node_id] != tree.children_right[node_id]:
                    feature_bin_edges.append(float(tree.threshold[node_id]))
            bins.append(torch.as_tensor(feature_bin_edges).unique())
        _check_bins(bins)
        return [x.to(device=X.device, dtype=X.dtype) for x in bins]


class _PiecewiseLinearEncodingImpl(nn.Module):
    """Piecewise-linear encoding.

    NOTE: THIS CLASS SHOULD NOT BE USED DIRECTLY.
    In particular, this class does *not* add any positional information
    to feature encodings. Thus, for Transformer-like models,
    `PiecewiseLinearEmbeddings` is the only valid option.

    Note:
        This is the *encoding* module, not the *embedding* module,
        so it only implements Equation 1 (Figure 1) from the paper,
        and does not have trainable parameters.

    **Shape**

    * Input: ``(*, n_features)``
    * Output: ``(*, n_features, max_n_bins)``,
      where ``max_n_bins`` is the maximum number of bins over all features:
      ``max_n_bins = max(len(b) - 1 for b in bins)``.

    To understand the output structure,
    consider a feature with the number of bins ``n_bins``.
    Formally, its piecewise-linear encoding is a vector of the size ``n_bins``
    that looks as follows::

        x_ple = [1, ..., 1, (x - this_bin_left_edge) / this_bin_width, 0, ..., 0]

    However, this class will instead produce a vector of the size ``max_n_bins``::

        x_ple_actual = [*x_ple[:-1], *zeros(max_n_bins - n_bins), x_ple[-1]]

    In other words:

    * The last encoding component is **always** located in the end,
      even if ``n_bins == 1`` (i.e. even if it is the only component).
    * The leading ``n_bins - 1`` components are located in the beginning.
    * Everything in-between is always set to zeros (like "padding", but in the middle).

    This implementation is *significantly* faster than the original one.
    It relies on two key observations:

    * The piecewise-linear encoding is just
      a non-trainable linear transformation followed by a clamp-based activation.
      Pseudocode: `PiecewiseLinearEncoding(x) = Activation(Linear(x))`.
      The parameters of the linear transformation are defined by the bin edges.
    * Aligning the *last* encoding channel across all features
      allows applying the aforementioned activation simultaneously to all features
      without the loop over features.
    """

    weight: Tensor
    """The weight of the linear transformation mentioned in the class docstring."""

    bias: Tensor
    """The bias of the linear transformation mentioned in the class docstring."""

    single_bin_mask: Optional[Tensor]
    """The indicators of the features with only one bin."""

    mask: Optional[Tensor]
    """The indicators of the "valid" (i.e. "non-padding") part of the encoding."""

    def __init__(self, bins: list[Tensor]) -> None:
        """
        Args:
            bins: the bins computed by `compute_bins`.
        """
        assert len(bins) > 0
        super().__init__()

        n_features = len(bins)
        n_bins = [len(x) - 1 for x in bins]
        max_n_bins = max(n_bins)

        self.register_buffer('weight', torch.zeros(n_features, max_n_bins))
        self.register_buffer('bias', torch.zeros(n_features, max_n_bins))

        single_bin_mask = torch.tensor(n_bins) == 1
        self.register_buffer(
            'single_bin_mask', single_bin_mask if single_bin_mask.any() else None
        )

        self.register_buffer(
            'mask',
            # The mask is needed if features have different number of bins.
            None
            if all(len(x) == len(bins[0]) for x in bins)
            else torch.row_stack(
                [
                    torch.cat(
                        [
                            # The number of bins for this feature, minus 1:
                            torch.ones((len(x) - 1) - 1, dtype=torch.bool),
                            # Unused components (always zeros):
                            torch.zeros(max_n_bins - (len(x) - 1), dtype=torch.bool),
                            # The last bin:
                            torch.ones(1, dtype=torch.bool),
                        ]
                    )
                    # x is a tensor containing the bin bounds for a given feature.
                    for x in bins
                ]
            ),
        )

        for i, bin_edges in enumerate(bins):
            # Formally, the piecewise-linear encoding of one feature looks as follows:
            # `[1, ..., 1, (x - this_bin_left_edge) / this_bin_width, 0, ..., 0]`
            # The linear transformation based on the weight and bias defined below
            # implements the expression in the middle before the clipping to [0, 1].
            # Note that the actual encoding layout produced by this class
            # is slightly different. See the docstring of this class for details.
            bin_width = bin_edges.diff()
            w = 1.0 / bin_width
            b = -bin_edges[:-1] / bin_width
            # The last encoding component:
            self.weight[i, -1] = w[-1]
            self.bias[i, -1] = b[-1]
            # The leading encoding components:
            self.weight[i, : n_bins[i] - 1] = w[:-1]
            self.bias[i, : n_bins[i] - 1] = b[:-1]
            # All in-between components will always be zeros,
            # because the weight and bias are initialized with zeros.

    def get_max_n_bins(self) -> int:
        return self.weight.shape[-1]

    def forward(self, x: Tensor) -> Tensor:
        """Do the forward pass."""
        x = torch.addcmul(self.bias, self.weight, x[..., None])
        if x.shape[-1] > 1:
            x = torch.cat(
                [
                    x[..., :1].clamp_max(1.0),
                    x[..., 1:-1].clamp(0.0, 1.0),
                    (
                        x[..., -1:].clamp_min(0.0)
                        if self.single_bin_mask is None
                        else torch.where(
                            # For features with only one bin,
                            # the whole "piecewise-linear" encoding effectively behaves
                            # like mix-max scaling
                            # (assuming that the edges of the single bin
                            #  are the minimum and maximum feature values).
                            self.single_bin_mask[..., None],
                            x[..., -1:],
                            x[..., -1:].clamp_min(0.0),
                        )
                    ),
                ],
                dim=-1,
            )
        return x


class PiecewiseLinearEncoding(nn.Module):
    """Piecewise-linear encoding.

    See README for detailed explanation.

    **Shape**

    - Input: ``(*, n_features)``
    - Output: ``(*, total_n_bins)``,
      where ``total_n_bins`` is the total number of bins for all features:
      ``total_n_bins = sum(len(b) - 1 for b in bins)``.

    Technically, the output of this module is the flattened output
    of `_PiecewiseLinearEncoding` with all "padding" values removed.
    """

    def __init__(self, bins: list[Tensor]) -> None:
        """
        Args:
            bins: the bins computed by `compute_bins`.
        """
        super().__init__()
        self.impl = _PiecewiseLinearEncodingImpl(bins)

    def get_output_shape(self) -> torch.Size:
        """Get the output shape without the batch dimensions."""
        total_n_bins = (
            self.impl.weight.shape.numel()
            if self.impl.mask is None
            else int(self.impl.mask.long().sum().cpu().item())
        )
        return torch.Size((total_n_bins,))

    def forward(self, x: Tensor) -> Tensor:
        """Do the forward pass."""
        x = self.impl(x)
        return x.flatten(-2) if self.impl.mask is None else x[:, self.impl.mask]


class PiecewiseLinearEmbeddings(nn.Module):
    """Piecewise-linear embeddings.

    **Shape**

    - Input: ``(batch_size, n_features)``
    - Output: ``(batch_size, n_features, d_embedding)``
    """

    def __init__(
        self,
        bins: list[Tensor],
        d_embedding: int,
        *,
        activation: bool,
        version: Literal[None, 'A', 'B'] = None,
    ) -> None:
        """
        Args:
            bins: the bins computed by `compute_bins`.
            d_embedding: the embedding size.
            activation: if True, the ReLU activation is additionally applied in the end.
            version: the preset for various implementation details, such as
                parametrization and initialization. See README for details.
        """
        if d_embedding <= 0:
            raise ValueError(
                f'd_embedding must be a positive integer, however: {d_embedding=}'
            )
        _check_bins(bins)
        if version is None:
            warnings.warn(
                'The `version` argument is not provided, so version="A" will be used'
                ' for backward compatibility.'
                ' See README for recommendations regarding `version`.'
                ' In future, omitting this argument will result in an exception.'
            )
            version = 'A'

        super().__init__()
        n_features = len(bins)
        # NOTE[DIFF]
        # version="B" was introduced in a different paper (about the TabM model).
        is_version_B = version == 'B'

        self.linear0 = (
            LinearEmbeddings(n_features, d_embedding) if is_version_B else None
        )
        self.impl = _PiecewiseLinearEncodingImpl(bins)
        self.linear = _NLinear(
            len(bins),
            self.impl.get_max_n_bins(),
            d_embedding,
            # For the version "B", the bias is already presented in self.linear0.
            bias=not is_version_B,
        )
        if is_version_B:
            # Because of the following line, at initialization,
            # the whole embedding behaves like a linear embedding.
            # The piecewise-linear component is incrementally learnt during training.
            nn.init.zeros_(self.linear.weight)
        self.activation = nn.ReLU() if activation else None

    def get_output_shape(self) -> torch.Size:
        """Get the output shape without the batch dimensions."""
        n_features = self.linear.weight.shape[0]
        d_embedding = self.linear.weight.shape[2]
        return torch.Size((n_features, d_embedding))

    def forward(self, x: Tensor) -> Tensor:
        """Do the forward pass."""
        if x.ndim != 2:
            raise ValueError(
                'For now, only inputs with exactly one batch dimension are supported.'
            )

        x_linear = None if self.linear0 is None else self.linear0(x)

        x_ple = self.impl(x)
        x_ple = self.linear(x_ple)
        if self.activation is not None:
            x_ple = self.activation(x_ple)
        return x_ple if x_linear is None else x_linear + x_ple
    
class RobustScaleSmoothClipTransform(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    def fit(self, X, y=None):
        # don't deal with dataframes for simplicity
        assert isinstance(X, np.ndarray)
        self._median = np.median(X, axis=-2)
        quant_diff = np.quantile(X, 0.75, axis=-2) - np.quantile(X, 0.25, axis=-2)
        max = np.max(X, axis=-2)
        min = np.min(X, axis=-2)
        idxs = quant_diff == 0.0
        # on indexes where the quantile difference is zero, do min-max scaling instead
        quant_diff[idxs] = 0.5 * (max[idxs] - min[idxs])
        factors = 1.0 / (quant_diff + 1e-30)
        # if feature is constant on the training data,
        # set factor to zero so that it is also constant at prediction time
        factors[quant_diff == 0.0] = 0.0
        self._factors = factors
        return self

    def transform(self, X, y=None):
        x_scaled = self._factors[None, :] * (X - self._median[None, :])
        return x_scaled / np.sqrt(1 + (x_scaled / 3) ** 2)