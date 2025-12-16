# 🌳 GRANDE: Gradient-Based Decision Tree Ensembles 🌳

[![PyPI version](https://img.shields.io/pypi/v/GRANDE)](https://pypi.org/project/GRANDE/) [![OpenReview](https://img.shields.io/badge/OpenReview-XEFWBxi075-blue)](https://openreview.net/forum?id=XEFWBxi075) [![arXiv](https://img.shields.io/badge/arXiv-2309.17130-b31b1b.svg)](https://arxiv.org/abs/2309.17130)


<div align="center">

<img src="figures/tabarena_leaderboard.pdf" alt="TabArena Leaderboard" width="60%"/>

<p><strong>TabArena.</strong> The updated PyTorch GRANDE has been evaluated on TabArena and achieved strong results.</p>

</div>

🔍 What's new?
- PyTorch-native implementation for seamless integration; TensorFlow is maintained as a legacy version.
- Strong results on TabArena (specifically for binary classification and regression; multi-class results are less strong dragging down the overall performance which can hopefully be fixed in a future release)
- Method updates for improved performance, including optional categorical and numerical embeddings.
- Training improvements (optimizers, schedulers, early stopping, optional SWA).
- Enhanced preprocessing pipeline with optional frequency encoding and robust normalization.

<div align="center">

<img src="figures/grande.jpg" alt="GRANDE Overview" width="50%"/>

<p><strong>Figure 1: Overview GRANDE.</strong> GRANDE learns hard, axis-aligned trees end-to-end via gradient descent, and uses dynamic instance-wise leaf weighting to combine estimators into a strong ensemble.</p>
</div>


🌳 GRANDE is a gradient-based decision tree ensemble for tabular data.
GRANDE trains ensembles of hard, axis-aligned decision trees end-to-end with gradient descent. Each estimator contributes via instance-wise leaf weights that are learned jointly with split locations and leaf values. This combines the strong inductive bias of trees with the flexibility of neural optimization. The PyTorch version optionally augments inputs with learnable categorical and numerical embeddings, improving representation capacity while preserving interpretability of splits.

📝 More details in the paper: https://openreview.net/forum?id=XEFWBxi075


## Cite us
```text
@inproceedings{
marton2024grande,
title={{GRANDE}: Gradient-Based Decision Tree Ensembles},
author={Sascha Marton and Stefan L{\"u}dtke and Christian Bartelt and Heiner Stuckenschmidt},
booktitle={The Twelfth International Conference on Learning Representations},
year={2024},
url={https://openreview.net/forum?id=XEFWBxi075}
}
```

## Installation
To install the latest release:
```bash
pip install git+https://github.com/s-marton/GRANDE.git
```

## Dependencies
Install core runtime requirements (and optional notebook/example dependencies) via:

```bash
pip install -r requirements.txt
```

Notes:
- The file contains a **core** section (library runtime deps) and a **notebook/example-only** section (OpenML/XGBoost/CatBoost).

## Usage (PyTorch)
Example aligned with the attached notebook (binary classification, OpenML dataset 46915). GPU is recommended.

```python
# Enable GPU (optional)
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Load data
from sklearn.model_selection import train_test_split
import openml
import numpy as np
import sklearn

dataset = openml.datasets.get_dataset(46915, download_data=True, download_qualities=True, download_features_meta_data=True)
X, y, categorical_indicator, attribute_names = dataset.get_data(target=dataset.default_target_attribute)

X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)

# GRANDE (PyTorch)
from GRANDE import GRANDE

params = {
    'depth': 5,
    'n_estimators': 1024,

    'learning_rate_weights': 0.001,
    'learning_rate_index': 0.01,
    'learning_rate_values': 0.05,
    'learning_rate_leaf': 0.05,
    'learning_rate_embedding': 0.02,  # used if embeddings are enabled

    # Embeddings (set True to enable)
    'use_category_embeddings': False,  # True to enable
    'embedding_dim_cat': 8,
    'use_numeric_embeddings': False,   # True to enable
    'embedding_dim_num': 8,
    'embedding_threshold': 1,          # low-cardinality split for categorical embeddings
    'loo_cardinality': 10,             # high-cardinality split for encoders

    'dropout': 0.2,
    'selected_variables': 0.8,
    'data_subset_fraction': 1.0,
    'bootstrap': False,
    'missing_values': False,

    'optimizer': 'adam',               # options: nadam, radam, adamw, adam
    'cosine_decay_restarts': False,
    'reduce_on_plateau_scheduler': True,
    'label_smoothing': 0.0,
    'use_class_weights': False,
    'focal_loss': False,
    'swa': False,
    'es_metric': True,  # AUC for binary, MSE for regression, val_loss for multiclass

    'epochs': 250,
    'batch_size': 256,
    'early_stopping_epochs': 50,

    'use_freq_enc': False,
    'use_robust_scale_smoothing': False,

    # Important: use problem_type, not objective
    'problem_type': 'binary',  # {'binary', 'multiclass', 'regression'}

    'random_seed': 42,
    'verbose': 2,
}

model_grande = GRANDE(params=params)
model_grande.fit(X=X_train, y=y_train, X_val=X_valid, y_val=y_valid)

# Predict
preds_grande = model_grande.predict_proba(X_test)

# Evaluate (binary)
accuracy = sklearn.metrics.accuracy_score(y_test, np.round(preds_grande[:, 1]))
f1 = sklearn.metrics.f1_score(y_test, np.round(preds_grande[:, 1]), average='macro')
roc_auc = sklearn.metrics.roc_auc_score(y_test, preds_grande[:, 1], average='macro')

print('Accuracy GRANDE:', accuracy)
print('F1 Score GRANDE:', f1)
print('ROC AUC GRANDE:', roc_auc)
```

Notes:
- Set use_category_embeddings/use_numeric_embeddings to True to enable embeddings.
- For multiclass, use problem_type='multiclass'. For regression, use 'regression'.
- TensorFlow is supported as a legacy version; the PyTorch path is the recommended/default.

## More
This is an experimental implementation. If you encounter issues, please open an issue or report unexpected behavior.
