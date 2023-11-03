# 🌳 GRANDE: Gradient-Based Decision Tree Ensembles 🌳

🌳 GRANDE is a novel gradient-based decision tree ensemble method for tabular data!

🔍 What's new?
- End-to-end gradient descent for tree ensembles.
- Combines inductive bias of hard, axis-aligned splits with the flexibility of a gradient descent optimization.
- Advanced instance-wise weighting to learn representations for both simple & complex relations in one model.

📝 Details on the method can be found in the preprint available under: https://arxiv.org/abs/2309.17130

## Installation
To download the latest official release of the package, use a pip command below:
```bash
pip install GRANDE
```
More details can be found under: https://pypi.org/project/GRANDE/

## Usage
Example usage is in the following or available in ***GRANDE_minimal_example.ipynb***. Please note that a GPU is required to achieve competitive runtimes.

### Load Data
```python
from sklearn.model_selection import train_test_split
import openml

dataset = openml.datasets.get_dataset(40536)
X, y, categorical_indicator, attribute_names = dataset.get_data(target=dataset.default_target_attribute)
categorical_feature_indices = [idx if idx_bool for idx, idx_bool in enumerate(categorical_indicator)]

X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train, X_valid, y_train, y_valid = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)

y_train = y_train.values.codes.astype(np.float64)
y_valid = y_valid.values.codes.astype(np.float64)
y_test = y_test.values.codes.astype(np.float64)
```

### Preprocessing, Hyperparameters and Training 
GRANDE requires categorical features to be encoded appropriately. The best results are achieved using Leave-One-Out Encoding for high-cardinality categorical features and One-Hot Encoding for low-cardinality categorical features. Furthermore, all features should be normalized using a quantile transformation. Passing the categorical indices to the model wil automatically preprocess the data accordingly.

In the following, we will train the model using the default parameters. GRANDE already archives great results with its default parameters, but a HPO can increase the performance even further. An appropriate grid is specified in the model class.

```python
from GRANDE import GRANDE

params = {
        'depth': 5,
        'n_estimators': 2048,

        'learning_rate_weights': 0.005,
        'learning_rate_index': 0.01,
        'learning_rate_values': 0.01,
        'learning_rate_leaf': 0.01,

        'optimizer': 'SWA',
        'cosine_decay_steps': 0,

        'initializer': 'RandomNormal',

        'loss': 'crossentropy',
        'focal_loss': False,
        'temperature': 0.0,

        'from_logits': True,
        'apply_class_balancing': True,

        'dropout': 0.0,

        'selected_variables': 0.8,
        'data_subset_fraction': 1.0,
}

args = {
    'epochs': 1_000,
    'early_stopping_epochs': 25,
    'batch_size': 64,

    'cat_idx': categorical_feature_indices, # put list of categorical indices
    'objective': 'binary',
    
    'metrics': ['F1'], # F1, Accuracy, R2
    'random_seed': 42,
    'verbose': 1,       
}

model_grande = GRANDE(params=params, args=args)

model_grande.fit(X_train=X_train,
          y_train=y_train,
          X_val=X_valid,
          y_val=y_valid)

preds_grande = model_grande.predict(X_test)

```

### Evaluate Model

```python
preds = model_grande.predict(X_test)

accuracy = sklearn.metrics.accuracy_score(y_test, np.round(preds[:,1]))
f1_score = sklearn.metrics.f1_score(y_test, np.round(preds[:,1]), average='macro')
roc_auc = sklearn.metrics.roc_auc_score(y_test, preds[:,1], average='macro')

print('Accuracy:', accuracy)
print('F1 Score:', f1_score)
print('ROC AUC:', roc_auc)
```

## More

Please note that this is an experimental implementation which is not fully tested yet. If you encounter any errors, or you observe unexpected behavior, please let me know.

The code for reproducing the experiments from the paper now is in a separate folder ***./experiments_paper/***