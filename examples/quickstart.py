"""GRANDE quickstart: classification and regression with the scikit-learn style API.

Runs on CPU or GPU (picked automatically). Uses small scikit-learn datasets, so no download is needed.
Takes well under a minute on a GPU and a few minutes on a CPU (the default configuration uses 1024 trees).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.model_selection import train_test_split

from GRANDE import GRANDEClassifier, GRANDERegressor


def classification() -> None:
    X, y = load_breast_cancer(return_X_y=True, as_frame=True)
    # add a categorical feature to show that mixed-type DataFrames can be passed directly
    X["size_group"] = pd.cut(X["mean radius"], bins=4, labels=["s", "m", "l", "xl"])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=0, stratify=y_train)

    model = GRANDEClassifier(verbose=0)
    model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
    proba = model.predict_proba(X_test)
    print(f"[classification] device={model.device_}  epochs={model.n_epochs_}  "
          f"ROC AUC={roc_auc_score(y_test, proba[:, 1]):.4f}")


def regression() -> None:
    X, y = load_diabetes(return_X_y=True, as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # without X_val, 20% of the training data is held out for early stopping
    model = GRANDERegressor(depth=5, epochs=300, early_stopping_epochs=30)
    model.fit(X_train, y_train)
    rmse = np.sqrt(mean_squared_error(y_test, model.predict(X_test)))
    print(f"[regression] device={model.device_}  epochs={model.n_epochs_}  RMSE={rmse:.2f}")


if __name__ == "__main__":
    classification()
    regression()
