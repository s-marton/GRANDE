# Changelog

## 1.0.0

### New
- `GRANDEClassifier` / `GRANDERegressor` with a scikit-learn style API:
  - `fit(X, y, X_val=None, y_val=None, cat_features=None, time_limit=None)`, `predict`, `predict_proba`
  - all hyperparameters as constructor arguments (`get_params`, `set_params`, `clone`)
  - pickling, `save` / `load`, and `to(device)`
- `time_limit` for `fit`; `validation_fraction` for the automatic validation split.
- Runs on CPU and GPU (`device=None` picks the GPU if available). `compile_model="auto"` uses `torch.compile` on
  both, on CPU only if a C++ compiler is available. With torch 2.10 and 2.11, inductor's CPU backend crashes on
  GRANDE's dropout; GRANDE therefore compiles for CPU with `fallback_random=True`, which avoids the crash in all
  torch versions and keeps dropout masks identical to eager mode. The CUDA compilation is unchanged.
- Inspection of fitted ensembles (`GRANDE.inspection`):
  - `get_tree` / `export_text` / `plot_tree`: single trees with conditions on the original features, pruned of
    redundant splits (thresholds outside the data range, contradictions, tautologies); leaves show the class
    distribution or value and the leaf weight; optional sample counts and highlighted decision paths.
  - `pruning_statistics`, `apply`.
  - `feature_importances_`, `compute_feature_importances(X)`, `plot_feature_importances`.
  - `explain_instance`, `plot_estimator_weights`, `plot_top_estimators`: instance-wise estimator weights and
    contributions of the trees for single predictions.
  - `explain_features`, `plot_feature_attributions`: Shapley values of the input features for single predictions
    (interventional, with a background sample; exact per reference for up to 12 relevant features, sampled
    otherwise).
  - `extract_rules`: the leaves the ensemble relies on as readable rules, with coverage, weight share, lift and
    precision / mean target.
  - matplotlib is an optional dependency (`GRANDE[plot]`).
- `GRANDE.hpo`: the TabArena default configuration (`TABARENA_DEFAULT_PARAMS`) and random search space
  (`sample_config`, `sample_configs`).

### Changed
- The defaults of the estimators are the TabArena default configuration.
- AutoGluon is no longer a dependency. Categorical columns are label-encoded by GRANDE itself, with
  the categories fixed at fit time.
- torch settings (`matmul_precision="high"`, `n_threads`) apply only during `fit` / `predict` and are restored
  afterwards, instead of being changed globally at import.
- Logging is controlled by `verbose`: no output by default, 1 for per-epoch logs, 2 for early-stopping details.
- When both numeric and categorical embeddings are enabled, both are optimized. The code behind the original
  TabArena submission optimized only the numeric embeddings in that case.
- Packaging moved to `pyproject.toml`. The code is split into modules (`estimator`, `model`, `preprocessing`,
  `training`, `embeddings`, `losses`, `hpo`). The numeric embeddings come from the `rtdl-num-embeddings` package
  instead of a copy of its code.

### Removed
- The dictionary interface `GRANDE(params=...)`. Use `GRANDEClassifier(**params)` / `GRANDERegressor(**params)`:
  the parameter names are the same, `random_seed` is `random_state`, and the problem type follows from the estimator
  and the number of classes.
- The TensorFlow version. It remains available as `GRANDE<=0.1.6` on PyPI.
- The rarely used options `use_multi_estimator_lr` / `estimator_lr_factors` (learning rates per group of trees) and
  `attach_preprocessed_data` (fully preprocessed features in addition to the embeddings).

### Fixed
- When several models were fit in one process (e.g. a hyperparameter search), torch.compile's recompile limit was
  reached after a few models and training silently fell back to eager mode, 15-35x slower. The limit is raised while
  GRANDE fits or predicts.
- Multiclass with `data_subset_fraction < 1` crashed (`index_add_` with a 3-D index).
- `data_subset_fraction < 1` crashed for training sets smaller than `2 * batch_size` or when a batch row was
  selected by no estimator.
- `use_freq_enc=True` together with numeric embeddings crashed (encoders were fit before the frequency columns
  were added).
- `use_freq_enc=True` with numpy input crashed (integer column names).
- Category codes of object/string columns could differ between fit and predict when a batch lacked some categories.
- With category embeddings, categories unseen during training were mapped to `max + 1` of the batch being
  predicted, so a prediction depended on the other rows of the batch. They now map to `max + 1` of the training
  data. The encoding of the training data is unchanged; validation/test rows with unseen categories are affected.
