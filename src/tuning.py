"""Hyperparameter search helpers: every model is compared default-vs-tuned with cross-validation."""
from __future__ import annotations

import time
import warnings

import numpy as np
from scipy.stats import loguniform, randint, uniform
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
N_JOBS = 2
RS = 42


def candidates(n_classes: int = 2):
    """(name, default estimator, search space) for the 4 model families we compare."""
    obj = "binary:logistic" if n_classes == 2 else "multi:softprob"
    xgb_kw = dict(objective=obj, tree_method="hist", n_jobs=N_JOBS, random_state=RS, verbosity=0)
    return [
        ("Logistic Regression",
         make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
         {"logisticregression__C": loguniform(1e-3, 1e2)}),
        ("Random Forest",
         RandomForestClassifier(n_estimators=200, n_jobs=N_JOBS, random_state=RS),
         {"n_estimators": randint(150, 450), "max_depth": randint(4, 22),
          "min_samples_leaf": randint(1, 40), "max_features": uniform(0.2, 0.7)}),
        ("HistGradientBoosting",
         HistGradientBoostingClassifier(random_state=RS),
         {"learning_rate": loguniform(0.02, 0.3), "max_leaf_nodes": randint(8, 64),
          "min_samples_leaf": randint(10, 120), "l2_regularization": loguniform(1e-4, 10),
          "max_iter": randint(100, 500)}),
        ("XGBoost",
         XGBClassifier(**xgb_kw),
         {"n_estimators": randint(150, 700), "max_depth": randint(2, 9),
          "learning_rate": loguniform(0.01, 0.3), "subsample": uniform(0.6, 0.4),
          "colsample_bytree": uniform(0.5, 0.5), "min_child_weight": randint(1, 12),
          "gamma": uniform(0, 2), "reg_lambda": loguniform(0.1, 20)}),
    ]


def tune_compare(X, y, scoring: str, n_classes: int = 2, n_iter: int = 15, cv: int = 4,
                 only: list[str] | None = None, verbose: bool = True):
    """Returns (rows, best_name, best_estimator). rows hold default vs tuned CV scores."""
    import os
    if os.environ.get("FAST"):
        n_iter, cv = 2, 2
    folds = StratifiedKFold(cv, shuffle=True, random_state=RS)
    rows, best = [], (None, -np.inf, None)
    for name, est, space in candidates(n_classes):
        if only and name not in only:
            continue
        t0 = time.time()
        default = cross_val_score(est, X, y, scoring=scoring, cv=folds, n_jobs=1).mean()
        search = RandomizedSearchCV(est, space, n_iter=n_iter, scoring=scoring, cv=folds,
                                    random_state=RS, n_jobs=1, refit=True)
        search.fit(X, y)
        params = {k.split("__")[-1]: (round(float(v), 5) if isinstance(v, (float, np.floating)) else int(v)
                                      if isinstance(v, (int, np.integer)) else v)
                  for k, v in search.best_params_.items()}
        rows.append({"model": name, "cv_default": float(default), "cv_tuned": float(search.best_score_),
                     "gain": float(search.best_score_ - default), "best_params": params,
                     "n_iter": n_iter, "folds": cv, "seconds": round(time.time() - t0, 1)})
        if verbose:
            print(f"   {name:22s} default={default:.4f} tuned={search.best_score_:.4f} ({time.time()-t0:.0f}s)")
        if search.best_score_ > best[1]:
            best = (name, search.best_score_, search.best_estimator_)
    return rows, best[0], best[2]
