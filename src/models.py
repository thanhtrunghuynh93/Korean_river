"""Model factories: estimator + hyper-parameter distribution for the nested random search.

ridge   impute(median) -> scale -> RidgeCV               linear reference
hgb     HistGradientBoostingRegressor (native NaN)        sklearn, no extra dependency
xgb     XGBRegressor (native NaN)                         sibling-project convention

Tier >= 2 matrices carry a `Site` column that `SiteTargetEncoder` turns into `site_te` inside the
pipeline, so the encoding is refit on every (inner and outer) training fold.
"""
from __future__ import annotations

from scipy.stats import loguniform, randint, uniform
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from src.features import SITE_COL, SiteTargetEncoder, feature_columns

MODELS = ("ridge", "hgb", "xgb")
BASELINES = ("median", "site_median", "persistence")
N_SEARCH_ITER = 12


def make_model(name: str, spec: int | str, seed: int = 0) -> tuple[Pipeline, dict]:
    """`spec` is an int tier or an experiment name (see src/experiments.py)."""
    steps = []
    if SITE_COL in feature_columns(spec):
        steps.append(("site_te", SiteTargetEncoder()))
    if name == "ridge":
        # add_indicator: columns that are 100 % missing for a site group (e.g. SUVA in treatment trains)
        # get a missingness flag instead of silently becoming the median
        steps += [("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
                  ("scale", StandardScaler()),
                  ("model", RidgeCV(alphas=[0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]))]
        params: dict = {}
    elif name == "hgb":
        steps.append(("model", HistGradientBoostingRegressor(random_state=seed, early_stopping=False)))
        params = {
            "model__max_iter": randint(100, 600),
            "model__learning_rate": loguniform(0.02, 0.2),
            "model__max_depth": randint(2, 7),
            "model__min_samples_leaf": randint(5, 30),
            "model__l2_regularization": uniform(0.0, 1.0),
        }
    elif name == "xgb":
        steps.append(("model", XGBRegressor(random_state=seed, n_jobs=1, tree_method="hist", verbosity=0)))
        params = {
            "model__n_estimators": randint(150, 600),
            "model__learning_rate": loguniform(0.02, 0.2),
            "model__max_depth": randint(2, 6),
            "model__subsample": uniform(0.6, 0.4),
            "model__colsample_bytree": uniform(0.6, 0.4),
            "model__min_child_weight": randint(1, 10),
            "model__reg_lambda": loguniform(0.1, 10),
        }
    else:
        raise ValueError(f"model must be one of {MODELS}")
    return Pipeline(steps), params
