"""Model factories: estimator + hyper-parameter distribution for the nested random search.

ridge         impute(median) -> scale -> RidgeCV          linear reference
hgb           HistGradientBoostingRegressor (native NaN)   sklearn, no extra dependency
xgb           XGBRegressor (native NaN)                    sibling-project convention
Round 2 (targets diagnosed weaknesses: tree extrapolation, site offsets, nonlinearity):
spline_ridge  cubic splines (linear extrapolation) on key chemistry, then ridge
xgb_mono      XGBoost with monotone (+) constraints on organic carbon and network target features
xgb_spec      XGBoost on specific formation potential, log(FP) - log(TOC), back-transformed
blend         equal-weight average of ridge and XGBoost
mixed         random-intercept model: fixed chemistry effects + empirical-Bayes site intercept (unseen -> 0)
chained       ridge prediction of the *other* DBP target from chemistry, fed to XGBoost as an extra input

Tier >= 2 matrices carry a `Site` column that `SiteTargetEncoder` turns into `site_te` inside the
pipeline, so the encoding is refit on every (inner and outer) training fold.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, VotingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler
from xgboost import XGBRegressor

from src.features import SITE_COL, SiteTargetEncoder, feature_columns

MODELS = ("ridge", "hgb", "xgb", "spline_ridge", "xgb_mono", "xgb_spec", "blend", "mixed", "chained")
ROUND2_MODELS = ("spline_ridge", "xgb_mono", "xgb_spec", "blend", "mixed", "chained")
OTHER_Y_COL = "_y_other"          # reserved column: other target's truth, used only when fitting `chained`
SPLINE_COLS = ["log_TOC", "log_HS", "log_BB", "log_Biopolymer", "Temp", "log_Turbidity"]
MONOTONE_POS = ["log_TOC", "log_HS", "log_BB", "lag1_y", "hist_mean_y", "up_y", "down_y", "basin_y",
                "grp_y", "plant_raw_y", "site_te", "basin_log_TOC", "basin_log_HS", "up_log_TOC", "down_log_TOC"]
ALPHAS = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]
XGB_SPACE = {
    "n_estimators": randint(150, 600),
    "learning_rate": loguniform(0.02, 0.2),
    "max_depth": randint(2, 6),
    "subsample": uniform(0.6, 0.4),
    "colsample_bytree": uniform(0.6, 0.4),
    "min_child_weight": randint(1, 10),
    "reg_lambda": loguniform(0.1, 10),
}


def _ridge_pipe() -> Pipeline:
    return Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
                     ("scale", StandardScaler()), ("ridge", RidgeCV(alphas=ALPHAS))])


def _xgb(seed: int, **kw) -> XGBRegressor:
    return XGBRegressor(random_state=seed, n_jobs=1, tree_method="hist", verbosity=0, **kw)


def _space(prefix: str) -> dict:
    return {f"{prefix}{k}": v for k, v in XGB_SPACE.items()}


class OffsetRegressor(BaseEstimator, RegressorMixin):
    """Fit `estimator` on y - X[offset_col] and add the offset back (specific formation potential).

    Missing offsets are filled with the training median."""

    def __init__(self, estimator=None, offset_col: str = "log_TOC"):
        self.estimator = estimator
        self.offset_col = offset_col

    def _offset(self, X):
        return X[self.offset_col].fillna(self.fill_).to_numpy()

    def fit(self, X, y):
        self.fill_ = float(X[self.offset_col].median())
        self.estimator_ = clone(self.estimator).fit(X, np.asarray(y) - self._offset(X))
        return self

    def predict(self, X):
        return self.estimator_.predict(X) + self._offset(X)


class MixedLMRegressor(BaseEstimator, RegressorMixin):
    """Random-intercept linear model, fitted in two stages (empirical Bayes).

    1. Fixed effects: least squares on the non-site columns (median-imputed with missing flags, standardised,
       reduced to a full-rank subset by pivoted QR).
    2. Random intercept per site: each site's mean residual shrunk towards 0 by n_s / (n_s + s2_e / s2_u),
       with the between-site variance s2_u estimated by the method of moments on the residuals.
    Prediction = fixed part + the site's shrunken intercept (0 for sites unseen in training).
    This is the standard BLUP form of a random-intercept model; it is used instead of iterative REML because
    statsmodels MixedLM did not converge on this collinear design."""

    def __init__(self, site_col: str = SITE_COL, rank_tol: float = 1e-6):
        self.site_col = site_col
        self.rank_tol = rank_tol

    def _exog(self, X):
        Z = self.prep_.transform(X.drop(columns=[self.site_col]))[:, self.keep_]
        return np.column_stack([np.ones(len(Z)), Z])

    def fit(self, X, y):
        from scipy.linalg import qr
        sites = X[self.site_col].astype(str).to_numpy()
        y = np.asarray(y, float)
        self.prep_ = Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
                               ("scale", StandardScaler())])
        Z = self.prep_.fit_transform(X.drop(columns=[self.site_col]))
        Zc = np.column_stack([np.ones(len(Z)), Z])
        _, R, piv = qr(Zc, mode="economic", pivoting=True)
        cols = np.sort(piv[: int((np.abs(np.diag(R)) / np.abs(R[0, 0]) > self.rank_tol).sum())])
        if 0 not in cols:
            cols = np.sort(np.r_[0, cols[:-1]])
        self.keep_ = cols[cols > 0] - 1
        E = self._exog(X)
        self.fe_ = np.linalg.lstsq(E, y, rcond=None)[0]
        resid = pd.Series(y - E @ self.fe_)
        g = resid.groupby(sites)
        n, mean = g.size(), g.mean()
        s2_e = float(g.var(ddof=1).mean())                                  # within-site residual variance
        s2_u = max(float(mean.var(ddof=1)) - s2_e / float(n.mean()), 0.0)   # between-site variance (moments)
        shrink = n / (n + (s2_e / s2_u if s2_u > 0 else np.inf))
        self.re_ = (shrink * mean).to_dict()
        self.s2_u_, self.s2_e_ = s2_u, s2_e
        return self

    def predict(self, X):
        fixed = self._exog(X) @ self.fe_
        return fixed + X[self.site_col].astype(str).map(self.re_).fillna(0.0).to_numpy()


class ChainedRegressor(BaseEstimator, RegressorMixin):
    """Stage 1: ridge predicts the other DBP target from the inputs (trained on OTHER_Y_COL truth).
    Stage 2: `estimator` predicts this target from the inputs + the stage-1 prediction.

    OTHER_Y_COL is only read in `fit`; `predict` ignores it, so no truth leaks at prediction time."""

    def __init__(self, estimator=None):
        self.estimator = estimator

    def _feats(self, X):
        return X.drop(columns=[OTHER_Y_COL], errors="ignore")

    def fit(self, X, y):
        F = self._feats(X)
        other = X[OTHER_Y_COL].to_numpy(float)
        ok = ~np.isnan(other)
        self.stage1_ = _ridge_pipe().fit(F[ok], other[ok])
        F2 = F.assign(pred_other=self.stage1_.predict(F))
        self.estimator_ = clone(self.estimator).fit(F2, y)
        return self

    def predict(self, X):
        F = self._feats(X)
        return self.estimator_.predict(F.assign(pred_other=self.stage1_.predict(F)))


BASELINES = ("median", "site_median", "persistence")
N_SEARCH_ITER = 12


def make_model(name: str, spec: int | str, seed: int = 0) -> tuple[Pipeline, dict]:
    """`spec` is an int tier or an experiment name (see src/experiments.py)."""
    cols = feature_columns(spec)
    steps = []
    if SITE_COL in cols and name != "mixed":            # the mixed model uses Site as its grouping factor
        steps.append(("site_te", SiteTargetEncoder()))
    feat_names = [c for c in cols if c != SITE_COL] + (["site_te"] if SITE_COL in cols else [])
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
        steps.append(("model", _xgb(seed)))
        params = _space("model__")
    elif name == "spline_ridge":
        spl = [c for c in SPLINE_COLS if c in feat_names]
        steps += [("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)
                   .set_output(transform="pandas")),
                  ("spline", ColumnTransformer([("spl", SplineTransformer(n_knots=4, degree=3, extrapolation="linear"), spl)],
                                               remainder="passthrough", verbose_feature_names_out=False)),
                  ("scale", StandardScaler()),
                  ("model", RidgeCV(alphas=ALPHAS))]
        params = {}
    elif name == "xgb_mono":
        mono = {c: 1 for c in MONOTONE_POS if c in feat_names}
        steps.append(("model", _xgb(seed, monotone_constraints=mono)))
        params = _space("model__")
    elif name == "xgb_spec":
        steps.append(("model", OffsetRegressor(_xgb(seed), "log_TOC")))
        params = _space("model__estimator__")
    elif name == "blend":
        steps.append(("model", VotingRegressor([("ridge", _ridge_pipe()), ("xgb", _xgb(seed))])))
        params = _space("model__xgb__")
    elif name == "mixed":
        if SITE_COL not in cols:
            raise ValueError("mixed needs a feature spec that carries the Site column (tier >= 2)")
        steps.append(("model", MixedLMRegressor()))
        params = {}
    elif name == "chained":
        steps.append(("model", ChainedRegressor(_xgb(seed))))
        params = _space("model__estimator__")
    else:
        raise ValueError(f"model must be one of {MODELS}")
    return Pipeline(steps), params
