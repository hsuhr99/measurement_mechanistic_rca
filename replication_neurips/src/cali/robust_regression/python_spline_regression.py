import math

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import QuantileRegressor
from sklearn.preprocessing import SplineTransformer

from .utils import arbitrary_threshold_cost


def _as_2d_y(y):
    return np.asarray(y).reshape(-1)


def _mad_sigma(residuals):
    residuals = np.asarray(residuals).reshape(-1)
    sigma = np.median(np.abs(residuals - np.median(residuals))) * 1.4826
    return max(float(sigma), 1e-6)


def select_k_per_column(X, k_max=30, k_min=5, frac_unique=0.7, power=1 / 3):
    n, d = X.shape
    k_base = int(n**power)

    ks = []
    for j in range(d):
        u = len(np.unique(X[:, j]))
        k_unique_cap = max(1, int(frac_unique * (u - 1)))
        k_j = min(k_base, k_unique_cap, k_max)
        ks.append(max(k_j, k_min))
    return ks


class AdditiveSplineTransformer(BaseEstimator, TransformerMixin):
    """Per-feature spline basis matching the additive GAM structure."""

    def __init__(self, ks=None, degree=3):
        self.ks = ks
        self.degree = degree

    def fit(self, X, y=None):
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array")

        self.n_features_in_ = X.shape[1]
        ks = self.ks if self.ks is not None else select_k_per_column(X)
        self.ks_ = list(ks)
        self.transformers_ = []

        for j, k in enumerate(self.ks_):
            col = X[:, [j]]
            unique_count = len(np.unique(col))
            if unique_count <= 1:
                self.transformers_.append(None)
                continue

            if unique_count <= self.degree + 1:
                self.transformers_.append("linear")
                continue

            n_knots = max(2, int(k) - self.degree + 1)
            n_knots = min(n_knots, unique_count)
            transformer = SplineTransformer(
                n_knots=n_knots,
                degree=self.degree,
                knots="quantile",
                extrapolation="continue",
                include_bias=False,
            )
            transformer.fit(col)
            self.transformers_.append(transformer)

        return self

    def transform(self, X):
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array")
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, got {X.shape[1]}"
            )

        columns = []
        for j, transformer in enumerate(self.transformers_):
            col = X[:, [j]]
            if transformer is None:
                continue
            if transformer == "linear":
                columns.append(col)
            else:
                columns.append(transformer.transform(col))

        if not columns:
            return np.zeros((X.shape[0], 0))
        return np.hstack(columns)


class PythonSplineModelWrapper:
    def __init__(self, model, d, sigma, scale=False, transformer=None):
        self.model = model
        self.d = d
        self.sigma = max(float(sigma), 1e-6)
        self.encode_uncertainty = scale
        self.transformer = transformer

    def _transform(self, X):
        if self.transformer is None:
            return np.asarray(X)
        return self.transformer.transform(X)

    def predict(self, X, se_fit=False):
        preds = np.asarray(self.model.predict(self._transform(X))).reshape(-1)
        if np.any(np.isnan(preds)) or np.any(np.isinf(preds)):
            raise ValueError("Inf or nan in predictions")

        se_vals = np.full(preds.shape, self.sigma) if se_fit else None
        return preds, se_vals

    def get_scaled_residuals(self, X, y):
        if self.encode_uncertainty:
            preds, se_vals = self.predict(X, se_fit=True)
            return (np.asarray(y).reshape(-1) - preds) / np.maximum(se_vals, 1e-6)
        return np.asarray(y).reshape(-1) - self.predict(X)[0]

    def model_complexity(self, decimal_places=10):
        score = math.log2(max(self.d, 1))

        coef = getattr(self.model, "coef_", None)
        if coef is not None:
            for value in np.asarray(coef).reshape(-1):
                score += arbitrary_threshold_cost(float(value), decimal_places)

        intercept = getattr(self.model, "intercept_", None)
        if intercept is not None:
            for value in np.asarray(intercept).reshape(-1):
                score += arbitrary_threshold_cost(float(value), decimal_places)

        if self.transformer is not None:
            for k in getattr(self.transformer, "ks_", []):
                score += math.log2(max(int(k), 1) + 1)

        return score


class PyGAMExpectileWrapper:
    def __init__(self, model, d, sigma, scale=False):
        self.model = model
        self.d = d
        self.sigma = max(float(sigma), 1e-6)
        self.encode_uncertainty = scale

    def predict(self, X, se_fit=False):
        preds = np.asarray(self.model.predict(X)).reshape(-1)
        if np.any(np.isnan(preds)) or np.any(np.isinf(preds)):
            raise ValueError("Inf or nan in predictions")

        se_vals = np.full(preds.shape, self.sigma) if se_fit else None
        return preds, se_vals

    def get_scaled_residuals(self, X, y):
        if self.encode_uncertainty:
            preds, se_vals = self.predict(X, se_fit=True)
            return (np.asarray(y).reshape(-1) - preds) / np.maximum(se_vals, 1e-6)
        return np.asarray(y).reshape(-1) - self.predict(X)[0]

    def model_complexity(self, decimal_places=10):
        score = math.log2(max(self.d, 1))
        for value in np.asarray(self.model.coef_).reshape(-1):
            score += arbitrary_threshold_cost(float(value), decimal_places)
        return score


def fit_sklearn_quantile_spline(
    X,
    y,
    qu=0.5,
    scale=False,
    alpha=1e-6,
    degree=3,
):
    X = np.asarray(X)
    y = _as_2d_y(y)
    transformer = AdditiveSplineTransformer(degree=degree).fit(X)
    X_basis = transformer.transform(X)

    model = QuantileRegressor(
        quantile=qu,
        alpha=alpha,
        fit_intercept=True,
        solver="highs",
    )
    model.fit(X_basis, y)

    residuals = y - model.predict(X_basis)
    return PythonSplineModelWrapper(
        model=model,
        d=X.shape[1],
        sigma=_mad_sigma(residuals),
        scale=scale,
        transformer=transformer,
    )


def fit_pygam_expectile(X, y, expectile=0.5, scale=False, degree=3):
    from pygam import ExpectileGAM, s

    X = np.asarray(X)
    y = _as_2d_y(y)
    ks = select_k_per_column(X)

    terms = None
    for j, k in enumerate(ks):
        term = s(j, n_splines=max(int(k), degree + 1), spline_order=degree)
        terms = term if terms is None else terms + term

    model = ExpectileGAM(terms=terms, expectile=expectile)
    model.fit(X, y)

    residuals = y - model.predict(X)
    return PyGAMExpectileWrapper(
        model=model,
        d=X.shape[1],
        sigma=_mad_sigma(residuals),
        scale=scale,
    )
