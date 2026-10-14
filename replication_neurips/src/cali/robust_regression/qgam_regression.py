import math

import numpy as np

from .python_spline_regression import (
    fit_pygam_expectile,
    fit_sklearn_quantile_spline,
    select_k_per_column,
)
from .utils import arbitrary_threshold_cost


qgam = None
mgcv = None
robjects = None
Formula = None
FloatVector = None


def _load_r_dependencies():
    global qgam, mgcv, robjects, Formula, FloatVector
    if qgam is not None and mgcv is not None:
        return

    import rpy2.robjects as _robjects
    from rpy2.robjects import FloatVector as _FloatVector
    from rpy2.robjects import Formula as _Formula
    from rpy2.robjects.packages import importr

    qgam = importr("qgam")
    mgcv = importr("mgcv")
    robjects = _robjects
    Formula = _Formula
    FloatVector = _FloatVector


class QGAMModelWrapper:
    def __init__(self, qgam_fit, d, sigma, scale):
        self.model = qgam_fit
        self.d = d
        self.sigma = max(float(sigma), 1e-6)
        self.encode_uncertainty = scale

    def predict(self, X, se_fit=False):
        preds, se_vals = predict_qgam(self.model, X, se_fit)
        preds = np.asarray(preds).reshape(-1)

        if np.any(np.isnan(preds)) or np.any(np.isinf(preds)):
            raise ValueError("Inf or nan in predictions")

        if se_vals is not None:
            se_vals = np.asarray(se_vals).reshape(-1)
        return preds, se_vals

    def get_scaled_residuals(self, X, y):
        if self.encode_uncertainty:
            raise ValueError("Uncertainty-scaled residuals are not supported for R qgam")
        return np.asarray(y).reshape(-1) - self.predict(X)[0]

    def model_complexity(self, decimal_places=10):
        score = math.log2(max(self.d, 1))

        params_list = extract_qgam_mdl_params(self.model)
        for params in params_list:
            for coef in params["coefficients"]:
                score += arbitrary_threshold_cost(coef, decimal_places)

        if self.encode_uncertainty:
            _load_r_dependencies()
            sigma_beta = np.array(robjects.r["vcov"](self.model, unconditional=True))
            p = sigma_beta.shape[0]
            for i in range(p):
                for j in range(i + 1):
                    score += arbitrary_threshold_cost(sigma_beta[i, j], decimal_places)

        return score


def fit_qgam(
    X,
    y,
    loss="L1",
    qu=0.5,
    scale=True,
    discrete=False,
    IQR=False,
    fallback_to_mgcv=True,
):
    _load_r_dependencies()

    X_scaled = np.asarray(X)
    y_scaled = np.asarray(y).reshape(-1)

    if np.any(np.isnan(X_scaled)) or np.any(np.isinf(X_scaled)):
        raise ValueError("Inf or NAN in X")
    if np.any(np.isnan(y_scaled)) or np.any(np.isinf(y_scaled)):
        raise ValueError("Inf or NAN in y")

    ks = select_k_per_column(X=X_scaled)
    df_dict = {
        f"x{i}": FloatVector(X_scaled[:, i])
        for i in range(X_scaled.shape[1])
    }
    df_dict["y"] = FloatVector(y_scaled)
    df = robjects.DataFrame(df_dict)

    formula_terms = " + ".join(
        f"s(x{i}, k={ks[i]}, bs='cr')" for i in range(X_scaled.shape[1])
    )
    formula = Formula(f"y ~ {formula_terms}")

    if loss == "L1":
        try:
            fit = qgam.qgam(formula, data=df, qu=qu, discrete=discrete)
        except Exception:
            if not fallback_to_mgcv:
                raise
            fit = mgcv.gam(formula, data=df, discrete=discrete)
    elif loss == "L2":
        fit = mgcv.gam(formula, data=df, discrete=discrete)
    else:
        raise ValueError("Unsupported loss. Expected one of {'L1', 'L2'}")

    preds, _ = predict_qgam(fit, X_scaled, False)
    residuals = y_scaled - np.asarray(preds).reshape(-1)
    sigma = np.median(np.abs(residuals - np.median(residuals))) * 1.4826
    return QGAMModelWrapper(fit, X.shape[1], sigma, scale=scale)


def predict_qgam(fit, X_scaled, se_fit=False):
    _load_r_dependencies()
    X_scaled = np.asarray(X_scaled)

    df_pred = robjects.DataFrame({
        f"x{i}": FloatVector(X_scaled[:, i])
        for i in range(X_scaled.shape[1])
    })
    preds = robjects.r.predict(fit, newdata=df_pred, se=se_fit)

    if not se_fit:
        return np.array(preds), None

    try:
        return np.array(preds.rx2("fit")), np.array(preds.rx2("se.fit"))
    except Exception as exc:
        raise RuntimeError("Expected R prediction output with 'fit' and 'se.fit'") from exc


def extract_qgam_mdl_params(fit):
    _load_r_dependencies()
    from rpy2.rinterface import NULLType

    parameters = []
    for smooth in fit.rx2("smooth"):
        knots = smooth.rx2("knots")
        if isinstance(knots, NULLType):
            knots = smooth.rx2("xp")
            if isinstance(knots, NULLType):
                raise ValueError(
                    "Neither 'knots' nor 'xp' found in smooth term"
                )

        parameters.append({
            "knots": np.array(knots),
            "k": fit.rx2("rank")[0],
            "coefficients": np.array(fit.rx2("coefficients")),
        })
    return parameters


def robustly_regress_qgam(
    X,
    y,
    mode="iterative",
    loss="L1",
    scale=True,
    discrete=False,
    IQR=False,
    backend="r_qgam",
    qu=0.5,
):
    backend_key = (backend or "r_qgam").lower()
    loss_key = (loss or "L1").lower()

    if backend_key in {"sklearn_quantile_spline", "quantile_spline", "python_quantile"}:
        return fit_sklearn_quantile_spline(X, y, qu=qu, scale=scale)

    if backend_key in {"pygam_expectile", "expectile"}:
        return fit_pygam_expectile(X, y, expectile=qu, scale=scale)

    if loss_key in {"sklearn_quantile_spline", "quantile_spline", "python_quantile"}:
        return fit_sklearn_quantile_spline(X, y, qu=qu, scale=scale)

    if loss_key in {"pygam_expectile", "expectile"}:
        return fit_pygam_expectile(X, y, expectile=qu, scale=scale)

    return fit_qgam(
        X,
        y,
        loss=loss,
        qu=qu,
        scale=scale,
        discrete=discrete,
        fallback_to_mgcv=backend_key != "r_qgam_strict",
    )
