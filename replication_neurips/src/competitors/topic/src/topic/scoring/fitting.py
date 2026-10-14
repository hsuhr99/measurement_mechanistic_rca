import warnings

import numpy as np
from pygam import GAM
from sklearn.gaussian_process.kernels import RBF

from ..util.util import data_scale
from ..scoring.models.mdl_fourierf import GaussianProcessFourierRegularized, FourierType
from ..scoring.models.mdl_gp import GaussianProcessRegularized
from enum import Enum


class GPType(Enum):
    EXACT = 'gp'
    FOURIER = 'ff'

    def get_model(self, fourier_type, n_ff, kernel, alpha, n_restarts_optimizer):
        gp_ff = GaussianProcessFourierRegularized(kernel=kernel, alpha=alpha, n_restarts_optimizer=n_restarts_optimizer)
        gp_ff.set_ps(fourier_type, n_ff)
        gp = GaussianProcessRegularized(kernel=kernel, alpha=alpha, n_restarts_optimizer=n_restarts_optimizer)
        return gp if self.value == GPType.EXACT.value else gp_ff


class MIType(Enum):
    TC = 'tc'
    MSS = 'mss'

class ScoreType(Enum):
    # based on MDL
    GAM = 0
    GP = GPType
    MI = MIType

    def get_model(self, **kwargs):
        raise ValueError(f"Only valid for ScoreType.GP, not {self.value}")

    def is_mi(self):
        return self.value in MIType._member_map_


class DataType(Enum):
    CONTINUOUS = 'cont'
    TIMESERIES = 'time'
    TIME_MCONTEXT = 'time_iv'
    CONT_MCONTEXT = 'cont_iv'
    MIXING = 'mixing'

    def __eq__(self, other):
        return self.value == other.value

    def is_time(self):
        return self.value in [self.TIMESERIES.value, self.TIME_MCONTEXT.value]

    def is_multicontext(self):
        return self.value in [self.TIME_MCONTEXT.value, self.CONT_MCONTEXT.value]


def fit_functional_model(
        X, y, M, **scoring_params):
    r""" fitting and scoring functional models

    :param X: parents
    :param y: target
    :param M: n all nodes
    :param scoring_params: hyperparameters

    :Keyword Arguments:
    * *score_type* (``ScoreType``) -- regressor and associated information-theoretic score
    """
    params_score_type = scoring_params.get("score_type", ScoreType.GAM)
    # Hyperparameters for regressions
    params_fourier_type = scoring_params.get("fourier_type", FourierType.QUADRATURE)
    params_gp_scale = scoring_params.get("gp_scale", True)
    params_gam_scale = scoring_params.get("gam_scale", False)
    params_n_ff = scoring_params.get("n_ff", 50)
    params_gp_alpha = scoring_params.get("gp_alpha", 1e-10)
    params_gp_n_restarts = scoring_params.get("gp_n_restarts", 5)
    params_gp_len_scale = scoring_params.get("gp_len_scale", 1.0)
    params_gp_len_scale_bounds = scoring_params.get("gp_len_scale_bounds", (1e-2, 1e2))

    models = [None for _ in range(len(X))]
    scores = [0 for _ in range(len(X))]

    for ci in range(len(X)):
        if params_score_type in GPType._member_map_.values():
            Xtr, ytr = (data_scale(X[ci]), data_scale(y[ci].reshape(-1, 1))) if params_gp_scale else (X[ci], y[ci])
            models[ci], scores[ci] = fit_score_gp(
                Xtr, ytr, params_score_type, params_fourier_type, params_n_ff, params_gp_len_scale,
                params_gp_len_scale_bounds, params_gp_alpha, params_gp_n_restarts)

        elif params_score_type == ScoreType.GAM:

            Xtr, ytr = (data_scale(X[ci]), data_scale(y[ci].reshape(-1, 1))) if params_gam_scale else (X[ci], y[ci])
            models[ci], scores[ci] = fit_score_gam(Xtr, ytr)

        else:
            raise ValueError(f"Invalid score {params_score_type}")
    return sum(scores)


def fit_score_gp(
        Xtr, ytr, score_type: ScoreType, fourier_type: FourierType,
        n_ff, length_scale, length_scale_bounds, alpha, n_restarts_optimizer):
    assert score_type in GPType._member_map_.values()
    kernel = 1 * RBF(length_scale=length_scale, length_scale_bounds=length_scale_bounds)
    gp_model = score_type.get_model(
        fourier_type=fourier_type, n_ff=n_ff, kernel=kernel,
        alpha=alpha, n_restarts_optimizer=n_restarts_optimizer)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gp_model.fit(Xtr, ytr)
    gp_score = gp_model.mdl_score_ytrain()[0]
    return gp_model, gp_score


def fit_score_gam(Xtr, ytr):
    gam = GAM()
    gam.fit(Xtr, ytr)
    n_splines, order = 20, 3
    mse = np.mean((gam.predict(Xtr) - ytr) ** 2)
    n = Xtr.shape[0]
    p = Xtr.shape[1] * n_splines * order
    gam.mdl_lik_train = n * np.log(mse)
    gam.mdl_model_train = 2 * p
    gam.mdl_pen_train = 0
    gam.mdl_train = gam.mdl_lik_train + gam.mdl_model_train + gam.mdl_pen_train
    return gam, gam.mdl_train
