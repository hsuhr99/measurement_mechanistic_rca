"""Finite intervention supports in robust-scaled coordinates."""

import numpy as np


# RobustScaler uses unit scale when a column's IQR is zero. Use the same
# minimum scale for intervention supports, rather than an epsilon-width
# density that would make constant values overwhelmingly favor intervention.
MIN_INTERVENTION_WIDTH = 1.0


def intervention_bounds(lower, upper):
    """Symmetrically widen supports narrower than one scaled unit.

    Ordinary ranges of at least one unit remain unchanged. Use these bounds
    for both intervention coding costs and sampling latent intervention values.
    """
    lower, upper = np.broadcast_arrays(
        np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
    )
    if not (np.all(np.isfinite(lower)) and np.all(np.isfinite(upper))):
        raise ValueError("Intervention bounds must be finite")
    if np.any(upper < lower):
        raise ValueError("Intervention upper bounds must not be below lower bounds")
    padding = np.maximum(MIN_INTERVENTION_WIDTH - (upper - lower), 0.0) / 2.0
    return lower - padding, upper + padding


def uniform_cost(lower, upper):
    """Negative log2 density of a regularized uniform intervention."""
    lower, upper = intervention_bounds(lower, upper)
    return np.log2(upper - lower)
