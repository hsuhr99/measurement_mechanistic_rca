"""Reproduce the README example and check both injected anomaly types.

Run from a checkout with the package installed:
    python examples/running_example.py
"""

import networkx as nx
import numpy as np

from measurement_mechanistic_rca import MeasurementMechanisticRCA

# Column indices: X=0, Y=1, Z=2, W=3.
names = ["X", "Y", "Z", "W"]
dag = nx.DiGraph([(0, 1), (1, 2), (0, 3)])


def make_data(n, seed, contaminated=False):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    y = 0.8 * x + rng.normal(scale=0.15, size=n)

    if contaminated:
        y[1] += 4.0  # Mechanism shift at Y, BEFORE generating its child Z.

    z = 1.2 * y + rng.normal(scale=0.15, size=n)
    w = -x + rng.normal(scale=0.15, size=n)

    if contaminated:
        y[0] += 4.0  # Measurement error at Y, AFTER generating Z.

    return np.column_stack([x, y, z, w])


data = make_data(n=500, seed=7, contaminated=True)

def main():
    # The method's internal sampler uses NumPy's global random state.
    np.random.seed(7)
    result = MeasurementMechanisticRCA().fit_explain(data, dag)


    def show_and_check(result):
        for row, expected_kind in [(0, "measurement"), (1, "mechanistic")]:
            cause = result.top_features(sample_idx=row, k=1)[0]
            print(f"row {row}: root cause {names[cause.feature]}, type {cause.anomaly_type}")
            assert cause.feature == 1, f"Wrong root cause for row {row}"
            assert cause.anomaly_type == expected_kind, f"Wrong anomaly type for row {row}"


    show_and_check(result)

    # Independent clean training set and a separate future batch.
    clean_data = make_data(n=500, seed=8)
    new_samples = make_data(n=100, seed=9, contaminated=True)

    # Illustrative per-feature probabilities, identical for every feature.
    # These express prior beliefs about future anomalies, not fitted frequencies.
    measurement_priors = np.full(4, 0.01)
    mechanistic_priors = np.full(4, 0.01)

    np.random.seed(7)
    model = MeasurementMechanisticRCA().fit(clean_data, dag)
    future_result = model.explain(
        new_samples,
        measurement_priors=measurement_priors,
        mechanistic_priors=mechanistic_priors,
    )
    show_and_check(future_result)

    np.random.seed(7)
    result_with_priors = MeasurementMechanisticRCA().fit_explain(
        data,
        dag,
        measurement_priors=measurement_priors,
        mechanistic_priors=mechanistic_priors,
    )
    show_and_check(result_with_priors)


if __name__ == "__main__":
    main()
