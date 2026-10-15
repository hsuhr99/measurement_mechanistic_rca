# Measurement-Mechanistic RCA

Python package for measurement-mechanistic root cause analysis from the NeurIPS paper [Root Cause Analysis of Measurement and Mechanistic Anomalies](https://openreview.net/forum?id=2sxylzR1R2).

The default installation uses Python spline expectile regression and requires no R.

Repository, executable notebook examples, and the full replication package are available at [github.com/hsuhr99/measurement_mechanistic_rca](https://github.com/hsuhr99/measurement_mechanistic_rca).

## Installation

Install the method package:

```bash
pip install measurement-mechanistic-rca
```

Install the optional NoGAM causal-discovery helper:

```bash
pip install "measurement-mechanistic-rca[causal-discovery]"
```

Install the optional R/qgam comparison backend only if R, `qgam`, `mgcv`, and `rpy2` are available:

```bash
pip install "measurement-mechanistic-rca[qgam]"
```

## Running Example: Same Root Cause, Different Anomaly Types

Consider the causal graph `W <- X -> Y -> Z`. We generate 500 observations
using `Y = 0.8 X + noise`, `Z = 1.2 Y + noise`, and `W = -X + noise`,
with independent Gaussian noise. Columns follow the node indices `[X, Y, Z, W]`.

We inject two anomalies at **Y**:

- **Row 0 — measurement:** add 4 to the recorded Y *after* generating Z.
  The physical process is unchanged, so Z does not respond to the recording error.
- **Row 1 — mechanistic:** add 4 to Y *before* generating Z.
  The shift propagates to Z through its usual mechanism.

Thus both rows violate `X -> Y`, but only the measurement anomaly also breaks
`Y -> Z`. Neither changes `X -> W`.

### 1. Generate the synthetic data

```python
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
```

![The two injected anomalies across the three causal relationships](https://raw.githubusercontent.com/hsuhr99/measurement_mechanistic_rca/main/examples/assets/running_example.png)

Gray points show normal observations, blue the measurement anomaly, and orange
the mechanistic anomaly.

### 2. Localize and classify the two anomalies

Use `fit_explain(data, dag)` to learn the mechanisms and explain the contaminated
data in one step.

```python
np.random.seed(7)
result = MeasurementMechanisticRCA().fit_explain(data, dag)


def show_results(result):
    for row in (0, 1):
        cause = result.top_features(sample_idx=row, k=1)[0]
        print(f"row {row}: root cause {names[cause.feature]}, type {cause.anomaly_type}")


show_results(result)
```

Expected output:

```text
row 0: root cause Y, type measurement
row 1: root cause Y, type mechanistic
```

Both anomalies are correctly localized at Y and classified by type. The unusual
Z value in row 1 is a downstream effect of the mechanism shift at Y.

To inspect the highest-ranked cause for a sample:

```python
cause = result.top_features(sample_idx=0, k=1)[0]
print(names[cause.feature], cause.anomaly_type)  # Y measurement
```

The result also includes the root-cause score and conditional confidence in its type:

```python
print(cause.score)
print(cause.conditional_confidence)
```

Conditional confidence compares the competing type explanations while keeping
other assignments fixed. It is a cost difference, not a probability. Types are
`measurement`, `mechanistic`, `uncertain` when the origin is unresolved, or
`both` when inference with priors selects both types.

To inspect the features assigned to each anomaly type:

```python
measurement_features = result.assignment[0]["measurement"]
mechanistic_features = result.assignment[1]["mechanistic"]

print([names[j] for j in measurement_features])  # ['Y']
print([names[j] for j in mechanistic_features])  # ['Y']
```

### 3. Train on clean data and explain future samples

Use `fit` to learn the normal mechanisms from clean data, then `explain` for
new samples. This requires prior probabilities for measurement and mechanistic
anomalies, since clean data does not reveal how often future interventions occur.
Here we set each probability to 1% per feature.

```python
clean_data = make_data(n=500, seed=8)
new_samples = make_data(n=100, seed=9, contaminated=True)

measurement_priors = np.full(4, 0.01)
mechanistic_priors = np.full(4, 0.01)

np.random.seed(7)
model = MeasurementMechanisticRCA().fit(clean_data, dag)
future_result = model.explain(
    new_samples,
    measurement_priors=measurement_priors,
    mechanistic_priors=mechanistic_priors,
)
show_results(future_result)
```

This identifies Y as the root cause for both new anomalies and returns the same
types as above. The learned mechanisms remain fixed during `explain`.

### Optional: supplied priors for contaminated data

The same priors can also be passed to `fit_explain`:

```python
np.random.seed(7)
result_with_priors = MeasurementMechanisticRCA().fit_explain(
    data,
    dag,
    measurement_priors=measurement_priors,
    mechanistic_priors=mechanistic_priors,
)
show_results(result_with_priors)
```

Priors can have shape `(d,)` for one probability per feature or `(n, d)` for
sample-specific probabilities. With supplied priors, the default `single_root`
mode assumes at most one root-cause feature per sample; `prior_mode="exhaustive"`
allows multiple root causes in small graphs.

Run the full example in the [notebook](https://github.com/hsuhr99/measurement_mechanistic_rca/blob/main/examples/method_usage.ipynb) or with `python examples/running_example.py` from a repository checkout.

## Causal Discovery

If the causal graph is unknown, the optional NoGAM helper can learn a DAG before RCA:

```python
from measurement_mechanistic_rca import MeasurementMechanisticRCA, discover_dag_nogam

learned_dag = discover_dag_nogam(clean_data, n_crossval=3)
result = MeasurementMechanisticRCA().fit_explain(data, learned_dag)
```

NoGAM is bundled through the optional `causal-discovery` extra and has no R dependency.

## Replication Package

For the paper experiments and Docker setup, see the [replication instructions](https://github.com/hsuhr99/measurement_mechanistic_rca/blob/main/replication_neurips/README.md).

## Citation

```bibtex
@inproceedings{suhr:2026:rca,
  title={Root Cause Analysis of Measurement and Mechanistic Anomalies},
  author={Hendrik Suhr and David Kaltenpoth and Jilles Vreeken},
  booktitle={Advances in Neural Information Processing Systems},
  volume={39},
  year={2026},
  url={https://openreview.net/forum?id=2sxylzR1R2}
}
```

## License

Original code developed for this package is released under CC BY 4.0. Bundled or adapted external code remains under the licenses of the respective upstream projects; see the source distribution's third-party notices.
