# Measurement-Mechanistic RCA

Python package for measurement-mechanistic root cause analysis from the NeurIPS paper [Root Cause Analysis of Measurement and Mechanistic Anomalies](https://openreview.net/forum?id=2sxylzR1R2).

This package exposes the method for practical use without requiring the full paper replication package. By default it uses a pure-Python spline expectile backend and has no R dependency.

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

## Quick Start

The graph must be a `networkx.DiGraph`, and data must be an `n x d` NumPy array whose columns match the graph node indices.

```python
import networkx as nx
import numpy as np

from measurement_mechanistic_rca import MeasurementMechanisticRCA

dag = nx.DiGraph([(0, 1), (1, 2)])
data = np.random.normal(size=(100, 3))

model = MeasurementMechanisticRCA()
result = model.fit_explain(data, dag)

print(result.assignment)
print(result.top_features(sample_idx=0, k=3))
```

`result.feature_scores[i, j]` scores feature `j` as a root cause for sample `i`. `result.classifications[i, j]` is `1` for measurement-origin explanations and `0` for generative-origin explanations.

## Provided Priors

If you have prior probabilities for measurement and generative anomalies, pass them directly. Priors can be global vectors of shape `(d,)` or sample-specific arrays of shape `(n, d)`.

```python
measurement_priors = np.array([0.01, 0.10, 0.01])
generative_priors = np.array([0.02, 0.02, 0.05])

result = model.fit_explain(
    data,
    dag,
    measurement_priors=measurement_priors,
    generative_priors=generative_priors,
)
```

## Clean Training Data

When mechanisms are learned from uncontaminated data, priors must be provided for inference on new samples.

```python
model = MeasurementMechanisticRCA()
model.fit(clean_data, dag)
result = model.explain(
    new_samples,
    measurement_priors=measurement_priors,
    generative_priors=generative_priors,
)
```

## Causal Discovery

If the causal graph is unknown, the optional NoGAM helper can learn a DAG before RCA:

```python
from measurement_mechanistic_rca import MeasurementMechanisticRCA, discover_dag_nogam

learned_dag = discover_dag_nogam(clean_data, n_crossval=3)
result = MeasurementMechanisticRCA().fit_explain(data, learned_dag)
```

NoGAM is bundled through the optional `causal-discovery` extra and has no R dependency.

## Replication Package

This PyPI package is intentionally lightweight. It includes the high-level RCA API, the minimal `cali` method implementation, and optional NoGAM support. It does not include the full NeurIPS replication package, experiment runners, plots, results, TOPIC, M2OE, or other external baselines.

For exact paper experiments, use the [git repository](https://github.com/hsuhr99/measurement_mechanistic_rca) and the replication instructions there.

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
