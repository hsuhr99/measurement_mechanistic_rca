# Measurement-Mechanistic RCA

This repository contains a Python package for measurement-mechanistic root cause analysis and the replication package for the NeurIPS paper [Root Cause Analysis of Measurement and Mechanistic Anomalies](https://openreview.net/forum?id=2sxylzR1R2).

The top-level package is the recommended entry point for users. It installs with `pip`, avoids the R dependency by default, and exposes the method for several practical settings:

- contaminated data, with priors learned implicitly by the method;
- contaminated data, with user-provided measurement and generative priors;
- uncontaminated training data, followed by inference on new samples with provided priors.
- causal discovery with the optional bundled NoGAM helper, followed by measurement-mechanistic RCA.

The original experiment code is kept in `replication_neurips/`. The git repository is intended to contain both the lightweight package and the complete replication artifact. A wheel or PyPI-style distribution should contain only the high-level package, the minimal `cali` implementation needed to run the method, and the bundled NoGAM causal-discovery helper. It should not contain the full replication package or external baselines.

## Citation

If you use this code, please cite:

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

## Installation

For the lightweight method package, install from PyPI:

```bash
pip install measurement-mechanistic-rca
```

Install the optional NoGAM causal-discovery dependencies with:

```bash
pip install "measurement-mechanistic-rca[causal-discovery]"
```

For local development from this repository:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

The default backend is the pure-Python spline expectile variant using `pygam`. A scikit-learn spline quantile backend is also available for comparison.

Editable installs from a full repository checkout can also use the bundled examples and replication helpers. Built package distributions are intentionally smaller: they include the public `measurement_mechanistic_rca` API, the minimal `cali` method implementation, and NoGAM via the bundled `dodiscover` code. The complete NeurIPS replication package remains in the git repository.

The R/qgam backend is still available for comparison if R, `qgam`, `mgcv`, and `rpy2` are installed:

```bash
pip install "measurement-mechanistic-rca[qgam]"
```

To run the notebook examples:

```bash
pip install -e ".[causal-discovery,notebooks]"
jupyter notebook examples/method_usage.ipynb
```

## Quick Start

The graph must be a `networkx.DiGraph`, and the data must be an `n x d` NumPy array whose columns match the graph node indices.

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

An executable walkthrough of the main usage patterns is provided in `examples/method_usage.ipynb`.

## Contaminated Data With Provided Priors

If you have prior probabilities for measurement and generative anomalies, pass them directly. Priors can be global vectors of shape `(d,)` or sample-specific arrays of shape `(n, d)`.

```python
measurement_priors = np.array([0.01, 0.10, 0.01])
generative_priors = np.array([0.02, 0.02, 0.05])

model = MeasurementMechanisticRCA(
    prior_mode="single_root",
)
result = model.fit_explain(
    data,
    dag,
    measurement_priors=measurement_priors,
    generative_priors=generative_priors,
)
```

Use `prior_mode="single_root"` for the fast setting that assumes one active root cause per sample. Use `prior_mode="exhaustive"` to enumerate all assignments; this is only practical for small graphs because it scales as `4^d`.

## Clean Training Data Then New Samples

When mechanisms are learned from uncontaminated data, priors must be provided for inference on new samples. The package keeps the training scaler, mechanisms, and residual distributions fixed.

```python
clean_data = np.random.normal(size=(500, 3))
new_samples = np.random.normal(size=(10, 3))

measurement_priors = np.array([0.01, 0.10, 0.01])
generative_priors = np.array([0.02, 0.02, 0.05])

model = MeasurementMechanisticRCA()
model.fit(clean_data, dag)
result = model.explain(
    new_samples,
    measurement_priors=measurement_priors,
    generative_priors=generative_priors,
)
```

The same workflow is included in the notebook:

```bash
jupyter notebook examples/method_usage.ipynb
```

## Causal Discovery Then RCA

The method consumes a `networkx.DiGraph`. If the causal graph is unknown, learn a DAG first and pass the learned graph into the same RCA API:

```python
from measurement_mechanistic_rca import MeasurementMechanisticRCA, discover_dag_nogam

learned_dag = discover_dag_nogam(clean_data, n_crossval=3)

model = MeasurementMechanisticRCA()
result = model.fit_explain(
    data,
    learned_dag,
    measurement_priors=measurement_priors,
    generative_priors=generative_priors,
)
```

`discover_dag_nogam` wraps the bundled Python NoGAM implementation. It has no R dependency and is available from PyPI when installing with the `causal-discovery` extra. The repository also includes `discover_dag_topic` for the bundled TOPIC implementation when running from a full checkout, and `discover_dag_from_order` for small examples where a causal order is known. Minimal wheel/PyPI-style distributions are not expected to bundle TOPIC.

## Regression Backends

Choose the spline regression backend through `regression_backend`:

- `expectile`: default pure-Python spline expectile regression.
- `quantile`: pure-Python scikit-learn spline median-regression replacement.
- `qgam`: R/qgam backend used in all paper experiments.

```python
MeasurementMechanisticRCA()
MeasurementMechanisticRCA(regression_backend="quantile")
MeasurementMechanisticRCA(regression_backend="qgam")
```

## Replication Package

The paper replication package lives in `replication_neurips/`. It contains the experiment runner, baselines, cached result files, plotting code, and Docker-based build instructions. Clone the git repository to run these exact paper experiments; they are intentionally not part of the PyPI package. For the full replication setup, including the recommended Docker build, see `replication_neurips/README.md`.

Representative paper experiments:

```bash
python main.py -e rca
python main.py -e classification
python main.py -e causal_discovery
python main.py -e rca_cc_strong
python main.py -e classification_cc_strong
python main.py -e sachs
```

These cover the standard synthetic root-cause localization experiment, anomaly-type classification, RCA with learned causal graphs, Causal Chamber real-data variants, and the Sachs real-world experiment. The replication package also contains robustness and sensitivity studies such as `rca_dag_misspecification`, `rca_confounding`, `rca_heteroscedasticity`, `rca_soft_interventions`, `rca_sample_sensitivity`, `rca_feature_sensitivity`, and the corresponding classification variants.

The generated result tables are written under `replication_neurips/src/results/`, and plots are written under `replication_neurips/src/plots/`.

## License

Code developed for this package is released under the CC BY 4.0 license; see `LICENSE`. Bundled or adapted external code remains under the licenses of the respective upstream projects; see `THIRD_PARTY_NOTICES.md`, the external repositories, and the notes in `replication_neurips/README.md`.
