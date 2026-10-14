# Reproducibility Package for Root Cause Analysis of Measurement and Mechanistic Anomalies

This repository contains the code and resources required to reproduce the experiments from the paper
***Root Cause Analysis of Measurement and Mechanistic Anomalies***, submitted to **NEURIPS 2026**.

---

##  Overview

This package provides:

* Scripts to reproduce all experiments presented in the paper
* Adapted implementations of competing methods from open-source repositories
* Detailed instructions for setting up and running the evaluation pipeline

---

##  Installation

The implementation is **fully dockerized**.
We recommend running the container with a mounted directory.
From the project root, run:

```bash
# Build the Docker image
docker build -t cali-reproduce:latest .

# Run the container with a mounted volume
docker run -it -v $(pwd):/app cali-reproduce:latest
```


---

##  Public Implementations Used

We acknowledge and build upon the following publicly available codebases.

### Competitor Methods

| Repository       | Authors      | Adapted Files/Folders      |
| ---------------- | ------------ | -------------------------- |
| [M2OE](https://github.com/simona-nistico/M2OE)     | Fabrizio Angiulli, Fabio Fassetti, Simona Nisticò, Luigi Palopoli1 | `src/competitors/M2OE/` |

### Causal Discovery
| Repository                                          | Authors                                  | Adapted Files/Folders        |
| --------------------------------------------------- | ---------------------------------------- | ---------------------------- |
| [CAM, SCORE, NoGAM](https://github.com/py-why/dodiscover) | Adam Li, Jaron Lee, Francesco Montagna, Chris Trevino, Robert Ness | `src/competitors/discovery/` |
| [Topic](https://github.com/srhmm/topic)             | Sascha Xu, Sarah Mameche, Jilles Vreeken | `src/competitors/topic/`     |

Some of the above code was modified to align with the evaluation settings described in our paper.

Remaining competitors and foundational libraries (e.g., pyod, pzflow, ...) are installed directly via requirements.txt. For more information, please refer to the respective repositories.


##  Usage

To reproduce experiments, run:

```bash
python3.11 main.py -e <experiment_name>
```

where `<experiment_name>` can be one of the following:

| Experiment Name | Description |
| - | - |
| `detection` | Sample-level outlier detection on synthetic data across anomaly strengths. |
| `detection_cc_strong` | Sample-level outlier detection on Causal Chamber real data with strong interventions. |
| `detection_cc_mid` | Sample-level outlier detection on Causal Chamber real data with mid-strength interventions. |
| `detection_hyper` | Hyperparameter grid search for sample-level outlier detection methods. |
| `classification` | Anomaly-type classification on synthetic data. |
| `classification_cc_strong` | Anomaly-type classification on Causal Chamber real data with strong interventions. |
| `classification_cc_mid` | Anomaly-type classification on Causal Chamber real data with mid-strength interventions. |
| `classification_hyper` | Hyperparameter grid search for anomaly-type classification. |
| `classification_rc_count_sensitivity` | Classification sensitivity study with varying numbers of root causes per anomalous sample. |
| `classification_correlated_measurement` | Classification robustness study with correlated measurement anomalies. |
| `classification_dag_misspecification` | Classification robustness study with misspecified causal graphs. |
| `classification_heteroscedasticity` | Classification robustness study with heteroscedastic noise. |
| `classification_soft_interventions` | Classification robustness study with soft interventions. |
| `classification_confounding` | Classification robustness study with hidden confounders. |
| `classification_priors_small_default` | Classification on small synthetic graphs using prior-informed CALI variants and default root-cause generation. |
| `sample_classification` | Sample-level classification comparison against the LAROSA baseline. |
| `rca` | Root cause localization on synthetic data. |
| `rca_only_meas` | Root cause localization on synthetic data with only measurement anomalies. |
| `rca_only_mech` | Root cause localization on synthetic data with only mechanistic anomalies. |
| `rca_cc_strong` | Root cause localization on Causal Chamber real data with strong interventions. |
| `rca_cc_mid` | Root cause localization on Causal Chamber real data with mid-strength interventions. |
| `rca_sample_sensitivity` | Root cause localization sensitivity study with varying sample sizes. |
| `rca_feature_sensitivity` | Root cause localization sensitivity study with varying numbers of features. |
| `rca_edge_prob_sensitivity` | Root cause localization sensitivity study with varying graph densities. |
| `rca_rc_count_sensitivity` | Root cause localization sensitivity study with varying numbers of root causes per anomalous sample. |
| `rca_correlated_measurement` | Root cause localization robustness study with correlated measurement anomalies. |
| `rca_dag_misspecification` | Root cause localization robustness study with misspecified causal graphs. |
| `rca_heteroscedasticity` | Root cause localization robustness study with heteroscedastic noise. |
| `rca_soft_interventions` | Root cause localization robustness study with soft interventions. |
| `rca_confounding` | Root cause localization robustness study with hidden confounders. |
| `rca_priors_small_default` | Root cause localization on small synthetic graphs using prior-informed CALI variants and default root-cause generation. |
| `causal_discovery` | Comparison of CALI's downstream performance with different causal discovery methods. |
| `sachs` | Classification and root cause localization on the Sachs real-world dataset. |



---

##  License

* **Code developed for this package:** CC BY 4.0
* **External code:** Remains under its original licenses. See the respective repositories and the top-level `THIRD_PARTY_NOTICES.md`.

---
