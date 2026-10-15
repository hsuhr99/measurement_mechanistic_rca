"""High-level Python API for measurement-mechanistic RCA."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any, Literal

import networkx as nx
import numpy as np
from sklearn.preprocessing import RobustScaler

from ._replication import ensure_replication_on_path

ensure_replication_on_path()

from cali.cali import calculate_residuals, cali, regress_dag  # noqa: E402
from cali.cali_with_priors import (  # noqa: E402
    STATE_BOTH,
    STATE_GENERATIVE,
    STATE_MEASUREMENT,
    STATE_NONE,
    _assignment_record,
    _local_flip_scores,
    _prepare_probabilities,
    _silent_sample_cost,
    _state_indices,
    _state_probabilities,
    cali_with_priors_exhaustive,
    cali_with_priors_single_root,
)
from cali.sampling import KNNSampler, RandomSampler  # noqa: E402
from cali.utils import fit_dists  # noqa: E402


RegressionBackend = Literal["quantile", "expectile", "qgam"]
PriorMode = Literal["single_root", "exhaustive"]


AnomalyType = Literal["measurement", "mechanistic", "uncertain", "both"]


@dataclass(frozen=True)
class RootCause:
    """One ranked candidate, with a conditional type-confidence score."""

    feature: int
    anomaly_type: AnomalyType
    score: float
    conditional_confidence: float


@dataclass
class RCAResult:
    """RCA assignments and per-sample, per-feature scores.

    Assignment groups are measurement, mechanistic, uncertain, and (when
    selected by inference with priors) both. Ranked candidates need not be
    selected assignments. Conditional confidence is a cost difference, not
    a probability; its definition follows the selected inference mode.
    """

    assignment: dict[int, dict[str, Any]]
    feature_scores: np.ndarray
    classifications: np.ndarray
    confidences: np.ndarray

    def __post_init__(self):
        key_names = {
            "measurement_corrections": "measurement",
            "generative_attributions": "mechanistic",
            "mechanistic_attributions": "mechanistic",
            "uncertain_origin": "uncertain",
        }
        self.assignment = {
            row: {
                key_names.get(key, key): (
                    {
                        feature: {
                            ("conditional_confidence" if field == "confidence" else field): value
                            for field, value in details.items()
                        }
                        for feature, details in group.items()
                    } if isinstance(group, dict) else group
                )
                for key, group in record.items()
            }
            for row, record in self.assignment.items()
        }

    @property
    def conditional_confidences(self) -> np.ndarray:
        """Conditional type-confidence scores, indexed by sample and feature."""
        return self.confidences

    def top_features(self, sample_idx: int, k: int = 5) -> list[RootCause]:
        """Return ranked candidates with named feature, type, and score fields."""
        if k < 0:
            raise ValueError("k must be non-negative")
        scores = np.asarray(self.feature_scores[sample_idx])
        order = np.argsort(-scores)[:k]
        record = self.assignment.get(sample_idx, {})
        rows = []
        for feature_idx in order:
            confidence = float(self.conditional_confidences[sample_idx, feature_idx])
            # Explicit assignments preserve unresolved origins and joint states.
            kind = next(
                (label for label in ("both", "uncertain", "measurement", "mechanistic")
                 if feature_idx in record.get(label, {})),
                None,
            )
            if kind is None:
                if confidence == 0 or not np.isfinite(confidence):
                    kind = "uncertain"
                else:
                    kind = "measurement" if self.classifications[sample_idx, feature_idx] >= 0.5 else "mechanistic"
            rows.append(RootCause(
                feature=int(feature_idx),
                anomaly_type=kind,
                score=float(scores[feature_idx]),
                conditional_confidence=confidence,
            ))
        return rows


def _make_result(raw_result, *, with_priors=False):
    assignment, scores, classifications, confidences = raw_result
    if with_priors:
        # The prior backend stores STATE_BOTH under its legacy uncertain key.
        assignment = {
            row: {("both" if key == "uncertain_origin" else key): value
                  for key, value in record.items()}
            for row, record in assignment.items()
        }
    return RCAResult(assignment, scores, classifications, confidences)


def labels_to_priors(labels: dict[int, dict[str, list[int]]], n_features: int, smoothing: float = 1e-3):
    """Estimate measurement and mechanistic priors; legacy "generative" labels are accepted."""
    measurement = np.full(n_features, smoothing, dtype=float)
    generative = np.full(n_features, smoothing, dtype=float)
    denom = max(len(labels), 1) + 2.0 * smoothing

    for row_labels in labels.values():
        for feature_idx in row_labels.get("measurement", []):
            measurement[feature_idx] += 1.0
        for feature_idx in set(row_labels.get("mechanistic", [])) | set(row_labels.get("generative", [])):
            generative[feature_idx] += 1.0

    return np.clip(measurement / denom, 1e-12, 1.0 - 1e-12), np.clip(generative / denom, 1e-12, 1.0 - 1e-12)


class MeasurementMechanisticRCA:
    """Convenience wrapper around the method implementation.

    Use ``fit_explain`` when the data you pass already contains anomalous
    samples. Use ``fit`` followed by ``explain`` when mechanisms are learned
    from clean data and root-cause inference is run on later samples.

    ``generative_priors`` remains a legacy keyword alias for ``mechanistic_priors``.
    Pass only one of these names.
    """

    def __init__(
        self,
        regression_backend: RegressionBackend = "expectile",
        prior_mode: PriorMode = "single_root",
        sampler: Literal["DEFAULT", "KNN"] = "DEFAULT",
        measurement_setting: Literal["MARGINALIZE", "MAP"] = "MARGINALIZE",
        regression_settings: dict[str, Any] | None = None,
    ):
        self.regression_backend = regression_backend
        self.prior_mode = prior_mode
        self.sampler = sampler
        self.measurement_setting = measurement_setting
        self.regression_settings = regression_settings or {}

    def fit(self, clean_data: np.ndarray, dag: nx.DiGraph) -> "MeasurementMechanisticRCA":
        """Learn mechanisms and residual distributions from uncontaminated data."""
        data = _as_2d_array(clean_data)
        self.dag_ = dag.copy()
        self.scaler_ = RobustScaler().fit(data)
        self.scaled_clean_data_ = self.scaler_.transform(data)
        self.data_min_ = self.scaled_clean_data_.min(axis=0)
        self.data_max_ = self.scaled_clean_data_.max(axis=0)

        parent_child_pairs = [(child, list(self.dag_.predecessors(child))) for child in self.dag_.nodes]
        self.functions_ = regress_dag(
            parent_child_pairs,
            self.scaled_clean_data_,
            regression_alg=self._regression_alg(),
            regression_settings=self._regression_settings(),
        )
        self.residuals_ = calculate_residuals(self.functions_, self.scaled_clean_data_)
        self.residual_distributions_ = fit_dists(self.residuals_)
        return self

    def explain(
        self,
        data: np.ndarray,
        measurement_priors,
        mechanistic_priors=None,
        prior_mode: PriorMode | None = None,
        *,
        generative_priors=None,
    ) -> RCAResult:
        """Explain new samples with fitted clean mechanisms.

        Priors are required in this mode because clean training data does not
        contain enough information to infer anomaly frequencies for new rows.
        """
        mechanistic_priors = _resolve_mechanistic_priors(mechanistic_priors, generative_priors)
        if measurement_priors is None or mechanistic_priors is None:
            raise ValueError("Pass both measurement_priors and mechanistic_priors.")
        self._check_is_fitted()
        scaled_data = self.scaler_.transform(_as_2d_array(data))
        return self._explain_scaled_with_fitted_model(
            scaled_data=scaled_data,
            measurement_priors=measurement_priors,
            generative_priors=mechanistic_priors,
            prior_mode=prior_mode or self.prior_mode,
        )

    def fit_explain(
        self,
        data: np.ndarray,
        dag: nx.DiGraph,
        measurement_priors=None,
        mechanistic_priors=None,
        prior_mode: PriorMode | None = None,
        *,
        generative_priors=None,
    ) -> RCAResult:
        """Run RCA on a contaminated batch, with optional fixed priors."""
        data = _as_2d_array(data)
        mechanistic_priors = _resolve_mechanistic_priors(mechanistic_priors, generative_priors)
        if (measurement_priors is None) != (mechanistic_priors is None):
            raise ValueError("Pass both measurement_priors and mechanistic_priors, or neither.")

        if measurement_priors is None:
            result = cali(
                dag=dag,
                data=data,
                regression_alg=self._regression_alg(),
                regression_settings=self._regression_settings(),
                sampler_setting=self.sampler,
                meas_setting=self.measurement_setting,
            )
        else:
            fn = _prior_method(prior_mode or self.prior_mode)
            result = fn(
                dag=dag,
                data=data,
                measurement_probs=measurement_priors,
                generative_probs=mechanistic_priors,
                regression_alg=self._regression_alg(),
                regression_settings=self._regression_settings(),
                sampler_setting=self.sampler,
                meas_setting=self.measurement_setting,
            )
        return _make_result(result, with_priors=measurement_priors is not None)

    def _explain_scaled_with_fitted_model(
        self,
        scaled_data: np.ndarray,
        measurement_priors,
        generative_priors,
        prior_mode: PriorMode,
    ) -> RCAResult:
        n, d = scaled_data.shape
        meas_probs = _prepare_probabilities(measurement_priors, n, d, "measurement_priors")
        gen_probs = _prepare_probabilities(generative_priors, n, d, "mechanistic_priors")
        state_probs = _state_probabilities(meas_probs, gen_probs)
        residuals = calculate_residuals(self.functions_, scaled_data)
        sampler = _make_sampler(self.sampler, scaled_data, self.dag_)

        assignment = {}
        feature_scores = np.zeros((n, d), dtype=float)
        classifications = np.zeros((n, d), dtype=float)
        confidences = np.zeros((n, d), dtype=float)

        for row_id in range(n):
            baseline_states = np.zeros(d, dtype=int)
            baseline_cost, _ = _silent_sample_cost(
                row_id=row_id,
                data=scaled_data,
                data_min=self.data_min_,
                data_max=self.data_max_,
                dag=self.dag_,
                functions=self.functions_,
                residuals=residuals,
                residual_distributions=self.residual_distributions_,
                sampler=sampler,
                meas_indices=[],
                gen_indices=[],
                setting=self.measurement_setting,
            )
            baseline_total = baseline_cost - np.log2(state_probs[row_id, :, STATE_NONE]).sum()
            best_total = baseline_total
            best_states = baseline_states

            candidates = _candidate_states(prior_mode, d)
            for states in candidates:
                meas_indices, gen_indices = _state_indices(states)
                sample_cost_value, _ = _silent_sample_cost(
                    row_id=row_id,
                    data=scaled_data,
                    data_min=self.data_min_,
                    data_max=self.data_max_,
                    dag=self.dag_,
                    functions=self.functions_,
                    residuals=residuals,
                    residual_distributions=self.residual_distributions_,
                    sampler=sampler,
                    meas_indices=meas_indices,
                    gen_indices=gen_indices,
                    setting=self.measurement_setting,
                )
                total_cost = sample_cost_value - np.log2(state_probs[row_id, np.arange(d), states]).sum()
                if total_cost < best_total:
                    best_total = total_cost
                    best_states = states.copy()

            row_scores, row_classes, row_confidences = _local_flip_scores(
                row_id=row_id,
                d=d,
                best_states=best_states,
                state_probs=state_probs,
                scaled_data=scaled_data,
                data_min=self.data_min_,
                data_max=self.data_max_,
                dag=self.dag_,
                functions=self.functions_,
                residuals=residuals,
                residual_distributions=self.residual_distributions_,
                sampler=sampler,
                meas_setting=self.measurement_setting,
            )
            feature_scores[row_id] = row_scores
            classifications[row_id] = row_classes
            confidences[row_id] = row_confidences

            record = _assignment_record(
                total_cost=best_total,
                baseline_total_cost=baseline_total,
                states=best_states,
                per_feature_confidences=row_confidences,
            )
            if record is not None:
                assignment[row_id] = record

        return _make_result((assignment, feature_scores, classifications, confidences), with_priors=True)

    def _regression_alg(self) -> str:
        if self.regression_backend == "quantile":
            return "SKLEARN_QUANTILE_SPLINE"
        if self.regression_backend == "expectile":
            return "PYGAM_EXPECTILE"
        if self.regression_backend == "qgam":
            return "QGAM"
        raise ValueError(f"Unsupported regression_backend={self.regression_backend!r}")

    def _regression_settings(self) -> dict[str, Any]:
        settings = {
            "mode": "default",
            "scale": False,
            "loss": "L1",
            "discrete": False,
            "qu": 0.5,
        }
        settings.update(self.regression_settings)
        return settings

    def _check_is_fitted(self):
        if not hasattr(self, "functions_"):
            raise ValueError("Call fit(clean_data, dag) before explain(...).")


def run_rca(data: np.ndarray, dag: nx.DiGraph, **kwargs) -> RCAResult:
    """One-call convenience function for contaminated-batch RCA."""
    return MeasurementMechanisticRCA(**kwargs).fit_explain(data=data, dag=dag)


def _prior_method(prior_mode: PriorMode):
    if prior_mode == "single_root":
        return cali_with_priors_single_root
    if prior_mode == "exhaustive":
        return cali_with_priors_exhaustive
    raise ValueError(f"Unsupported prior_mode={prior_mode!r}")


def _candidate_states(prior_mode: PriorMode, d: int):
    if prior_mode == "single_root":
        for feature_idx in range(d):
            for state in (STATE_NONE, STATE_GENERATIVE, STATE_MEASUREMENT, STATE_BOTH):
                states = np.zeros(d, dtype=int)
                states[feature_idx] = state
                yield states
        return
    if prior_mode == "exhaustive":
        for states_tuple in product(range(4), repeat=d):
            yield np.asarray(states_tuple, dtype=int)
        return
    raise ValueError(f"Unsupported prior_mode={prior_mode!r}")


def _make_sampler(sampler: str, data: np.ndarray, dag: nx.DiGraph):
    if sampler == "DEFAULT":
        return RandomSampler(data, dag)
    if sampler == "KNN":
        return KNNSampler(data, dag)
    raise ValueError(f"Unsupported sampler={sampler!r}")


def _as_2d_array(data: np.ndarray) -> np.ndarray:
    arr = np.asarray(data, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"Expected a two-dimensional data array, got shape {arr.shape}")
    if np.any(~np.isfinite(arr)):
        raise ValueError("Data must contain only finite values.")
    return arr


def _resolve_mechanistic_priors(mechanistic_priors, generative_priors):
    if mechanistic_priors is not None and generative_priors is not None:
        raise ValueError("Pass only mechanistic_priors; generative_priors is its legacy alias.")
    return mechanistic_priors if mechanistic_priors is not None else generative_priors
