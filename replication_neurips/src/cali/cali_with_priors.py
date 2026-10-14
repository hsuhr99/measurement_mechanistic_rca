import io
from contextlib import redirect_stdout
from itertools import product

import networkx as nx
import numpy as np
from sklearn.preprocessing import RobustScaler

from .cali import calculate_residuals, regress_dag, sample_cost
from .sampling import KNNSampler, RandomSampler
from .utils import fit_dists


STATE_NONE = 0
STATE_GENERATIVE = 1
STATE_MEASUREMENT = 2
STATE_BOTH = 3
STATE_NAMES = ("none", "generative", "measurement", "both")


def _prepare_probabilities(probs, n, d, name):
    arr = np.asarray(probs, dtype=float)
    if arr.ndim == 1:
        if arr.shape[0] != d:
            raise ValueError(f"{name} must have shape ({d},) or ({n}, {d}), got {arr.shape}")
        arr = np.broadcast_to(arr.reshape(1, d), (n, d)).copy()
    elif arr.ndim == 2:
        if arr.shape != (n, d):
            raise ValueError(f"{name} must have shape ({d},) or ({n}, {d}), got {arr.shape}")
        arr = arr.copy()
    else:
        raise ValueError(f"{name} must be one- or two-dimensional, got ndim={arr.ndim}")

    if np.any((arr < 0) | (arr > 1)):
        raise ValueError(f"{name} entries must lie in [0, 1]")
    return arr


def _state_probabilities(measurement_probs, generative_probs, eps=1e-12):
    meas = np.clip(measurement_probs, eps, 1.0 - eps)
    gen = np.clip(generative_probs, eps, 1.0 - eps)

    probs = np.empty(meas.shape + (4,), dtype=float)
    probs[..., STATE_NONE] = (1.0 - meas) * (1.0 - gen)
    probs[..., STATE_GENERATIVE] = (1.0 - meas) * gen
    probs[..., STATE_MEASUREMENT] = meas * (1.0 - gen)
    probs[..., STATE_BOTH] = meas * gen
    return np.clip(probs, eps, 1.0)


def _oracle_prior_arrays_from_labels(labels, n, d):
    measurement_counts = np.zeros(d, dtype=float)
    generative_counts = np.zeros(d, dtype=float)

    for _, row_labels in labels.items():
        measurement_features = row_labels.get("measurement", [])
        generative_features = row_labels.get("generative", [])

        for feature_idx in measurement_features:
            measurement_counts[feature_idx] += 1.0
        for feature_idx in generative_features:
            generative_counts[feature_idx] += 1.0

    measurement_probs = measurement_counts / n
    generative_probs = generative_counts / n

    return measurement_probs, generative_probs


def _state_indices(states):
    states = np.asarray(states, dtype=int)
    meas_indices = np.flatnonzero((states == STATE_MEASUREMENT) | (states == STATE_BOTH)).tolist()
    gen_indices = np.flatnonzero((states == STATE_GENERATIVE) | (states == STATE_BOTH)).tolist()
    return meas_indices, gen_indices


def _silent_sample_cost(
    row_id,
    data,
    data_min,
    data_max,
    dag,
    functions,
    residuals,
    residual_distributions,
    sampler,
    meas_indices,
    gen_indices,
    setting,
):
    with redirect_stdout(io.StringIO()):
        cost, clean_residuals = sample_cost(
            row_id=row_id,
            data=data,
            data_min=data_min,
            data_max=data_max,
            dag=dag,
            functions=functions,
            residuals=residuals,
            residual_distributions=residual_distributions,
            meas_indices=meas_indices,
            gen_indices=gen_indices,
            sampler=sampler,
            setting=setting,
        )
    return float(np.asarray(cost).reshape(-1)[0]), clean_residuals


def _assignment_record(total_cost, baseline_total_cost, states, per_feature_confidences):
    record = {
        "measurement_corrections": {},
        "generative_attributions": {},
        "uncertain_origin": {},
        "sample_cost": total_cost,
    }

    gain = baseline_total_cost - total_cost
    for feature_idx, state in enumerate(states):
        confidence = per_feature_confidences[feature_idx]
        if state == STATE_MEASUREMENT:
            record["measurement_corrections"][feature_idx] = {
                "base_gain": gain,
                "gain": gain,
                "confidence": confidence,
            }
        elif state == STATE_GENERATIVE:
            record["generative_attributions"][feature_idx] = {
                "base_gain": gain,
                "gain": gain,
                "confidence": confidence,
            }
        elif state == STATE_BOTH:
            record["uncertain_origin"][feature_idx] = {
                "base_gain": gain,
                "gain": gain,
                "confidence": confidence,
            }

    if not (
        record["measurement_corrections"]
        or record["generative_attributions"]
        or record["uncertain_origin"]
    ):
        return None
    return record


def _single_feature_posterior_summary(
    row_id,
    d,
    baseline_cost,
    state_log_costs,
):
    feature_scores = np.zeros(d, dtype=float)
    classifications = np.zeros(d, dtype=float)
    confidences = np.zeros(d, dtype=float)

    for feature_idx in range(d):
        generative_like = min(
            state_log_costs[(row_id, feature_idx, STATE_GENERATIVE)],
            state_log_costs[(row_id, feature_idx, STATE_BOTH)],
        )
        measurement_like = min(
            state_log_costs[(row_id, feature_idx, STATE_MEASUREMENT)],
            state_log_costs[(row_id, feature_idx, STATE_BOTH)],
        )
        best_non_none = min(generative_like, measurement_like)
        feature_scores[feature_idx] = max(baseline_cost - best_non_none, 0.0)
        classifications[feature_idx] = 1.0 if measurement_like <= generative_like else 0.0
        confidences[feature_idx] = abs(measurement_like - generative_like)

    return feature_scores, classifications, confidences


def _local_flip_scores(
    row_id,
    d,
    best_states,
    state_probs,
    scaled_data,
    data_min,
    data_max,
    dag,
    functions,
    residuals,
    residual_distributions,
    sampler,
    meas_setting,
):
    feature_scores = np.zeros(d, dtype=float)
    classifications = np.zeros(d, dtype=float)
    confidences = np.zeros(d, dtype=float)

    best_states = np.asarray(best_states, dtype=int)
    row_state_probs = state_probs[row_id]

    for feature_idx in range(d):
        base_states = best_states.copy()
        base_states[feature_idx] = STATE_NONE
        meas_indices, gen_indices = _state_indices(base_states)
        base_sample_cost, _ = _silent_sample_cost(
            row_id=row_id,
            data=scaled_data,
            data_min=data_min,
            data_max=data_max,
            dag=dag,
            functions=functions,
            residuals=residuals,
            residual_distributions=residual_distributions,
            sampler=sampler,
            meas_indices=meas_indices,
            gen_indices=gen_indices,
            setting=meas_setting,
        )
        base_total_cost = base_sample_cost - np.log2(
            row_state_probs[np.arange(d), base_states]
        ).sum()

        candidate_costs = {}
        for state in (STATE_GENERATIVE, STATE_MEASUREMENT, STATE_BOTH):
            candidate_states = base_states.copy()
            candidate_states[feature_idx] = state
            meas_indices, gen_indices = _state_indices(candidate_states)
            sample_cost_value, _ = _silent_sample_cost(
                row_id=row_id,
                data=scaled_data,
                data_min=data_min,
                data_max=data_max,
                dag=dag,
                functions=functions,
                residuals=residuals,
                residual_distributions=residual_distributions,
                sampler=sampler,
                meas_indices=meas_indices,
                gen_indices=gen_indices,
                setting=meas_setting,
            )
            total_cost = sample_cost_value - np.log2(
                row_state_probs[np.arange(d), candidate_states]
            ).sum()
            candidate_costs[state] = total_cost

        best_active_state = min(candidate_costs, key=candidate_costs.get)
        best_active_cost = candidate_costs[best_active_state]

        # Gain is measured against the same assignment with this feature forced to none.
        feature_scores[feature_idx] = base_total_cost - best_active_cost
        classifications[feature_idx] = 1.0 if best_active_state in (STATE_MEASUREMENT, STATE_BOTH) else 0.0
        confidences[feature_idx] = abs(
            min(candidate_costs[STATE_MEASUREMENT], candidate_costs[STATE_BOTH])
            - min(candidate_costs[STATE_GENERATIVE], candidate_costs[STATE_BOTH])
        )

    return feature_scores, classifications, confidences


def _setup_common(
    dag,
    data,
    measurement_probs,
    generative_probs,
    functions,
    regression_alg,
    regression_settings,
    sampler_setting,
):
    scaled_data = RobustScaler().fit_transform(data)
    n, d = scaled_data.shape
    data_min = scaled_data.min(axis=0)
    data_max = scaled_data.max(axis=0)

    parent_child_pairs = [(child, list(dag.predecessors(child))) for child in dag.nodes]
    if functions is None:
        functions = regress_dag(
            parent_child_pairs,
            scaled_data,
            regression_alg=regression_alg,
            regression_settings=regression_settings,
        )

    if sampler_setting == "KNN":
        sampler = KNNSampler(scaled_data, dag)
    elif sampler_setting == "DEFAULT":
        sampler = RandomSampler(scaled_data, dag)
    else:
        raise ValueError(f"Unsupported sampler_setting={sampler_setting}")

    residuals = calculate_residuals(functions, scaled_data)
    residual_distributions = fit_dists(residuals)

    meas_probs = _prepare_probabilities(measurement_probs, n, d, "measurement_probs")
    gen_probs = _prepare_probabilities(generative_probs, n, d, "generative_probs")
    state_probs = _state_probabilities(meas_probs, gen_probs)

    return {
        "data": scaled_data,
        "n": n,
        "d": d,
        "data_min": data_min,
        "data_max": data_max,
        "functions": functions,
        "sampler": sampler,
        "residuals": residuals,
        "residual_distributions": residual_distributions,
        "state_probs": state_probs,
    }


def cali_with_priors_exhaustive(
    dag: nx.DiGraph,
    data: np.ndarray,
    measurement_probs,
    generative_probs,
    functions=None,
    regression_alg="QGAM",
    regression_settings=None,
    sampler_setting="KNN",
    meas_setting="MARGINALIZE",
):
    """
    Exhaustive CALI variant with fixed priors.

    For each sample, this enumerates all 4^d assignments over the state space
    `none`, `generative`, `measurement`, `both`, and picks the minimum
    posterior cost:

        sample_cost(data | assignment) - sum_j log2 p(state_j)
    """
    if regression_settings is None:
        regression_settings = {
            "mode": "default",
            "scale": False,
            "loss": "L1",
            "discrete": False,
        }

    common = _setup_common(
        dag=dag,
        data=data,
        measurement_probs=measurement_probs,
        generative_probs=generative_probs,
        functions=functions,
        regression_alg=regression_alg,
        regression_settings=regression_settings,
        sampler_setting=sampler_setting,
    )

    scaled_data = common["data"]
    n = common["n"]
    d = common["d"]
    data_min = common["data_min"]
    data_max = common["data_max"]
    functions = common["functions"]
    sampler = common["sampler"]
    residuals = common["residuals"]
    residual_distributions = common["residual_distributions"]
    state_probs = common["state_probs"]

    assignment = {}
    feature_scores = np.zeros((n, d), dtype=float)
    classifications = np.zeros((n, d), dtype=float)
    confidences = np.zeros((n, d), dtype=float)

    state_log_costs = {}

    for row_id in range(n):
        baseline_states = np.zeros(d, dtype=int)
        baseline_cost, _ = _silent_sample_cost(
            row_id=row_id,
            data=scaled_data,
            data_min=data_min,
            data_max=data_max,
            dag=dag,
            functions=functions,
            residuals=residuals,
            residual_distributions=residual_distributions,
            sampler=sampler,
            meas_indices=[],
            gen_indices=[],
            setting=meas_setting,
        )
        baseline_total = baseline_cost - np.log2(state_probs[row_id, :, STATE_NONE]).sum()

        best_total = baseline_total
        best_sample_cost = baseline_cost
        best_states = baseline_states

        for states_tuple in product(range(4), repeat=d):
            states = np.asarray(states_tuple, dtype=int)
            meas_indices, gen_indices = _state_indices(states)
            sample_cost_value, _ = _silent_sample_cost(
                row_id=row_id,
                data=scaled_data,
                data_min=data_min,
                data_max=data_max,
                dag=dag,
                functions=functions,
                residuals=residuals,
                residual_distributions=residual_distributions,
                sampler=sampler,
                meas_indices=meas_indices,
                gen_indices=gen_indices,
                setting=meas_setting,
            )
            total_cost = sample_cost_value - np.log2(state_probs[row_id, np.arange(d), states]).sum()
            if total_cost < best_total:
                best_total = total_cost
                best_sample_cost = sample_cost_value
                best_states = states.copy()

        for feature_idx in range(d):
            for state in (STATE_GENERATIVE, STATE_MEASUREMENT, STATE_BOTH):
                candidate_states = np.zeros(d, dtype=int)
                candidate_states[feature_idx] = state
                meas_indices, gen_indices = _state_indices(candidate_states)
                sample_cost_value, _ = _silent_sample_cost(
                    row_id=row_id,
                    data=scaled_data,
                    data_min=data_min,
                    data_max=data_max,
                    dag=dag,
                    functions=functions,
                    residuals=residuals,
                    residual_distributions=residual_distributions,
                    sampler=sampler,
                    meas_indices=meas_indices,
                    gen_indices=gen_indices,
                    setting=meas_setting,
                )
                total_cost = sample_cost_value - np.log2(
                    state_probs[row_id, np.arange(d), candidate_states]
                ).sum()
                state_log_costs[(row_id, feature_idx, state)] = total_cost

        row_scores, row_classes, row_confidences = _local_flip_scores(
            row_id=row_id,
            d=d,
            best_states=best_states,
            state_probs=state_probs,
            scaled_data=scaled_data,
            data_min=data_min,
            data_max=data_max,
            dag=dag,
            functions=functions,
            residuals=residuals,
            residual_distributions=residual_distributions,
            sampler=sampler,
            meas_setting=meas_setting,
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

    return assignment, feature_scores, classifications, confidences


def cali_with_priors_single_root(
    dag: nx.DiGraph,
    data: np.ndarray,
    measurement_probs,
    generative_probs,
    functions=None,
    regression_alg="QGAM",
    regression_settings=None,
    sampler_setting="KNN",
    meas_setting="MARGINALIZE",
):
    """
    Single-root CALI variant with fixed priors.

    For each sample, this assumes at most one root-cause feature is active and
    compares the requested 4d candidates: for every feature, the states
    `none`, `generative`, `measurement`, `both`, with all other features fixed
    to `none`. The `none` case is therefore repeated once per feature, exactly
    matching the requested 4d search space.
    """
    if regression_settings is None:
        regression_settings = {
            "mode": "default",
            "scale": False,
            "loss": "L1",
            "discrete": False,
        }

    common = _setup_common(
        dag=dag,
        data=data,
        measurement_probs=measurement_probs,
        generative_probs=generative_probs,
        functions=functions,
        regression_alg=regression_alg,
        regression_settings=regression_settings,
        sampler_setting=sampler_setting,
    )

    scaled_data = common["data"]
    n = common["n"]
    d = common["d"]
    data_min = common["data_min"]
    data_max = common["data_max"]
    functions = common["functions"]
    sampler = common["sampler"]
    residuals = common["residuals"]
    residual_distributions = common["residual_distributions"]
    state_probs = common["state_probs"]

    assignment = {}
    feature_scores = np.zeros((n, d), dtype=float)
    classifications = np.zeros((n, d), dtype=float)
    confidences = np.zeros((n, d), dtype=float)

    state_log_costs = {}

    for row_id in range(n):
        baseline_states = np.zeros(d, dtype=int)
        baseline_cost, _ = _silent_sample_cost(
            row_id=row_id,
            data=scaled_data,
            data_min=data_min,
            data_max=data_max,
            dag=dag,
            functions=functions,
            residuals=residuals,
            residual_distributions=residual_distributions,
            sampler=sampler,
            meas_indices=[],
            gen_indices=[],
            setting=meas_setting,
        )
        baseline_total = baseline_cost - np.log2(state_probs[row_id, :, STATE_NONE]).sum()

        best_total = baseline_total
        best_sample_cost = baseline_cost
        best_states = baseline_states

        for feature_idx in range(d):
            for state in range(4):
                candidate_states = np.zeros(d, dtype=int)
                candidate_states[feature_idx] = state
                meas_indices, gen_indices = _state_indices(candidate_states)
                sample_cost_value, _ = _silent_sample_cost(
                    row_id=row_id,
                    data=scaled_data,
                    data_min=data_min,
                    data_max=data_max,
                    dag=dag,
                    functions=functions,
                    residuals=residuals,
                    residual_distributions=residual_distributions,
                    sampler=sampler,
                    meas_indices=meas_indices,
                    gen_indices=gen_indices,
                    setting=meas_setting,
                )
                total_cost = sample_cost_value - np.log2(
                    state_probs[row_id, np.arange(d), candidate_states]
                ).sum()

                if state != STATE_NONE:
                    state_log_costs[(row_id, feature_idx, state)] = total_cost

                if total_cost < best_total:
                    best_total = total_cost
                    best_sample_cost = sample_cost_value
                    best_states = candidate_states.copy()

        row_scores, row_classes, row_confidences = _local_flip_scores(
            row_id=row_id,
            d=d,
            best_states=best_states,
            state_probs=state_probs,
            scaled_data=scaled_data,
            data_min=data_min,
            data_max=data_max,
            dag=dag,
            functions=functions,
            residuals=residuals,
            residual_distributions=residual_distributions,
            sampler=sampler,
            meas_setting=meas_setting,
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

    return assignment, feature_scores, classifications, confidences


method_a = cali_with_priors_exhaustive
method_b = cali_with_priors_single_root


def cali_with_priors_exhaustive_oracle(
    dag: nx.DiGraph,
    data: np.ndarray,
    labels,
    functions=None,
    regression_alg="QGAM",
    regression_settings=None,
    sampler_setting="KNN",
    meas_setting="MARGINALIZE",
):
    measurement_probs, generative_probs = _oracle_prior_arrays_from_labels(
        labels=labels,
        n=data.shape[0],
        d=data.shape[1],
    )
    return cali_with_priors_exhaustive(
        dag=dag,
        data=data,
        measurement_probs=measurement_probs,
        generative_probs=generative_probs,
        functions=functions,
        regression_alg=regression_alg,
        regression_settings=regression_settings,
        sampler_setting=sampler_setting,
        meas_setting=meas_setting,
    )


def cali_with_priors_single_root_oracle(
    dag: nx.DiGraph,
    data: np.ndarray,
    labels,
    functions=None,
    regression_alg="QGAM",
    regression_settings=None,
    sampler_setting="KNN",
    meas_setting="MARGINALIZE",
):
    measurement_probs, generative_probs = _oracle_prior_arrays_from_labels(
        labels=labels,
        n=data.shape[0],
        d=data.shape[1],
    )
    return cali_with_priors_single_root(
        dag=dag,
        data=data,
        measurement_probs=measurement_probs,
        generative_probs=generative_probs,
        functions=functions,
        regression_alg=regression_alg,
        regression_settings=regression_settings,
        sampler_setting=sampler_setting,
        meas_setting=meas_setting,
    )
