import os

import json
import pickle
import pandas as pd
import numpy as np
from .utils import evaluate_sample_level, evaluate_feature_level, evaluate_assignment
import time
import matplotlib.pyplot as plt
import os
import json
import pickle
import time
from scipy.stats import spearmanr, weightedtau
from sklearn.metrics import f1_score, roc_auc_score


def _local_data_path(path):
    if os.path.exists(path):
        return path

    marker = "src/datasets/"
    path_str = str(path)
    marker_idx = path_str.find(marker)
    if marker_idx >= 0:
        candidate = path_str[marker_idx:]
        if os.path.exists(candidate):
            return candidate

    return path


def run_classification(methods, data_dir_path, result_dir):
    # Load existing results if available

    os.makedirs(result_dir,exist_ok=True)
    result_file_path = f"{result_dir}/result.pkl"
    if os.path.exists(result_file_path):
        with open(result_file_path, "rb") as file:
            results = pickle.load(file)
    else:
        results = []

    # Keep track of already computed (method, config_file) pairs
    existing_pairs = {
        (r["method"], r["config_file"])
        for r in results
        if r.get("status", "ok") == "ok"
    }

    dataset_name = "classification"

    # Get all files containing the substring
    matching_config_files = [f for f in os.listdir(data_dir_path) if (dataset_name in f) and ("json" in f)]

    for name, method_config in methods.items():
        function = method_config["method"]

        for config_file_name in matching_config_files:
            # Skip if this combination is already computed
            if (name, config_file_name) in existing_pairs:
                continue
            t0 = time.perf_counter()
            data_file_path = None
            try:
                with open(f"{data_dir_path}/{config_file_name}", "r") as file:
                    data_config = json.load(file)
                data_file_path = f"{data_config['data_file_path']}"

                with open(data_file_path, "rb") as file:
                    total_data = pickle.load(file)

                corrupt_data = total_data["corrupt_data"]
                dag = total_data.get("observed_dag", total_data.get("corrupt_dag", total_data["dag"]))
                labels = total_data["labels"]
                params = {}
                params["data"] = corrupt_data
                params["dag"] = dag

                method_params = method_config.get("params", {}).copy()
                pass_labels = method_params.pop("pass_labels", False)
                params.update(method_params)
                if pass_labels:
                    params["labels"] = labels

                if "LEARN" not in name:
                    assignment, feature_scores, classifications, confidences = function(**params)
                else:
                    from ..competitors.causal_discovery import dag_search_ae_anomalies, nogam_wrapper

                    dag = dag_search_ae_anomalies(corrupt_data, nogam_wrapper)
                    params["dag"] = dag
                    assignment, feature_scores, classifications, confidences = function(**params)

                if "BASELINE" in name:
                    root_causes = []
                    label_mask = np.zeros(corrupt_data.shape[0], dtype=bool)
                    label_mask[list(labels.keys())] = 1
                    for row_id in assignment:
                        if row_id not in labels:
                            continue

                        ass_cols = set(assignment[row_id]["generative_attributions"].keys()) | set(assignment[row_id]["measurement_corrections"].keys())
                        label_cols = set(labels[row_id]["generative"]) | set(labels[row_id]["measurement"])

                        for col_id in (ass_cols & label_cols):
                            root_causes.append((row_id, col_id))

                    params["root_causes"] = root_causes
                    params["anomaly_mask"] = label_mask
                    print(params.keys())
                    classifications = function(**params)

                t1 = time.perf_counter()

                results.append({
                    "method": name,
                    "config_file": config_file_name,
                    "data_file_name": data_file_path,
                    "classifications": classifications,
                    "assignment": assignment,
                    "feature_scores": feature_scores,
                    "confidences": confidences,
                    "run_time": t1 - t0,
                    "status": "ok",
                })
            except Exception as exc:
                t1 = time.perf_counter()
                results.append({
                    "method": name,
                    "config_file": config_file_name,
                    "data_file_name": data_file_path,
                    "run_time": t1 - t0,
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                })

            # Save results incrementally
            with open(result_file_path, "wb") as file:
                pickle.dump(results, file, pickle.HIGHEST_PROTOCOL)



def _detected_assignment_cols(assignment, row_id):
    if assignment is None or row_id not in assignment:
        return set()

    row_assignment = assignment[row_id]
    detected = set(row_assignment.get("generative_attributions", {}).keys())
    detected.update(row_assignment.get("measurement_corrections", {}).keys())
    detected.update(row_assignment.get("uncertain_origin", {}).keys())
    return detected


def _true_class_events(labels):
    events = []
    for row_id, row_labels in labels.items():
        for col_id in row_labels["measurement"]:
            events.append((row_id, col_id, 1))
        for col_id in row_labels["generative"]:
            events.append((row_id, col_id, 0))
    return events


def _common_true_positive_events(result_df):
    common_by_config = {}

    for config_file, config_runs in result_df.groupby("config_file"):
        detected_by_method = []

        for _, run in config_runs.iterrows():
            with open(_local_data_path(run["data_file_name"]), "rb") as file:
                total_data = pickle.load(file)

            true_labels = total_data["labels"]
            assignment = run["assignment"]
            detected_events = {
                (row_id, col_id, true_class)
                for row_id, col_id, true_class in _true_class_events(true_labels)
                if col_id in _detected_assignment_cols(assignment, row_id)
            }
            detected_by_method.append(detected_events)

        common_by_config[config_file] = (
            set.intersection(*detected_by_method) if detected_by_method else set()
        )

    return common_by_config


def evaluate_classification(result_dir,plot_dir,hidden=False,common_tp_only=False):

    os.makedirs(plot_dir,exist_ok=True)
    os.makedirs(result_dir,exist_ok=True)
    evaluation_results = []

    classification_results = []
    result_file_path = f"{result_dir}/result.pkl"
    eval_result_file_path = f"{result_dir}/eval_result.csv"
    with open(result_file_path,"rb") as file:
        result = pickle.load(file)

    
   
    result_df = pd.DataFrame(result)
    if "status" in result_df.columns:
        result_df = result_df[result_df["status"] == "ok"]

    common_tp_events_by_config = (
        _common_true_positive_events(result_df) if common_tp_only else {}
    )

    for method in result_df["method"].unique():
        evaluated = [] 
        runs = result_df[result_df["method"] == method]

        classification_confidences = []

        for id, run in runs.iterrows():  # iterate rows correctly
            try:
                data_file_path = run["data_file_name"]

                with open(_local_data_path(data_file_path), "rb") as file:
                    total_data = pickle.load(file)

                true_labels = total_data["labels"]
                data = total_data["corrupt_data"]
                true_dag = total_data.get("true_dag", total_data["dag"])

                assignment = run["assignment"]
                classifications = run["classifications"]
                confidences = run["confidences"]

                metrics = []
                true_classes = []
                predicted_classes = []
                classified = 0
                correct_classifications = 0

                if assignment is not None:
                    if common_tp_only:
                        class_events = common_tp_events_by_config.get(run["config_file"], set())
                        for row_id, col_id, true_class in class_events:
                            predicted_class = classifications[row_id, col_id]
                            predicted_classes.append(predicted_class)
                            true_classes.append(true_class)
                            classified += 1
                            correct_classifications += predicted_class == true_class
                    else:
                        for row_id in assignment:
                            if row_id not in true_labels:
                                continue

                            assignment_cols = _detected_assignment_cols(assignment, row_id)

                            for col_id in true_labels[row_id]["measurement"]:
                                if col_id in assignment_cols:
                                    predicted_classes.append(classifications[row_id, col_id])
                                    true_classes.append(1)
                                    classified += 1
                                else:
                                    continue

                                if classifications[row_id, col_id] == 1:
                                    correct_classifications += 1

                            for col_id in true_labels[row_id]["generative"]:
                                if col_id in assignment_cols:
                                    predicted_classes.append(classifications[row_id, col_id])
                                    true_classes.append(0)
                                    classified += 1
                                else:
                                    continue

                                if classifications[row_id, col_id] == 0:
                                    correct_classifications += 1

                    metrics.append(("macro_f1", f1_score(true_classes, predicted_classes, average="macro") if true_classes else 0))
                    metrics.append(("tp_accuracy", correct_classifications / classified if classified else 0))
                    metrics.append(("tp_classification_count", classified))

                    sample_f1_score, sample_precision, sample_recall, feature_f1_score, feature_precision, feature_recall, accuracy, evaluated_run, confusion = evaluate_assignment(true_labels, assignment, true_dag, data.shape[0], data.shape[1])
                    classification_confidences += evaluated
                    CLASSES = ["none", "generative", "measurement", "both"]

                    precision_c = {}
                    recall_c = {}
                    f1_c = {}
                    present_classes = []

                    for c in CLASSES:
                        TP = confusion[c][c]
                        FP = sum(confusion[true][c] for true in CLASSES) - TP
                        FN = sum(confusion[c][pred] for pred in CLASSES) - TP

                        precision_c[c] = TP / (TP + FP) if (TP + FP) > 0 else 0
                        recall_c[c] = TP / (TP + FN) if (TP + FN) > 0 else 0
                        f1_c[c] = (
                            2 * precision_c[c] * recall_c[c] / (precision_c[c] + recall_c[c])
                            if (precision_c[c] + recall_c[c]) > 0 else 0
                        )

                        if TP + FN > 0:
                            present_classes.append(c)

                    if present_classes:
                        macro_precision = sum(precision_c[c] for c in present_classes) / len(present_classes)
                        macro_recall = sum(recall_c[c] for c in present_classes) / len(present_classes)
                        macro_f1 = sum(f1_c[c] for c in present_classes) / len(present_classes)
                    else:
                        macro_precision = macro_recall = macro_f1 = 0

                    for c in CLASSES:
                        metrics.extend([
                            (f"{c}_precision", precision_c[c]),
                            (f"{c}_recall", recall_c[c]),
                            (f"{c}_f1", f1_c[c]),
                        ])

                else:
                    sample_f1_score = sample_precision = sample_recall = None
                    feature_f1_score = feature_precision = feature_recall = None
                    accuracy = None

                counter = 0
                accuracy = 0
                confs_total = []
                confs_tp = []

                accuracies_total = []
                accuracies_tp = []

                counter = 0
                evaluated_run = []
                max_confidence = 0
                min_confidence = 0
                for row_id in true_labels:
                    for child in true_labels[row_id]["measurement"]:
                        if true_dag.successors(child):
                            counter += 1
                            accuracy += classifications[row_id, child] == 1

                            evaluated_run.append({
                                "correct": classifications[row_id, child] == 1,
                                "confidence": confidences[row_id, child]
                            })
                            accuracies_total.append(classifications[row_id, child] == 1)
                            confs_total.append(confidences[row_id, child])

                            if row_id in assignment and ((child in assignment[row_id]["generative_attributions"]) or (child in assignment[row_id]["measurement_corrections"])):
                                accuracies_tp.append(classifications[row_id, child] == 1)
                                confs_tp.append(confidences[row_id, child])
                    for child in true_labels[row_id]["generative"]:
                        if true_dag.successors(child):
                            counter += 1
                            accuracy += classifications[row_id, child] == 0
                            evaluated_run.append({
                                "correct": classifications[row_id, child] == 0,
                                "confidence": confidences[row_id, child]
                            })
                            confs_total.append(confidences[row_id, child])
                            accuracies_total.append(classifications[row_id, child] == 0)
                            max_confidence = max(max_confidence, confidences[row_id, child])
                            min_confidence = min(min_confidence, confidences[row_id, child])

                            if row_id in assignment and ((child in assignment[row_id]["generative_attributions"]) or (child in assignment[row_id]["measurement_corrections"])):
                                confs_tp.append(confidences[row_id, child])
                                accuracies_tp.append(classifications[row_id, child] == 0)

                print("COnsider points", len(evaluated_run))

                confs_tp = np.array(confs_tp)
                accuracies_tp = np.array(accuracies_tp)

                if len(confs_tp) > 1:
                    rho_tp, _ = weightedtau(confs_tp, accuracies_tp)
                    spear_rho, p_value = spearmanr(confs_tp, accuracies_tp)
                else:
                    rho_tp = spear_rho = p_value = np.nan

                class_auc = np.nan
                if len(np.unique(accuracies_tp)) > 1:
                    class_auc = roc_auc_score(accuracies_tp, confs_tp)

                metrics.append(("weightedtau_tp", rho_tp))
                metrics.append(("spear_rho", spear_rho))
                metrics.append(("spear_p_value", p_value))
                metrics.append(("class_auc", class_auc))

                evaluated = evaluated_run

                metrics.append(("total_accuracy", accuracy / counter))
                metrics.append(("run_time", run["run_time"]))

                for metric_name, metric_val in metrics:
                    evaluation_results.append({
                        "method": method,
                        "config_file": run["config_file"],
                        "metric": metric_name,
                        "value": metric_val
                    })
            except Exception as exc:
                print(f"Skipping evaluation for method={method}, config={run.get('config_file')}: {type(exc).__name__}: {exc}")
                continue
       

      
        # Only plot if evaluated is not empty
      
        if evaluated:
            # Sort by confidence descending
            evaluated_sorted = sorted(evaluated, key=lambda x: -x["confidence"])

            bin_size = 1  # adjust as needed
            binned_accuracies = []
            binned_confidences = []

            for i in range(0, len(evaluated_sorted), bin_size):
                bin_slice = evaluated_sorted[i:i+bin_size]
                if not bin_slice:
                    continue

                # Average confidence in this bin
                avg_conf = np.mean([e["confidence"][0] if isinstance(e["confidence"], np.ndarray) and e["confidence"].size else e["confidence"]
                                    for e in bin_slice])
                binned_confidences.append(avg_conf)

                # Accuracy in this bin
                bin_acc = sum(e["correct"] for e in bin_slice) / len(bin_slice)
                binned_accuracies.append(bin_acc)
        
# Now binned_confidences and binned_accuracies can be used for plotting
# e.g., plt.plot(binned_confidences, binned_accuracies)

            print("Smallest confidence, ",min(binned_confidences))
            plt.plot(binned_confidences, binned_accuracies)
            plt.xlabel("Top-k Predictions (by gain)")
            plt.ylabel("Cumulative Accuracy")
            plt.title(f"Accuracy vs. Number of Predictions (Run {run['config_file']}, {method})")
            plt.grid(True)
            plt.savefig(f"{plot_dir}/{method}_conf_acc.png")
            plt.close()

        classification_results.append({"method":method,"confidences":binned_confidences,"accuracies":binned_accuracies})

    # Save evaluation results
    classification_result_filepath = f"{result_dir}/class_result.pkl"
    print(classification_results)
    with open(classification_result_filepath,"wb") as file:
        pickle.dump(classification_results,file,pickle.HIGHEST_PROTOCOL)

    eval_result_df = pd.DataFrame(evaluation_results)
    eval_result_df.to_csv(eval_result_file_path, index=False)

    return

       # data_file_path = 
       

        # Evaluate score based sample level and feature level AP, and AUC
        # Evaluate Assignment precision, recall F1 on sample and feature level if assignment is provided
        # Evaluate Classification Accuracy of all true positves on feature level 
