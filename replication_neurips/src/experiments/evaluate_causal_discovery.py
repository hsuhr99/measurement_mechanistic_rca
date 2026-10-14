import os

import json
import pickle
import pandas as pd
import numpy as np
from .utils import evaluate_sample_level, evaluate_feature_level, evaluate_assignment
import time
import matplotlib.pyplot as plt
from ..competitors.causal_discovery import dag_search_oracle_anomalies, dag_search_ae_anomalies, dag_search_z_anomalies
from ..cali.cali import cali
import os
import json
import pickle
import time


import numpy as np
import networkx as nx
from cdt.metrics import SHD,SID


def run_causal_discovery(cd_methods, data_dir_path,dataset_name, result_dir):
    # Load existing results if available
    os.makedirs(result_dir,exist_ok=True)
    result_file_path = f"{result_dir}/result.pkl"
    if os.path.exists(result_file_path):
        with open(result_file_path, "rb") as file:
            results = pickle.load(file)
    else:
        results = []

    # Keep track of already computed (method, config_file) pairs
    existing_pairs = {(r["method"], r["config_file"]) for r in results}

    

    # Get all files containing the substring
    matching_config_files = [f for f in os.listdir(f"{os.getcwd()}/{data_dir_path}") if (dataset_name in f) and ("json" in f)]
    print(f"Matching config files: {matching_config_files}")
    for name, function in cd_methods.items():
        

        for config_file_name in matching_config_files:
            # Skip if this combination is already computed
            if (name, config_file_name) in existing_pairs:
                continue
            if (f"{name}_oracle", config_file_name) in existing_pairs:
                continue

            with open(f"{os.getcwd()}/{data_dir_path}/{config_file_name}", "r") as file:
                data_config = json.load(file)
            print(data_config["data_file_path"])
            data_file_path = data_config['data_file_path']

            with open(data_file_path, "rb") as file:
                total_data = pickle.load(file)

            corrupt_data = total_data["corrupt_data"]
            dag = total_data["dag"]
            labels = total_data["labels"]

            t0 = time.perf_counter()
            oracle_dag = dag_search_oracle_anomalies(corrupt_data,labels,function)
            t1 = time.perf_counter()
            ae_dag = dag_search_ae_anomalies(corrupt_data,function)
            t2 = time.perf_counter()
            z_dag = dag_search_z_anomalies(corrupt_data,function)
            t3 = time.perf_counter()

            run_time_oracle = t1 - t0

            run_time_ae = t2 - t1
            # params = method_config["params"].copy()  # Avoid modifying original params
            # params["data"] = corrupt_data
            # params["dag"] = dag
            run_time_z = t3 - t2

        
            assignment_oracle, feature_scores_oracle,class_oracle,conf_oracle = cali(oracle_dag,corrupt_data)

            assignment_ae, feature_scores_ae, class_ae,conf_ae = cali(ae_dag,corrupt_data)

            assignment_z, feature_scores_z,class_z, conf_z_ = cali(z_dag,corrupt_data)
            

            results.append({
                "method": f"{name}_oracle",
                "config_file": config_file_name,
                "data_file_name": data_file_path,
              
                "assignment": assignment_oracle,
                "feature_scores": feature_scores_oracle,
                "classifications":class_oracle,
                "confidences":conf_oracle,
                "dag":oracle_dag,
                "run_time": run_time_oracle,
            })

            results.append({
                "method": f"{name}_ae",
                "config_file": config_file_name,
                "data_file_name": data_file_path,
                "classifications":class_ae,
                "confidences":conf_ae,
                "assignment": assignment_ae,
                "feature_scores": feature_scores_ae,
                "dag": ae_dag,
                "run_time": run_time_ae,
            })

            results.append({
                "method": f"{name}_z",
                "config_file": config_file_name,
                "data_file_name": data_file_path,
                "classifications":class_z,
                "confidences":conf_z_,
                "assignment": assignment_z,
                "feature_scores": feature_scores_z,
                "dag": z_dag,
                "run_time": run_time_z,
            })

            # Save results incrementally
            with open(result_file_path, "wb") as file:
                pickle.dump(results, file, pickle.HIGHEST_PROTOCOL)


def remove_inf_nan(data):
    return np.nan_to_num(
        data,
        nan=0.0,
        posinf=np.finfo(data.dtype).max,
        neginf=np.finfo(data.dtype).min
    )


def evaluate_causal_discovery(result_dir, plot_dir):
    os.makedirs(plot_dir, exist_ok=True)
    result_file_path = f"{result_dir}/result.pkl"
    eval_result_file_path = f"{result_dir}/eval_result.csv"
    evaluation_results = []

    with open(result_file_path, "rb") as file:
        result = pickle.load(file)

    result_df = pd.DataFrame(result)
    
    for method in result_df["method"].unique():
        runs = result_df[result_df["method"] == method]
        classification_confidences = []

        for _, run in runs.iterrows():
            data_file_path = run["data_file_name"]
            with open(data_file_path, "rb") as file:
                total_data = pickle.load(file)

            true_labels = total_data["labels"]
            corrupt_data = total_data["corrupt_data"]
            true_dag_nx = total_data["dag"]  # nx.DiGraph
            learned_dag_nx = run["dag"]      # nx.DiGraph

            # mdl_score = run["mdl_score"]
            assignment = run["assignment"]
            feature_scores = run["feature_scores"]

            feature_scores = remove_inf_nan(feature_scores)
            # print(feature_scores.type)
            # print(feature_scores.shape)
            # ----- Existing evaluations -----
            max_sample_eval_auc, max_sample_eval_ap = evaluate_sample_level(feature_scores, true_labels, "max")
            mean_sample_eval_auc, mean_sample_eval_ap = evaluate_sample_level(feature_scores, true_labels, "mean")
            feature_level = evaluate_feature_level(feature_scores, true_labels,hidden=False,dag=true_dag_nx)
            metrics = []
            for name,val in feature_level.items():
                metrics.append((name,val))
            
            metrics.extend([
                ("max_sample_auc", max_sample_eval_auc),
                ("max_sample_ap", max_sample_eval_ap),
                ("mean_sample_auc", mean_sample_eval_auc),
                ("mean_sample_ap", mean_sample_eval_ap),
               
            ])
            n,d = corrupt_data.shape
            evaluated = []
            if assignment is not None:
                sample_f1_score, sample_precision, sample_recall, feature_f1_score, feature_precision, feature_recall, accuracy, evaluated,_ = evaluate_assignment(true_labels, assignment, true_dag_nx,n,d)
                classification_confidences += evaluated
                metrics.extend(
                    [ 
                        ("sample_f1_score", sample_f1_score),
                        ("sample_precision", sample_precision),
                        ("sample_recall", sample_recall),
                        ("feature_f1_score", feature_f1_score),
                        ("feature_precision", feature_precision),
                        ("feature_recall", feature_recall),
                        ("meas_gen_accuracy", accuracy)
                    ]
                )

            metrics.append(("run_time", run["run_time"]))

            # ----- DAG quality metrics -----
            try:
               

                shd_value = SHD(true_dag_nx, learned_dag_nx)
                # sid_value = SID(true_dag_nx, learned_dag_nx)
            except Exception as e:
                print(f"Error computing DAG metrics for {method}, {run['config_file']}: {e}")
                shd_value = sid_value = None

            metrics.extend([
                ("shd", shd_value),
                # ("sid", sid_value)
            ])

            # ----- Save metrics -----
            for metric_name, metric_val in metrics:
                evaluation_results.append({
                    "method": method,
                    "config_file": run["config_file"],
                    "metric": metric_name,
                    "value": metric_val
                })

            # ----- Plot classification confidence vs accuracy -----
            if evaluated:
                evaluated_sorted = sorted(evaluated, key=lambda x: -x["confidence"])
                accuracies = []
                gains = []
                cumulative_correct = 0
                for i, e in enumerate(evaluated_sorted, 1):
                    if e["correct"]:
                        cumulative_correct += 1

                    confidence = e["confidence"]
                    if isinstance(confidence, np.ndarray) and confidence.size:
                        confidence = confidence[0]

                    gains.append(float(confidence))
                    accuracies.append(cumulative_correct / i)

                plt.plot(gains, accuracies)
                plt.xlabel("Top-k Predictions (by gain)")
                plt.ylabel("Cumulative Accuracy")
                plt.title(f"Accuracy vs. Number of Predictions (Run {run['config_file']}, {method})")
                plt.grid(True)
                plt.savefig(f"{plot_dir}/{method}_conf_acc.png")
                plt.close()

    # Save evaluation results
    eval_result_df = pd.DataFrame(evaluation_results)
    eval_result_df.to_csv(eval_result_file_path, index=False)
    return

