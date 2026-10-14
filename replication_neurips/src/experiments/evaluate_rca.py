import os 
import json 
import pandas as pd 
import pickle 
import time 
import numpy as np
from ..competitors.causal_discovery import dag_search_ae_anomalies, nogam_wrapper 
from .utils import evaluate_feature_level
from sklearn.metrics import f1_score, precision_score, recall_score

import os
import pickle
import json
import time
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

def _rca_worker(args):
    name, method_config, config_file_name, data_dir_path = args

    if isinstance(method_config, dict):
        function = method_config["method"]
        method_params = method_config.get("params", {}).copy()
    else:
        function = method_config
        method_params = {}

    t0 = time.perf_counter()
    data_file_path = None
    try:
        with open(os.path.join(os.getcwd(), data_dir_path, config_file_name), "r") as file:
            data_config = json.load(file)

        data_file_path = data_config["data_file_path"]

        with open(data_file_path, "rb") as file:
            total_data = pickle.load(file)

        corrupt_data = total_data["corrupt_data"]
        dag = total_data.get("observed_dag", total_data.get("corrupt_dag", total_data["dag"]))
        labels = total_data["labels"]

        label_indices = np.array(list(labels.keys()))
        sample_alpha = len(label_indices) / corrupt_data.shape[0]

        all_errors = set()
        for row_id in labels:
            for col_id in list(labels[row_id]["measurement"]) + list(labels[row_id]["generative"]):
                all_errors.add((row_id, col_id))
        feature_alpha = len(all_errors) / (len(label_indices) * corrupt_data.shape[1])

        if "CALI" in name:
            pass_labels = method_params.pop("pass_labels", False)
            call_kwargs = {
                "data": corrupt_data,
                "dag": dag,
                **method_params,
            }
            if pass_labels:
                call_kwargs["labels"] = labels

            if "LEARN" not in name:
                assignment, feature_scores, _, _ = function(**call_kwargs)
            else:
                dag = dag_search_ae_anomalies(
                    data=corrupt_data,
                    cd_algorithm=nogam_wrapper
                )
                call_kwargs["dag"] = dag
                assignment, feature_scores, _, _ = function(**call_kwargs)

            t1 = time.perf_counter()
            return [{
                "method": name,
                "config_file": config_file_name,
                "data_file_name": data_file_path,
                "assignment": assignment,
                "feature_scores": feature_scores,
                "run_time": t1 - t0,
                "status": "ok"
            }]

        if "REALISTIC" in name:
            predictions = function(
                data=corrupt_data,
                dag=dag,
                sample_alpha=sample_alpha,
                feature_alpha=feature_alpha
            )
            t1 = time.perf_counter()
            return [{
                "method": name,
                "config_file": config_file_name,
                "data_file_name": data_file_path,
                "predictions": predictions,
                "run_time": t1 - t0,
                "status": "ok"
            }]

        feature_scores = function(
            data=corrupt_data,
            dag=dag,
            eval_indices=label_indices
        )

        t1 = time.perf_counter()
        return [{
            "method": name,
            "config_file": config_file_name,
            "data_file_name": data_file_path,
            "feature_scores": feature_scores,
            "run_time": t1 - t0,
            "status": "ok"
        }]
    except Exception as exc:
        t1 = time.perf_counter()
        return [{
            "method": name,
            "config_file": config_file_name,
            "data_file_name": data_file_path,
            "run_time": t1 - t0,
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}"
        }]


def run_rca(methods, data_dir_path, dataset_name, result_dir, max_workers=None):

    os.makedirs(result_dir, exist_ok=True)
    result_file_path = f"{result_dir}/result.pkl"

    # Load existing results
    if os.path.exists(result_file_path):
        with open(result_file_path, "rb") as file:
            results = pickle.load(file)
    else:
        results = []

    existing_pairs = {
        (r["method"], r["config_file"])
        for r in results
        if r.get("status", "ok") == "ok"
    }

    matching_config_files = [
        f for f in os.listdir(os.path.join(os.getcwd(), data_dir_path))
        if dataset_name in f and f.endswith(".json")
    ]
    print(f"Matching config files: {matching_config_files}")

    # --------------------------------------------------
    # Build task list
    # --------------------------------------------------
    tasks = []

    for name, method_config in methods.items():
        for config_file_name in matching_config_files:
            if (name, config_file_name) in existing_pairs:
                continue

            if name in ("CALI_TRUE", "CALI_LEARN") and (
                f"{name}_max", config_file_name
            ) in existing_pairs:
                continue

            tasks.append((name, method_config, config_file_name, data_dir_path))

    # --------------------------------------------------
    # Parallel execution
    # --------------------------------------------------
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {
            executor.submit(_rca_worker, task): task
            for task in tasks
        }

        for future in as_completed(future_to_task):
            new_results = future.result()
            results.extend(new_results)

            # Incremental save
            with open(result_file_path, "wb") as file:
                pickle.dump(results, file, pickle.HIGHEST_PROTOCOL)

   

    return results


def evaluate_rca(result_dir,plot_dir,hidden=False):

    os.makedirs(plot_dir,exist_ok=True)
    evaluation_results = []
    result_file_path = f"{result_dir}/result.pkl"
    eval_result_file_path = f"{result_dir}/eval_result.csv"
    
    with open(result_file_path,"rb") as file:
        result = pickle.load(file)

    result_df = pd.DataFrame(result)
    if "status" in result_df.columns:
        result_df = result_df[result_df["status"] == "ok"]
    
    for method in result_df["method"].unique():

        
        runs = result_df[result_df["method"] == method]

        for id, run in runs.iterrows():  # iterate rows correctly
            
            data_file_path = run["data_file_name"]
            try:
                with open(data_file_path, "rb") as file:
                    total_data = pickle.load(file)
            except Exception:
                continue
           
            true_labels = total_data["labels"]
            corrupt_data = total_data["corrupt_data"]
            true_dag = total_data.get("true_dag", total_data["dag"])
            
            y_true = np.zeros_like(corrupt_data,dtype=bool)
            
            for row_id in true_labels:
                
                for col_id in list(true_labels[row_id]["measurement"]) + list(true_labels[row_id]["generative"]):
            
                    y_true[row_id,col_id] = 1

            # mdl_score = run["mdl_score"]
            # assignment = run["assignment"]

            metrics = []
            if "REALISTIC" in method:
                # Calculate only f1 scores 

                y_pred = run["predictions"]
                f1 = f1_score(y_true.flatten(),y_pred.flatten())
                prec = precision_score(y_true.flatten(),y_pred.flatten())
                recall = recall_score(y_true.flatten(),y_pred.flatten())

                metrics.append((
                    "F1",f1
                ))
                metrics.append((
                    "Prec",prec,
                ))
                
                metrics.append((
                    "Recall",recall
                ))
                

            elif "CALI" in method and pd.notna(run["assignment"]) :

                # Calculate F1 scores, and feature_eval 

                assignment = run["assignment"]
                feature_scores = run["feature_scores"]

                feature_scores = np.where(np.isnan(feature_scores),0,feature_scores)
                feature_scores = np.where(np.isinf(feature_scores),1e12,feature_scores)
                y_pred = np.zeros_like(y_true,dtype=int)
                for row_id in assignment:

                    for col_id in list(assignment[row_id]["generative_attributions"]) + list(assignment[row_id]["measurement_corrections"]) +list(assignment[row_id]["uncertain_origin"]):
                        y_pred[row_id,col_id] =  1
                f1 = f1_score(y_true.flatten(),y_pred.flatten())
                prec = precision_score(y_true.flatten(),y_pred.flatten())
                recall = recall_score(y_true.flatten(),y_pred.flatten())

                metrics.append((
                    "F1",f1
                ))
                metrics.append((
                    "Prec",prec,
                ))
                
                metrics.append((
                    "Recall",recall
                ))
                

                print("Method",method)
                # print(feature_scores)

                feature_eval = evaluate_feature_level(feature_scores,true_labels,hidden,dag=true_dag)
                for name, val in feature_eval.items():
                    metrics.append((name,val))
              
         

           
            
            else:
                feature_scores = run["feature_scores"]

                if feature_scores.shape[0] != corrupt_data.shape[0]:

                    if feature_scores.shape[0] != len(true_labels):
                        raise ValueError("Feature scores size does not match data size or true labels size.")
                    new_feature_scores = np.zeros_like(corrupt_data)

                    for i, idx in enumerate(true_labels):
                        new_feature_scores[idx] = feature_scores[i]

                    feature_scores = new_feature_scores

                # print("NAN count before fix:",np.isnan(feature_scores).sum())
                # print("INF count before fix:",np.isinf(feature_scores).sum())
                feature_scores = np.where(np.isnan(feature_scores),0,feature_scores)
                feature_scores = np.where(np.isinf(feature_scores),1e12,feature_scores)


                feature_eval = evaluate_feature_level(feature_scores,true_labels,hidden,dag=true_dag)
                for name, val in feature_eval.items():
                    metrics.append((name,val))
              
                
                # calculate only featue eval 



            metrics.append(
                ("run_time", run["run_time"])
            )
           

            for metric_name, metric_val in metrics:
                evaluation_results.append({
                    "method": method,
                    "config_file": run["config_file"],
                    
                    "metric":metric_name,
                    "value":metric_val
                    
                
                })
            
           
    # Save evaluation results
    eval_result_df = pd.DataFrame(evaluation_results)
    eval_result_df.to_csv(eval_result_file_path, index=False)

    return
