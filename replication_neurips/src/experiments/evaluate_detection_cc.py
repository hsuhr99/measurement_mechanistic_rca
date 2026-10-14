import os 
import json 
import pickle 
import pandas as pd
import time 
from ..competitors.causal_discovery import dag_search_ae_anomalies, nogam_wrapper
import numpy as np
from .utils import evaluate_feature_level
from sklearn.metrics import f1_score, precision_score, recall_score,roc_auc_score, average_precision_score
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed

def _worker(args):
    """Wrapper so it can be pickled by multiprocessing."""
    name, function, config_file_name, data_dir_path = args
    print(f"[INFO] Running {name} on {config_file_name} (parallel mode)")
    return process_single_feature(name, function, config_file_name, data_dir_path)

def process_single_feature(method_name, function, config_file_name, data_dir_path):
    """Process a single (method, config_file) pair."""
    with open(f"{os.getcwd()}/{data_dir_path}/{config_file_name}", "rb") as file:
        total_data = pickle.load(file)

    corrupt_data = total_data["data"]
    dag = total_data["dag"]
    labels = total_data["labels"]

    t0 = time.perf_counter()

    # Oracle methods
    if "CALI" in method_name:
        if "LEARN" not in method_name:
            assignment, feature_scores, classifications, confidences = function(
                data=corrupt_data, dag=dag, labels=labels
            )
        else:
            dag = dag_search_ae_anomalies(data=corrupt_data, cd_algorithm=nogam_wrapper)
            assignment, feature_scores, classifications, confidences = function(
                data=corrupt_data, dag=dag, labels=labels
            )

        result = {
            "method": method_name,
            "config_file": config_file_name,
            "assignment": assignment,
            "feature_scores": feature_scores,
            "scores":feature_scores.max(axis=1),
            "classifications": classifications,
            "confidences": confidences,
            "run_time": time.perf_counter() - t0,
        }
    else:
        # Other methods
        n, d = corrupt_data.shape
      
        sample_scores = function(corrupt_data )

        result = {
            "method": method_name,
            "config_file": config_file_name,
            "scores": sample_scores,
            "run_time": time.perf_counter() - t0,
        }

    return result



def run_detection(methods, data_dir_path, dataset_name, result_dir):
    max_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
    print("MAX_WORKERS",max_workers)
    os.makedirs(result_dir, exist_ok=True)
    result_file_path = f"{result_dir}/result.pkl"

    # Load existing results
    if os.path.exists(result_file_path):
        with open(result_file_path, "rb") as file:
            results = pickle.load(file)
    else:
        results = []
    print(type(results))
    for r in results:
        print(r)
    existing_pairs = {(r["method"], r["config_file"]) for r in results}
    print(existing_pairs)
    matching_config_files = [
        f for f in os.listdir(f"{os.getcwd()}/{data_dir_path}") if (dataset_name in f) and ("pkl" in f)
    ]

    print(data_dir_path)
    print(os.getcwd())
    print(matching_config_files)

    tasks = []

    for name, function in methods.items():
        for config_file_name in matching_config_files:
            # skip already processed runs
            if (name, config_file_name) in existing_pairs:
                continue
            if ("CALI" in name) and ((f"{name}_max", config_file_name) in existing_pairs):
                continue

            tasks.append((name, function, config_file_name, data_dir_path))

        # --------------------------------------------------
        # Parallel execution
        # --------------------------------------------------
    # results = results  # keep existing results if already loaded

    max_workers = None  # or set explicitly, e.g. os.cpu_count()

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {
            executor.submit(_worker, task): task
            for task in tasks
        }

        for future in as_completed(future_to_task):
            try:
                new_results = future.result()
                results.append(new_results)

                # Incremental save (safe: only main process writes)
                with open(result_file_path, "wb") as file:
                    pickle.dump(results, file, pickle.HIGHEST_PROTOCOL)

            except Exception as e:
                name, _, config_file_name, _ = future_to_task[future]
                print(f"[ERROR] {name} on {config_file_name} failed: {e}")



    return results



def evaluate_detection(result_dir,experiment,plot_dir,hidden=False):

    #os.makedirs(plot_dir,exist_ok=True)
    evaluation_results = []
    result_file_path = f"{result_dir}/result.pkl"
    eval_result_file_path = f"{result_dir}/eval_result.csv"
    
    with open(result_file_path,"rb") as file:
        result = pickle.load(file)

    result_df = pd.DataFrame(result)
    
    for method in result_df["method"].unique():

        runs = result_df[result_df["method"] == method]

        for id, run in runs.iterrows():  # iterate rows correctly
            
            data_file_path = f'src/datasets/{experiment}/{run["config_file"]}'
    
            with open(data_file_path, "rb") as file:
                total_data = pickle.load(file)
      
           
            true_labels = total_data["labels"]
            corrupt_data = total_data["data"]
            true_dag = total_data["dag"]

            # mdl_score = run["mdl_score"]
            # assignment = run["assignment"]
            scores = run["scores"]
            scores = np.array(scores, copy=True)

       

            max_finite = np.max(scores[np.isfinite(scores)])

            # Replace inf and -inf with max_finite
            scores[np.isinf(scores)] = max_finite
            scores[np.isnan(scores)] = 0
            n,d = corrupt_data.shape
            print(method)
            # if "AE" in method and (not "feat" in method):
            #     continue
            print(type(scores))
            y_true = np.zeros_like(scores,dtype=int)
            for row_id in true_labels:
                y_true[row_id] = 1

            metrics = [
                ("AUC",roc_auc_score(y_true,scores)),
                ("AP",average_precision_score(y_true,scores))
            ]

            


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
