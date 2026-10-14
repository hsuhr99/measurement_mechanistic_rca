import os 
import json 
import pickle 
import pandas as pd
import time 
from ..competitors.causal_discovery import dag_search_ae_anomalies, nogam_wrapper, lingam_wrapper
import numpy as np
from .utils import evaluate_feature_level
from sklearn.metrics import f1_score, precision_score, recall_score
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
import faulthandler
faulthandler.enable()  # you already do this
import traceback

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

    sample_alpha = len(labels) / corrupt_data.shape[0]

    all_errors = set()

    for row_id in labels:
        
        for col_id in list(labels[row_id]["measurement"]) + list(labels[row_id]["generative"]):
            
            all_errors.add((row_id,col_id))
    feature_alpha = len(all_errors) / (len(labels) * corrupt_data.shape[1])

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
            "classifications": classifications,
            "confidences": confidences,
            "run_time": time.perf_counter() - t0,
        }
    else:
        # Other methods

        if "REALISTIC" in method_name:
            
            predictions =  function(corrupt_data,dag=dag,feature_alpha=feature_alpha,sample_alpha=sample_alpha )

            result = {
                "method": method_name,
                "config_file": config_file_name,
                "predictions": predictions,
                "run_time": time.perf_counter() - t0,
            }

        else:

            n, d = corrupt_data.shape
        
            feature_scores = function(corrupt_data,dag=dag,eval_indices=np.array(list(labels.keys()))  )

            result = {
                "method": method_name,
                "config_file": config_file_name,
                "feature_scores": feature_scores,
                "run_time": time.perf_counter() - t0,
            }

    return result



def run_rca(methods, data_dir_path, dataset_name, result_dir):
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

    # max_workers = None  # or set explicitly, e.g. os.cpu_count()

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {
            executor.submit(_worker, task): task
            for task in tasks
        }

        for future in as_completed(future_to_task):
            

   

            try:
                new_results = future.result()
            except Exception as e:
    
                traceback.print_exc()
                raise e
            results.append(new_results)

            # Incremental save (safe: only main process writes)
            with open(result_file_path, "wb") as file:
                pickle.dump(results, file, pickle.HIGHEST_PROTOCOL)
    # for (name, function, config_file_name, data_dir_path) in tasks:
    #     results.append(process_single_feature(name,function,config_file_name,data_dir_path))



    return results



def evaluate_rca(result_dir,experiement,plot_dir,hidden=False):

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
            
            data_file_path = f'src/datasets/{experiement}/{run["config_file"]}'
    
            with open(data_file_path, "rb") as file:
                total_data = pickle.load(file)
      
           
            true_labels = total_data["labels"]
            corrupt_data = total_data["data"]
            true_dag = total_data["dag"]
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
                

            elif "CALI" in method:

                # Calculate F1 scores, and feature_eval 

                assignment = run["assignment"]
                feature_scores = run["feature_scores"]

                feature_scores = np.where(np.isnan(feature_scores),0,feature_scores)
                feature_scores = np.where(np.isinf(feature_scores),1e12,feature_scores)
                y_pred = np.zeros_like(y_true,dtype=bool)
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