import numpy as np
import pandas as pd
import os
import pickle
import json
import time
from sklearn.metrics import roc_auc_score, average_precision_score
from ..competitors.causal_discovery import cam_wrapper,dag_search_ae_anomalies
from .utils import evaluate_feature_level, evaluate_feature_ap_by_type, evaluate_feature_auc_by_type
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score

def sample_wise_localization(scores: np.ndarray, y: np.ndarray, plot: bool=True,plot_file=""):
    """
    Computes per-sample average precision (AP) over sample-wise sorted feature scores.
    Also plots cumulative localization.

    Parameters
    ----------
    scores : np.ndarray
        Shape (n_samples, n_features). Feature scores per sample.
    y : np.ndarray
        Shape (n_samples, n_features). Binary ground truth per sample.
    plot : bool, default=True
        Whether to plot cumulative localization.

    Returns
    -------
    aps : np.ndarray
        Per-sample average precision values.
    mean_ap : float
        Mean AP across samples.
    """
    sample_scores =scores.max(axis=1)
    
    
    aps = []
    cum_aps = []

    for row_id in np.argsort(-sample_scores):
        s = scores[row_id]
        gt = y[row_id]
        if not y[row_id].sum():
            continue

       
        ap = average_precision_score(gt, s)
        aps.append(ap)
        cum_aps.append(np.array(aps).mean())

    ranks = list(range(len(cum_aps)))
    aps = np.array(aps)
    mean_ap = aps.mean()

    if plot:
        # avg_precision_curve = cum_precision / n_samples
        plt.figure(figsize=(6,4))
        plt.plot(ranks, cum_aps, marker='o')
        plt.xlabel("Anomaly rank")
        plt.ylabel("Average Precision")
       
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.savefig(plot_file)

    return aps, mean_ap


def run_sample_ad(methods, data_dir_path,dataset_name, result_dir):
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
    for name,function in methods.items():

        # if name == "AE":
        #     function = ae_wrapper
        # elif name == "nf":
        #     function = nf_wrapper
        # elif name == "lof":
        #     function = lof_wrapper
        # elif name == "iso":
        #     function = iforest_wrapper
        # elif name == "deepsvdd":
        #     function = deepsvdd_wrapper
        # elif name == "copod":
        #     function = copod_wrapper

        for config_file_name in matching_config_files:
            # Skip if this combination is already computed
            if (name, config_file_name) in existing_pairs:
                continue
            

            if  ("CALI" in name) and (f"{name}_max", config_file_name) in existing_pairs:
                continue

            with open(f"{os.getcwd()}/{data_dir_path}/{config_file_name}", "r") as file:
                data_config = json.load(file)
            print(data_config["data_file_path"])
            data_file_path = data_config['data_file_path']

            with open(data_file_path, "rb") as file:
                total_data = pickle.load(file)

            corrupt_data = total_data["corrupt_data"]
            # # Log-transform the data
            # corrupt_data = np.log1p(corrupt_data)
            dag = total_data["dag"]
            labels = total_data["labels"]
            t0 = time.perf_counter()
            
            if "CALI" in name:
            
                if "TRUE" in name:

               
                    assignment, feature_scores, _, _ = function(data=corrupt_data,dag=dag)
                    max_scores = feature_scores.max(axis=1)
                    mean_scores = feature_scores.mean(axis=1)
               
            # ae_dag = dag_search_ae_anomalies(corrupt_data,function)
            # t2 = time.perf_counter()
                if "LEARN" in name:

                    dag = dag_search_ae_anomalies(data=corrupt_data,cd_algorithm=cam_wrapper)
                    assignment, feature_scores, _, _ = function(data=corrupt_data,dag=dag)
                    max_scores = feature_scores.max(axis=1)
                    mean_scores = feature_scores.mean(axis=1)
            else:
                print(F"Running {name}")
                scores = function(corrupt_data)
                
            t1 = time.perf_counter()
            # run_time_oracle = t1 - t0

            # run_time_ae = t2 - t1
            # params = method_config["params"].copy()  # Avoid modifying original params
            # params["data"] = corrupt_data
            # params["dag"] = dag

        
            if "CALI" in name:

                results.append({
                    "method": f"{name}_max",
                    "config_file": config_file_name,
                    "data_file_name": data_file_path,
                    "sample_scores": max_scores,
                    "run_time":t1-t0
                })

                results.append({
                    "method": f"{name}_mean",
                    "config_file": config_file_name,
                    "data_file_name": data_file_path,
                    "sample_scores": mean_scores,
                    "run_time":t1-t0
                })
            else:
                results.append({
                    "method": f"{name}",
                    "config_file": config_file_name,
                    "data_file_name": data_file_path,
                    "sample_scores": scores,
                    "run_time":t1-t0
                })

            # Save results incrementally
            with open(result_file_path, "wb") as file:
                pickle.dump(results, file, pickle.HIGHEST_PROTOCOL)


def evaluate_sample_ad(result_dir,plot_dir):

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
            data_file_path = run["data_file_name"]
           
            with open(data_file_path, "rb") as file:
                total_data = pickle.load(file)
          

            true_labels = total_data["labels"]
            corrupt_data = total_data["corrupt_data"]
            true_dag = total_data["dag"]

            # mdl_score = run["mdl_score"]
            # assignment = run["assignment"]
            sample_scores = np.array(run["sample_scores"])
            # residual_feature_scores = run["residual_feature_scores"]

            # Evaluate
            label_array = np.array([i in true_labels for i in range(corrupt_data.shape[0])])
            auc = roc_auc_score(label_array,sample_scores)
            ap = average_precision_score(label_array,sample_scores)

            # top k precision 
            k = len(true_labels)
            score_indices = np.argsort(sample_scores)[-k:][::-1]


            true_positives = sum(i in true_labels for i in score_indices)


            top_k_precision = true_positives / k



            random_ap = len(true_labels)/ corrupt_data.shape[0]
            print(f"Random AP: {random_ap}")
            
           
            metrics = [
               ("AUC", auc),
                ("AP", ap),
                ("AP Lift",(ap-random_ap)/(1-random_ap)),
                ("Top-K-Prec",top_k_precision),
                ("Top-K-Prec-Lift",(top_k_precision - random_ap)/(1-random_ap) )
            ]
         
           

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




def run_feature_ad(methods, data_dir_path,dataset_name, result_dir):
    # Load existing results if available
    os.makedirs(result_dir,exist_ok=True)
    result_file_path = f"{result_dir}/result.pkl"
    if os.path.exists(result_file_path):
        print(result_file_path)
        print(os.path.exists(result_file_path))
        with open(result_file_path, "rb") as file:
            results = pickle.load(file)
    else:
        results = []

    # Keep track of already computed (method, config_file) pairs
    existing_pairs = {(r["method"], r["config_file"]) for r in results}

    # Get all files containing the substring
    matching_config_files = [f for f in os.listdir(f"{os.getcwd()}/{data_dir_path}") if (dataset_name in f) and ("json" in f)]
    print(f"Matching config files: {matching_config_files}")
    for name,function in methods.items():

 

        for config_file_name in matching_config_files:
            # Skip if this combination is already computed
            if (name, config_file_name) in existing_pairs:
                continue
            

            if  name == ("CALI_TRUE" or name == "CALI_LEARN") and (f"{name}_max", config_file_name) in existing_pairs:
                continue

            with open(f"{os.getcwd()}/{data_dir_path}/{config_file_name}", "r") as file:
                data_config = json.load(file)

            #print(data_config["data_file_path"])
            
            data_file_path = data_config['data_file_path']

            with open(data_file_path, "rb") as file:
                total_data = pickle.load(file)

            corrupt_data = total_data["corrupt_data"]
            dag = total_data["dag"]
            labels = total_data["labels"]
            t0 = time.perf_counter()

            if "CALI"  in name:
                
                if "LEARN" not in name:
                    
                    if "ORACLE" in name:
                        assignment, feature_scores, _, _ = function(data=corrupt_data,dag=dag,labels=labels)
                    else:
                        assignment, feature_scores, _, _ = function(data=corrupt_data,dag=dag)
                    # max_scores = feature_scores.max(axis=1)
                    # mean_scores = feature_scores.mean(axis=1)

                    #
                else:
                
               
                
                    dag = dag_search_ae_anomalies(data=corrupt_data,cd_algorithm=cam_wrapper)
                    if "ORACLE" in name:
                        assignment, feature_scores, _, _ = function(data=corrupt_data,dag=dag,labels=labels)
                    else:
                        assignment, feature_scores, _, _ = function(data=corrupt_data,dag=dag)
           
            else:
                 
                n,d = corrupt_data.shape
                if  "ROOTCLAM" in name or (("SHAP" in name) and ("QGAM" in name)):
                    feature_scores = function(data=corrupt_data,dag=dag,labels=labels)
                elif name == "DICE" or ("M2OE" in name) or name == "AE-SHAP" :
                    feature_scores = function(data=corrupt_data,labels=labels)
                else:
                    feature_scores = function(corrupt_data)
                
            t1 = time.perf_counter()
            # run_time_oracle = t1 - t0

            # run_time_ae = t2 - t1
            # params = method_config["params"].copy()  # Avoid modifying original params
            # params["data"] = corrupt_data
            # params["dag"] = dag


        
            

            results.append({
                "method": f"{name}",
                "config_file": config_file_name,
                "data_file_name": data_file_path,
                "feature_scores": feature_scores,
                "run_time":t1-t0
            })

      

            # Save results incrementally
            with open(result_file_path, "wb") as file:
                pickle.dump(results, file, pickle.HIGHEST_PROTOCOL)


def evaluate_feature_ad(result_dir,plot_dir,hidden=False):

    os.makedirs(plot_dir,exist_ok=True)
    evaluation_results = []
    result_file_path = f"{result_dir}/result.pkl"
    eval_result_file_path = f"{result_dir}/eval_result.csv"
    
    with open(result_file_path,"rb") as file:
        result = pickle.load(file)

    result_df = pd.DataFrame(result)
    
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
            true_dag = total_data["dag"]

            # mdl_score = run["mdl_score"]
            # assignment = run["assignment"]
            feature_scores = run["feature_scores"]
            n,d = corrupt_data.shape
            
            if feature_scores.shape[0] != n:
                new_feature_scores = np.zeros_like(corrupt_data)

                for i, idx in enumerate(true_labels):
                    new_feature_scores[idx] = feature_scores[i]

                feature_scores = new_feature_scores

            feature_eval = evaluate_feature_level(feature_scores,true_labels,hidden,dag=true_dag)

            y = np.zeros_like(feature_scores)

            for row_id in true_labels:
                for col_id in list(true_labels[row_id]["generative"]) + list(true_labels[row_id]["measurement"]):
                    y[row_id,col_id] = 1 


            # plot_file = f"{plot_dir}/accumulated_ap_{method}_{id}.png"
            # sample_wise_localization(feature_scores,y,True,plot_file)



            #meas_ap, gen_ap = evaluate_feature_ap_by_type(feature_scores)
            # residual_feature_scores = run["residual_feature_scores"]
            metrics = [ ]
            # # Evaluate
            # label_array = np.array([i in true_labels for i in range(corrupt_data.shape[0])])
            # auc = roc_auc_score(label_array,sample_scores)
            # ap = average_precision_score(label_array,sample_scores)

            
            
            for name, val in feature_eval.items():
                metrics.append((name,val))
              
         

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