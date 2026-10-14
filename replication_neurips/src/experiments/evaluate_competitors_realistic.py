import numpy as np
import pandas as pd
import os
import pickle
import json
import time
from sklearn.metrics import roc_auc_score, average_precision_score
from ..competitors.sample_level_ad import ae_wrapper, nf_wrapper, lof_wrapper, iforest_wrapper, deepsvdd_wrapper, copod_wrapper
from ..competitors.causal_discovery import cam_wrapper,dag_search_ae_anomalies
from .utils import evaluate_feature_level, evaluate_feature_ap_by_type, evaluate_feature_auc_by_type
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_score, recall_score, f1_score




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


            sample_alpha = len(labels)/corrupt_data.shape[0]

            feature_count =0
            for _, d in labels.items():

                for _ in list(d["measurement"]) + list(d["generative"]):
                    feature_count += 1

            feature_alpha = feature_count / (len(labels) * corrupt_data.shape[1])
            t0 = time.perf_counter()

            if "CALI"  in name:
                
                if "LEARN" not in name:
                    
                    if "ORACLE" in name:
                        assignment,feature_scores,_, _ = function(data=corrupt_data,dag=dag,labels=labels)
                    else:
                       assignment, feature_scores,_, _ = function(data=corrupt_data,dag=dag)
                    # max_scores = feature_scores.max(axis=1)
                    # mean_scores = feature_scores.mean(axis=1)

                    #
                else:
                
               
                
                    dag = dag_search_ae_anomalies(data=corrupt_data,cd_algorithm=cam_wrapper)
                    if "ORACLE" in name:
                        assignment,feature_scores,_, _ = function(data=corrupt_data,dag=dag,labels=labels)
                    else:
                        assignment,feature_scores,_, _ = function(data=corrupt_data,dag=dag)

                t1 = time.perf_counter()
                results.append({
                    "method": f"{name}",
                    "config_file": config_file_name,
                    "data_file_name": data_file_path,
                    "assignment":assignment,
                    "feature_scores": feature_scores,
                    "sample_alpha":sample_alpha,
                    "feature_alpha":feature_alpha,
                    "run_time":t1-t0
                })
                
            elif (("AE" in name) and ("SHAP" not in name)):
                feature_scores = function(corrupt_data)
                t1 = time.perf_counter()
                results.append({
                    "method": f"{name}",
                    "config_file": config_file_name,
                    "data_file_name": data_file_path,
                    "feature_scores": feature_scores,
                    "sample_alpha":sample_alpha,
                    "feature_alpha":feature_alpha,
                    "run_time":t1-t0
                })
            elif  ("QGAM" in name): 

                y_pred, feature_scores = function(corrupt_data,dag=dag,sample_alpha=sample_alpha)
                t1 = time.perf_counter()
                results.append({
                    "method": f"{name}",
                    "config_file": config_file_name,
                    "data_file_name": data_file_path,
                    "feature_scores": feature_scores,
                    "y_pred":y_pred,
                    "sample_alpha":sample_alpha,
                    "feature_alpha":feature_alpha,
                    "run_time":t1-t0
                })
            else:
                 
                n,d = corrupt_data.shape
                if  "ROOTCLAM" in name or (("SHAP" in name) and ("QGAM" in name)):
                    y_pred, feature_scores = function(data=corrupt_data,dag=dag,sample_alpha=sample_alpha,feature_alpha=feature_alpha)
                elif name == "DICE" or ("M2OE" in name) or name == "AE-SHAP" :
                    y_pred, feature_scores = function(data=corrupt_data,sample_alpha=sample_alpha,feature_alpha=feature_alpha)
                else:
                    # print(name)
                    
                    y_pred, feature_scores = function(corrupt_data)
                t1 = time.perf_counter()
                results.append({
                    "method": f"{name}",
                    "config_file": config_file_name,
                    "data_file_name": data_file_path,
                    "y_pred":y_pred,
                    "sample_alpha":sample_alpha,
                    "feature_alpha":feature_alpha,
                    "feature_scores": feature_scores,
                    "run_time":t1-t0
                })

                
           
            # run_time_oracle = t1 - t0

            # run_time_ae = t2 - t1
            # params = method_config["params"].copy()  # Avoid modifying original params
            # params["data"] = corrupt_data
            # params["dag"] = dag


        
            

           
      

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
            feature_scores = run["feature_scores"]
            
            y = np.zeros_like(corrupt_data)

            for row_id in true_labels:
                for col_id in list(true_labels[row_id]["generative"]) + list(true_labels[row_id]["measurement"]):
                    y[row_id,col_id] = 1

            y_true_flat = y.flatten()

            n,d = corrupt_data.shape
            
            metrics = []
            
            if "CALI" in method:
                assignment = run["assignment"]

                y_assignment = np.zeros_like(corrupt_data)

                for row_id in assignment:
                    for col_id in list(assignment[row_id]["measurement_corrections"].keys()) + list(assignment[row_id]["generative_attributions"].keys()) + list(assignment[row_id]["uncertain_origin"].keys()):

                        y_assignment[row_id,col_id] = 1
                
                y_assignment_flat = y_assignment.flatten()
                
                precision_assignment = precision_score(y_true_flat,y_assignment_flat)
                recall_assignment = recall_score(y_true_flat,y_assignment_flat)
                f1_assignment = f1_score(y_true_flat,y_assignment_flat)
                if assignment:
                    metrics.append(("Prec. assignment",precision_assignment))
                    metrics.append(("Rec. assignment",recall_assignment))
                    metrics.append(("F1. assignment",f1_assignment))

                sample_scores = np.max(feature_scores,axis=1)
            
                sample_alpha = run["sample_alpha"]
                y_pred = sample_scores >= np.quantile(sample_scores,1-sample_alpha)
                
                feature_scores = feature_scores[y_pred]
            
            elif (("AE" in method) and ("SHAP" not in method)):

                sample_alpha = run["sample_alpha"]
                sample_scores = feature_scores.mean(axis=1)
                y_pred = sample_scores >= np.quantile(sample_scores,1-sample_alpha)
          
                feature_scores = feature_scores[y_pred]
            else:
                y_pred = run["y_pred"]
            print(method)

            print(y_pred)
            print(y_pred.sum())
            print(feature_scores.shape)
            y_feature_pred_005 = np.zeros_like(corrupt_data)
            y_feature_pred_one_per_row = np.zeros_like(corrupt_data)
            feature_alpha = run["feature_alpha"]

            feature_pred = feature_scores >= np.quantile(feature_scores,1-feature_alpha)

            for i,row_id in enumerate(np.nonzero(y_pred)[0]):
                y_feature_pred_005[row_id] = feature_pred[i]
                y_feature_pred_one_per_row[row_id,np.argmax(feature_scores[i])] = 1


           
            y_pred_005_flat = y_feature_pred_005.flatten()
            y_pred_one_per_row_flat = y_feature_pred_one_per_row.flatten()
            precision_005 = precision_score(y_true_flat, y_pred_005_flat)
            recall_005 = recall_score(y_true_flat, y_pred_005_flat)
            f1_005 = f1_score(y_true_flat, y_pred_005_flat)

            # compute precision, recall, f1 for largest feature per row
            precision_one = precision_score(y_true_flat, y_pred_one_per_row_flat)
            recall_one = recall_score(y_true_flat, y_pred_one_per_row_flat)
            f1_one = f1_score(y_true_flat, y_pred_one_per_row_flat)

       
            metrics.extend([
                ("Prec. qauntile",precision_005),
                ("Rec. quantile",recall_005),
                ("F1 quantile",f1_005),
                
                ("Prec. 1 per row",precision_one),
                ("Rec. 1 per row",recall_one),
                ("F1 1 per row",f1_one)
            ])


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