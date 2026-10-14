import numpy as np
import pandas as pd
import os
import pickle
import json
import time
# from ..cali.robust_dag_scoring import score_dag
from sklearn.metrics import roc_auc_score, average_precision_score
from ..competitors.sample_level_ad import ae_wrapper, nf_wrapper, lof_wrapper, iforest_wrapper, deepsvdd_wrapper, copod_wrapper
from ..competitors.causal_discovery import nogam_wrapper,dag_search_ae_anomalies, topic_wrapper
from .utils import evaluate_feature_level, evaluate_feature_ap_by_type, evaluate_feature_auc_by_type
from scipy.stats import rankdata
from sklearn.metrics import f1_score, precision_score, recall_score
from .utils import evaluate_assignment
import os
import pickle
import time
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

from scipy.stats import spearmanr, weightedtau

def process_single_run(name, function, config_file_name, data_dir_path):
    """Runs a single (method, config_file) pair and returns results."""
    import pickle  # Needed inside subprocess
    import time

    with open(f"{os.getcwd()}/{data_dir_path}/{config_file_name}", "rb") as file:
        total_data = pickle.load(file)

    corrupt_data = total_data["data"]
    dag = total_data["dag"]
    labels = total_data["labels"]

    t0 = time.perf_counter()

    if "CALI" in name:
        if "TRUE" in name:
            assignment, feature_scores, classifications, confidences = function(
                data=corrupt_data, dag=dag
            )
            max_scores = feature_scores.max(axis=1)
            mean_scores = feature_scores.mean(axis=1)
        else:
            dag = dag_search_ae_anomalies(data=corrupt_data, cd_algorithm=nogam_wrapper)
            assignment, feature_scores, classifications, confidences = function(
                data=corrupt_data, dag=dag
            )
            max_scores = feature_scores.max(axis=1)
            mean_scores = feature_scores.mean(axis=1)
            t1 = time.perf_counter()
            return [
            {
                "method": f"{name}_max",
                "config_file": config_file_name,
                "dag":dag,
                "assignment":assignment,
                "sample_scores": max_scores,
                "classifications": classifications,
                "confidences": confidences,
                "run_time": t1 - t0,
            },
            {
                "method": f"{name}_mean",
                "config_file": config_file_name,
                "sample_scores": mean_scores,
                "run_time": t1 - t0,
            },
            ]
            

        t1 = time.perf_counter()
        return [
            {
                "method": f"{name}_max",
                "config_file": config_file_name,
                "assignment":assignment,
                "sample_scores": max_scores,
                "classifications": classifications,
                "confidences": confidences,
                "run_time": t1 - t0,
            },
            {
                "method": f"{name}_mean",
                "config_file": config_file_name,
                "sample_scores": mean_scores,
                "run_time": t1 - t0,
            },
        ]
    else:
        scores = function(corrupt_data)
        t1 = time.perf_counter()
        return [
            {
                "method": name,
                "config_file": config_file_name,
                "sample_scores": scores,
                "run_time": t1 - t0,
            }
        ]


def run_sample_ad(methods, data_dir_path, dataset_name, result_dir, max_workers=None):
    # Load existing results
    os.makedirs(result_dir, exist_ok=True)
    result_file_path = f"{result_dir}/result.pkl"
    if os.path.exists(result_file_path):
        with open(result_file_path, "rb") as file:
            results = pickle.load(file)
    else:
        results = []

    # Already computed pairs
    existing_pairs = {(r["method"], r["config_file"]) for r in results}

    # Get matching files
    matching_config_files = [
        f for f in os.listdir(f"{os.getcwd()}/{data_dir_path}")
        if (dataset_name in f) and ("pkl" in f)
    ]
    print(f"Matching config files: {matching_config_files}")

    # Parallel execution
  
    # Sequential (non-parallel) execution version
    for name, function in methods.items():
        for config_file_name in matching_config_files:
            # skip already processed runs
            if (name, config_file_name) in existing_pairs:
                continue
            if ("CALI" in name) and ((f"{name}_max", config_file_name) in existing_pairs):
                continue

            print(f"[INFO] Running {name} on {config_file_name} (sequential mode)")
        
            new_results = process_single_run(name, function, config_file_name, data_dir_path)
            results.extend(new_results)

            # Save incrementally after each run
            with open(result_file_path, "wb") as file:
                pickle.dump(results, file, pickle.HIGHEST_PROTOCOL)

       


            # # Save incrementally
            # with open(result_file_path, "wb") as file:
            #     pickle.dump(results, file, pickle.HIGHEST_PROTOCOL)



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
            data_file_path = f'src/datasets/cc_lt_full/{run["config_file"]}'
           
            with open(data_file_path, "rb") as file:
                total_data = pickle.load(file)
          

            true_labels = total_data["labels"]
            corrupt_data = total_data["data"]
            true_dag = total_data["dag"]

            

            # mdl_score = run["mdl_score"]
            # assignment = run["assignment"]
            sample_scores = np.array(run["sample_scores"])
            # residual_feature_scores = run["residual_feature_scores"]
            print(method)
            # if method == "AE":
            #     continue
            print(sample_scores.shape)

            # Evaluate
            label_array = np.array([i in true_labels for i in range(corrupt_data.shape[0])])
            print(label_array.shape)
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

            # all rows with generative outliers:
            y_true_gen = np.zeros_like(label_array)
            y_true_meas = np.zeros_like(label_array)

            if "CALI" in method:
                classifications = run["classifications"]
                assignment = run["assignment"]
                confidences = run["confidences"]

                if type(assignment) == dict:

                    sample_f1_score, sample_precision, sample_recall, feature_f1_score, feature_precision, feature_recall, accuracy, evaluated_run = evaluate_assignment(true_labels, assignment,true_dag)

                    metrics.append(("tp_accuracy",accuracy))


        
            
            correct = 0
            counter = 0
            confs_total = []
            confs_tp = []
           
            accuracies_total = []
            accuracies_tp = []
            
            for row_id in true_labels:
                for child_id in true_labels[row_id]["measurement"]:
                    y_true_meas[row_id] = 1

                    if ("CALI" in method) and (not ("mean" in method)):
                        if list(true_dag.successors(child_id)):
                            correct += classifications[row_id,child_id] == 1
                            counter += 1
                            confs_total.append(confidences[row_id,child_id])
                            accuracies_total.append(classifications[row_id,child_id] == 1)

                            if row_id in assignment and ((child_id in assignment[row_id]["generative_attributions"]) or (child_id in assignment[row_id]["measurement_corrections"])):
                                accuracies_tp.append(classifications[row_id,child_id] == 1)
                                confs_tp.append(confidences[row_id,child_id])

                for child_id in true_labels[row_id]["generative"]:
                    y_true_gen[row_id] = 1

                    if ("CALI" in method) and (not ("mean" in method)):
                        if list(true_dag.successors(child_id)):
                            correct += classifications[row_id,child_id] == 0
                            counter += 1
                            confs_total.append(confidences[row_id,child_id])
                            accuracies_total.append(classifications[row_id,child_id] == 0)

                            if row_id in assignment and ((child_id in assignment[row_id]["generative_attributions"]) or (child_id in assignment[row_id]["measurement_corrections"])):
                                accuracies_tp.append(classifications[row_id,child_id] == 1)
                                confs_tp.append(confidences[row_id,child_id])
            if ("CALI" in method) and (not ("mean" in method)):
                total_accuracy = correct/counter 
                metrics.append(("total_accuracy",total_accuracy))
                confs_total = np.array(confs_total)
                confs_tp = np.array(confs_tp)
                accuracies_total = np.array(accuracies_total)
                accuracies_tp = np.array(accuracies_tp)

                rho, p_value = weightedtau(confs_total, accuracies_total)
                rho_tp, p_value = weightedtau(confs_tp, accuracies_tp)
                metrics.append(
                    ("weightedtau",rho
                    )
                )
                metrics.append(
                    ("weightedtau_tp",rho_tp
                    )
                )

                # Filter out measurment errors 
            only_gen_mask =  ~(y_true_meas & (~y_true_gen))
            only_meas_mask =  ~(y_true_gen & (~y_true_meas))

            only_gen_ap = average_precision_score(label_array[only_gen_mask],sample_scores[only_gen_mask])
            only_gen_auc = roc_auc_score(label_array[only_gen_mask],sample_scores[only_gen_mask])

            only_meas_ap = average_precision_score(label_array[only_meas_mask],sample_scores[only_meas_mask])
            only_meas_auc = roc_auc_score(label_array[only_meas_mask],sample_scores[only_meas_mask])
            
            metrics.extend(
                [
                    ("only_gen_ap",only_gen_ap),
                    ("only_gen_auc",only_gen_auc),
                    ("only_meas_ap",only_meas_ap),
                    ("only_meas_auc",only_meas_auc)
                ]
            )

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






from concurrent.futures import ProcessPoolExecutor, as_completed
import os, pickle, time

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
            dag = dag_search_ae_anomalies(data=corrupt_data, cd_algorithm=cam_wrapper)
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
        n, d = corrupt_data.shape
        if "ROOTCLAM" in method_name or (("SHAP" in method_name) and ("QGAM" in method_name)):
            feature_scores = function(data=corrupt_data, dag=dag, labels=labels)
        elif method_name in ["DICE", "AE-SHAP"] or "M2OE" in method_name:
            feature_scores = function(data=corrupt_data, labels=labels)
        else:
            feature_scores = function(corrupt_data)

        result = {
            "method": method_name,
            "config_file": config_file_name,
            "feature_scores": feature_scores,
            "run_time": time.perf_counter() - t0,
        }

    return result


def run_feature_ad_oracle(methods, data_dir_path, dataset_name, result_dir, max_workers=None):
    os.makedirs(result_dir, exist_ok=True)
    result_file_path = f"{result_dir}/result.pkl"

    # Load existing results
    if os.path.exists(result_file_path):
        with open(result_file_path, "rb") as file:
            results = pickle.load(file)
    else:
        results = []

    existing_pairs = {(r["method"], r["config_file"]) for r in results}
    matching_config_files = [
        f for f in os.listdir(f"{os.getcwd()}/{data_dir_path}") if (dataset_name in f) and ("pkl" in f)
    ]

    print(data_dir_path)
    print(os.getcwd())
    print(matching_config_files)


    # Sequential (non-parallel) execution version
    for name, function in methods.items():
        for config_file_name in matching_config_files:
            # skip already processed runs
            if (name, config_file_name) in existing_pairs:
                continue
            if ("CALI" in name) and ((f"{name}_max", config_file_name) in existing_pairs):
                continue

            print(f"[INFO] Running {name} on {config_file_name} (sequential mode)")
          
            new_results = process_single_feature(name, function, config_file_name, data_dir_path)
            results.extend(new_results)

            # Save incrementally after each run
            with open(result_file_path, "wb") as file:
                pickle.dump(results, file, pickle.HIGHEST_PROTOCOL)



    return results



def evaluate_feature_ad_oracle(result_dir,plot_dir,hidden=False):

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
            
            data_file_path = f'src/datasets/cc_lt_full/{run["config_file"]}'
            try:
                with open(data_file_path, "rb") as file:
                    total_data = pickle.load(file)
            except Exception:
                continue
           
            true_labels = total_data["labels"]
            corrupt_data = total_data["data"]
            true_dag = total_data["dag"]

            # mdl_score = run["mdl_score"]
            # assignment = run["assignment"]
            feature_scores = run["feature_scores"]
            max_finite = np.max(feature_scores[np.isfinite(feature_scores)])

            # Replace inf and -inf with max_finite
            feature_scores[np.isinf(feature_scores)] = max_finite
            feature_scores[np.isnan(feature_scores)] = 0
            n,d = corrupt_data.shape
            print(method)
            # if "AE" in method and (not "feat" in method):
            #     continue
            print(type(feature_scores))
            if feature_scores.shape[0] != n:
                new_feature_scores = np.zeros_like(corrupt_data)

                for i, idx in enumerate(true_labels):
                    new_feature_scores[idx] = feature_scores[i]

                feature_scores = new_feature_scores

            feature_eval = evaluate_feature_level(feature_scores,true_labels,hidden,dag=true_dag)


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




from concurrent.futures import ProcessPoolExecutor, as_completed
import os, pickle, time

def process_single_feature_realistic(method_name, function, config_file_name, data_dir_path):
    """Process a single (method, config_file) pair for realistic data."""
    with open(f"{os.getcwd()}/{data_dir_path}/{config_file_name}", "rb") as file:
        total_data = pickle.load(file)

    corrupt_data = total_data["data"]
    dag = total_data["dag"]
    labels = total_data["labels"]

    # Compute sample_alpha and feature_alpha
    sample_alpha = len(labels) / corrupt_data.shape[0]
    feature_count = 0
    for _, d in labels.items():
        for _ in list(d["measurement"]) + list(d["generative"]):
            feature_count += 1
    feature_alpha = feature_count / (len(labels) * corrupt_data.shape[1])

    t0 = time.perf_counter()

    # COLI methods
    if "CALI" in method_name:
        if "LEARN" not in method_name:
            if "ORACLE" in method_name:
                assignment, feature_scores, classifications, confidences = function(
                    data=corrupt_data, dag=dag, labels=labels
                )
            else:
                assignment, feature_scores, classifications, confidences = function(
                    data=corrupt_data, dag=dag
                )
        else:
            dag = dag_search_ae_anomalies(data=corrupt_data, cd_algorithm=cam_wrapper)
            if "ORACLE" in method_name:
                assignment, feature_scores, classifications, confidences = function(
                    data=corrupt_data, dag=dag, labels=labels
                )
            else:
                assignment, feature_scores, classifications, confidences = function(
                    data=corrupt_data, dag=dag
                )

        result = {
            "method": method_name,
            "config_file": config_file_name,
            "assignment": assignment,
            "feature_scores": feature_scores,
            "classifications": classifications,
            "confidences": confidences,
            "sample_alpha": sample_alpha,
            "feature_alpha": feature_alpha,
            "run_time": time.perf_counter() - t0,
        }

    # AE methods without SHAP
    elif ("AE" in method_name) and ("SHAP" not in method_name):
        feature_scores = function(corrupt_data)
        result = {
            "method": method_name,
            "config_file": config_file_name,
            "feature_scores": feature_scores,
            "sample_alpha": sample_alpha,
            "feature_alpha": feature_alpha,
            "run_time": time.perf_counter() - t0,
        }

    # Other methods
    else:
        n, d = corrupt_data.shape
        if "ROOTCLAM" in method_name or (("SHAP" in method_name) and ("QGAM" in method_name)):
            y_pred, feature_scores = function(data=corrupt_data, dag=dag, feature_alpha=feature_alpha, sample_alpha=sample_alpha)
        elif method_name in ["DICE", "AE-SHAP"] or "M2OE" in method_name:
            y_pred, feature_scores = function(data=corrupt_data, feature_alpha=feature_alpha, sample_alpha=sample_alpha)
        else:
            y_pred, feature_scores = function(corrupt_data)

        result = {
            "method": method_name,
            "config_file": config_file_name,
            "y_pred": y_pred,
            "feature_scores": feature_scores,
            "sample_alpha": sample_alpha,
            "feature_alpha": feature_alpha,
            "run_time": time.perf_counter() - t0,
        }

    return result


def run_feature_ad_realistic(methods, data_dir_path, dataset_name, result_dir, max_workers=None):
    os.makedirs(result_dir, exist_ok=True)
    result_file_path = f"{result_dir}/result.pkl"

    # Load existing results
    if os.path.exists(result_file_path):
        with open(result_file_path, "rb") as file:
            results = pickle.load(file)
    else:
        results = []

    existing_pairs = {(r["method"], r["config_file"]) for r in results}
    matching_config_files = [
        f for f in os.listdir(f"{os.getcwd()}/{data_dir_path}") if (dataset_name in f) and ("pkl" in f)
    ]

    for name, function in methods.items():
        for config_file_name in matching_config_files:
            # skip already processed runs
            if (name, config_file_name) in existing_pairs:
                continue
            if ("CALI" in name) and ((f"{name}_max", config_file_name) in existing_pairs):
                continue

            print(f"[INFO] Running {name} on {config_file_name} (sequential mode)")
          
            new_results = process_single_feature_realistic(name, function, config_file_name, data_dir_path)
            results.extend(new_results)

            # Save incrementally after each run
            with open(result_file_path, "wb") as file:
                pickle.dump(results, file, pickle.HIGHEST_PROTOCOL)



    return results




def evaluate_feature_ad_realistic(result_dir,plot_dir,hidden=False):

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
            
            data_file_path = data_file_path = f'src/datasets/cc_lt_full/{run["config_file"]}'
            
            try:
                with open(data_file_path, "rb") as file:
                    total_data = pickle.load(file)
            except Exception:
                continue
           
            true_labels = total_data["labels"]
            corrupt_data = total_data["data"]
            true_dag = total_data["dag"]
            feature_scores = run["feature_scores"]
            y = np.zeros_like(corrupt_data)

            for row_id in true_labels:
                for col_id in list(true_labels[row_id]["generative"]) + list(true_labels[row_id]["measurement"]):
                    y[row_id,col_id] = 1

            y_true_flat = y.flatten()

            # mdl_score = run["mdl_score"]
            # assignment = run["assignment"]
            

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
                print(np.isnan(feature_scores).sum())
                print(np.isnan(sample_scores).sum())
                print(sample_scores.shape)
               
                sample_alpha = run["sample_alpha"]
                y_pred = sample_scores >= np.quantile(sample_scores,1 - sample_alpha)
                print(y_pred)
                print(y_pred.sum())
                feature_scores = feature_scores[y_pred]
            
            elif ("AE" in method) and ("SHAP" not in method):

                print(method)
                sample_scores = feature_scores.mean(axis=1)
                sample_alpha = run["sample_alpha"]
                y_pred = sample_scores >= np.quantile(sample_scores,1-sample_alpha)
                print(y_pred.shape)
                print(feature_scores.shape)
                feature_scores = feature_scores[y_pred]

                
            else:
                print(run)
                y_pred = run["y_pred"]
            print(method)

            print(y_pred)
            print(y_pred.sum())
            print(feature_scores.shape)
            y_feature_pred_005 = np.zeros_like(corrupt_data)
            y_feature_pred_one_per_row = np.zeros_like(corrupt_data)
            feature_alpha = run["feature_alpha"]
            # print(f"Feature alpha: {feature_alpha}")
            # print("Feature score range",feature_scores.max(),feature_scores.min())
            # print(f"INF calues: {np.isinf(feature_scores).sum()}")


     
            max_finite = np.max(feature_scores[np.isfinite(feature_scores)])

            # Replace inf and -inf with max_finite
            feature_scores[np.isinf(feature_scores)] = max_finite

        

            feature_pred = feature_scores >= np.quantile(feature_scores.astype(float),1-feature_alpha)

            for i,row_id in enumerate(np.nonzero(y_pred)[0]):
                y_feature_pred_005[row_id] = feature_pred[i]
                y_feature_pred_one_per_row[row_id,np.argmax(feature_scores[i])] = 1

        
            y_pred_005_flat = y_feature_pred_005.flatten()
            y_pred_one_per_row_flat = y_feature_pred_one_per_row.flatten()
            precision_005 = precision_score(y_true_flat, y_pred_005_flat)
            recall_005 = recall_score(y_true_flat, y_pred_005_flat)
            f1_005 = f1_score(y_true_flat, y_pred_005_flat)
            print(F"Method: {method}")
            print(F"Predicted {y_pred_005_flat.sum()} features to be anaomalous")
            print(F"Quantile pecision {precision_005}, recall {recall_005}")
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