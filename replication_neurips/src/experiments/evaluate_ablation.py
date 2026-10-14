import os

import json
import pickle
import pandas as pd
import numpy as np
from .utils import evaluate_sample_level, evaluate_feature_level, evaluate_assignment
import time
import matplotlib.pyplot as plt
from ..competitors.causal_discovery import dag_search_ae_anomalies, cam_wrapper
import os
import json
import pickle
import time
from scipy.stats import spearmanr, weightedtau

def run_ablation(methods, data_dir_path, result_dir):
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

    dataset_name = "ablation"

    # Get all files containing the substring
    matching_config_files = [f for f in os.listdir(data_dir_path) if (dataset_name in f) and ("json" in f)]

    for name, method_config in methods.items():
        function = method_config["method"]

        for config_file_name in matching_config_files:
            # Skip if this combination is already computed
            if (name, config_file_name) in existing_pairs:
                continue

            with open(f"{data_dir_path}/{config_file_name}", "r") as file:
                data_config = json.load(file)
            #print(data_config)
            data_file_path = f"{data_config['data_file_path']}"

            with open(data_file_path, "rb") as file:
                total_data = pickle.load(file)

            corrupt_data = total_data["corrupt_data"]
            dag = total_data["dag"]

            params = method_config["params"].copy()  # Avoid modifying original params
            params["data"] = corrupt_data
            params["dag"] = dag

            t0 = time.perf_counter()

            if not "LEARN" in name:
                
                assignment, feature_scores, classifications, confidences = function(**params)
            
            else:
                
                dag = dag_search_ae_anomalies(corrupt_data,cam_wrapper)
                params["dag"] = dag
                assignment, feature_scores, classifications, confidences = function(**params)



            t1 = time.perf_counter()

            results.append({
                "method": name,
                "config_file": config_file_name,
                "data_file_name": data_file_path,
                "classifications": classifications,
                "assignment": assignment,
                "feature_scores": feature_scores,
                "confidences":confidences,
                "run_time": t1 - t0,
            })

            # Save results incrementally
            with open(result_file_path, "wb") as file:
                pickle.dump(results, file, pickle.HIGHEST_PROTOCOL)



def evaluate_ablation(result_dir,plot_dir,hidden=False):

    os.makedirs(plot_dir,exist_ok=True)
    os.makedirs(result_dir,exist_ok=True)
    evaluation_results = []

    classification_results = []
    result_file_path = f"{result_dir}/result.pkl"
    eval_result_file_path = f"{result_dir}/eval_result.csv"
    with open(result_file_path,"rb") as file:
        result = pickle.load(file)

    
   
    result_df = pd.DataFrame(result)

    for method in result_df["method"].unique():
        evaluated = [] 
        runs = result_df[result_df["method"] == method]

        classification_confidences = []

        for id, run in runs.iterrows():  # iterate rows correctly
            data_file_path = run["data_file_name"]

            with open(data_file_path, "rb") as file:
                total_data = pickle.load(file)

            true_labels = total_data["labels"]
            corrupt_data = total_data["corrupt_data"]
            true_dag = total_data["dag"]

         
            assignment = run["assignment"]
            feature_scores = run["feature_scores"]
            classifications = run["classifications"]
            confidences = run["confidences"]
            
          

            # Evaluate
            max_sample_eval_auc, max_sample_eval_ap = evaluate_sample_level(feature_scores, true_labels, "max")
            mean_sample_eval_auc, mean_sample_eval_ap = evaluate_sample_level(feature_scores, true_labels, "mean")
            fl = evaluate_feature_level(feature_scores, true_labels,hidden=hidden,dag=true_dag)
            
            feature_auc = fl["global_auc"]
            feature_ap = fl["global_ap"]
            
            per_sample_feature_auc = fl["per_sample_auc"]
            per_sample_feature_ap = fl["per_sample_ap"]
      
            
            metrics = [
               ("max_sample_auc", max_sample_eval_auc),
                ("max_sample_ap", max_sample_eval_ap),
                ("mean_sample_auc", mean_sample_eval_auc),
                ("mean_sample_ap", mean_sample_eval_ap),
                ("feature_auc", feature_auc),
                ("feature_ap", feature_ap),
                ("per_sample_feature_auc",per_sample_feature_auc),
                ("per_sample_feature_ap",per_sample_feature_ap)
            ]
             # ensure defined

            
            if assignment is not None:
                sample_f1_score, sample_precision, sample_recall, feature_f1_score, feature_precision, feature_recall, accuracy, evaluated_run = evaluate_assignment(true_labels, assignment,true_dag)
                classification_confidences += evaluated
                #evaluated.extend(evaluated_run)
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

               
            else:
                sample_f1_score = sample_precision = sample_recall = None
                feature_f1_score = feature_precision = feature_recall = None
                accuracy = None
            
            # Calculate full meas/ gen accuracy:
            counter = 0
            accuracy = 0
            confs_total = []
            confs_tp = []
            import numpy as np
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
                        accuracy += classifications[row_id,child] == 1
                       
                        evaluated_run.append({
                            "correct": classifications[row_id,child] == 1,
                            "confidence": confidences[row_id,child]
                        })
                        accuracies_total.append(classifications[row_id,child] == 1)
                        confs_total.append(confidences[row_id,child])
                        # max_confidence = max(max_confidence,confidences[row_id,child])
                        # min_confidence = min(min_confidence,confidences[row_id,child])  

                        if row_id in assignment and ((child in assignment[row_id]["generative_attributions"]) or (child in assignment[row_id]["measurement_corrections"])):
                            accuracies_tp.append(classifications[row_id,child] == 1)
                            confs_tp.append(confidences[row_id,child])
                for child in true_labels[row_id]["generative"]:
                    
                    if true_dag.successors(child):
                        counter += 1
                        accuracy += classifications[row_id,child] == 0
                        evaluated_run.append({
                            "correct": classifications[row_id,child] == 0,
                            "confidence": confidences[row_id,child]
                        })
                        confs_total.append(confidences[row_id,child])
                        accuracies_total.append(classifications[row_id,child] == 0)
                        max_confidence = max(max_confidence,confidences[row_id,child])
                        min_confidence = min(min_confidence,confidences[row_id,child])  

                        if row_id in assignment and ((child in assignment[row_id]["generative_attributions"]) or (child in assignment[row_id]["measurement_corrections"])):
                            confs_tp.append(confidences[row_id,child])
                            accuracies_tp.append(classifications[row_id,child] == 0)




            print("COnsider points",len(evaluated_run))
            # for element in evaluated_run:
                # if max_confidence > min_confidence:
                #     element["confidence"] = (element["confidence"] - min_confidence) / (max_confidence - min_confidence)
                # # else:
                #     element["confidence"] = 0.0

            
            rho, p_value = weightedtau(confs_total, accuracies_total)

            confs_total = np.array(confs_total)
            confs_tp = np.array(confs_tp)
            accuracies_total = np.array(accuracies_total)
            accuracies_tp = np.array(accuracies_tp)
            
            weighted_accuracy_total = np.sum(confs_total *accuracies_total) / np.sum(confs_total)

            weighted_accuracy_tp = np.sum(confs_tp *accuracies_tp) / np.sum(confs_tp) 
            
            # metrics.append(
            #     ("weighted_accuracy_total", weighted_accuracy_total
            #     )
            # )

            # metrics.append(
            #     ("weighted_accuracy_tp", weighted_accuracy_tp
            #     )
            # )
            rho_tp, p_value = weightedtau(confs_tp, accuracies_tp)
            metrics.append(
                ("weightedtau",rho
                )
            )
            metrics.append(
                ("weightedtau_tp",rho_tp
                )
            )
            
            # metrics.append(
            #     ("weightedtau_significance",p_value
            #     )
            # )
            
            evaluated = evaluated_run    
            
            metrics.append(
                ("total_accuracy",accuracy/counter)
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
       
    eval_result_df = pd.DataFrame(evaluation_results)
    eval_result_df.to_csv(eval_result_file_path, index=False)

    return

       # data_file_path = 
       

        # Evaluate score based sample level and feature level AP, and AUC
        # Evaluate Assignment precision, recall F1 on sample and feature level if assignment is provided
        # Evaluate Classification Accuracy of all true positves on feature level 

