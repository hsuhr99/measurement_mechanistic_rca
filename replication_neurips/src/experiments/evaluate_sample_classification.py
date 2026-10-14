import os

import json
import pickle
import pandas as pd
import numpy as np
from .utils import evaluate_sample_level, evaluate_feature_level, evaluate_assignment
import time
import matplotlib.pyplot as plt
from ..competitors.causal_discovery import dag_search_ae_anomalies, nogam_wrapper
from ..cali.cali import cali
import os
import json
import pickle
import time
from scipy.stats import spearmanr, weightedtau
from pyod.models.auto_encoder import AutoEncoder
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import precision_recall_fscore_support


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
    existing_pairs = {(r["method"], r["config_file"]) for r in results}

    dataset_name = "classification"

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
            labels = total_data["labels"]
            params = method_config["params"].copy()  # Avoid modifying original params
            params["data"] = corrupt_data
           

            t0 = time.perf_counter()
            if "CALI" in name:
                params["dag"] = dag
                if not "LEARN" in name:
                    
                    assignment, feature_scores, classifications, confidences = cali(**params)
                
                else:
                    
                    dag = dag_search_ae_anomalies(corrupt_data,nogam_wrapper)
                    params["dag"] = dag
                    assignment, feature_scores, classifications, confidences = cali(**params)

                
                # Clean 0, measurement 1,, process 2, both = 3
                predictions = np.zeros(corrupt_data.shape[0],dtype=int)

                for row_id in assignment:

                    # Count assignt outliers
                    
                    if assignment[row_id]["generative_attributions"]:
                        
                        # if assignment[row_id]["measurement_corrections"]:
                        predictions[row_id] = 2 
                        # else:
                        #     predictions[row_id] = 2 

                    else:

                        # I f no mechanistc outliers either meas or sink node. if we assign both => multiple marginals. affected
                        
                        count = len( assignment[row_id]["measurement_corrections"]) + len(assignment[row_id]["uncertain_origin"])
                       
                        if count > 1:
                            predictions[row_id] = 2
                        else:
                            predictions[row_id] = 1

            else:

                predictions = np.zeros(corrupt_data.shape[0],dtype=int)
                
                corrupt_data = RobustScaler().fit_transform(corrupt_data)
                alpha = len(labels)/ corrupt_data.shape[0]
                model = AutoEncoder()
                model.fit(corrupt_data)

                scores = model.decision_function(corrupt_data)

                threshold = np.quantile(scores,1-alpha)

                anomalies = scores >= threshold 

                clean_data = corrupt_data[~anomalies]
          
                test_data = corrupt_data[anomalies]
                
                predictions[anomalies] = function(training_data=clean_data,test_data=test_data)
          
            t1 = time.perf_counter()

            results.append({
                "method": name,
                "config_file": config_file_name,
                "data_file_name": data_file_path,
                "predictions":predictions,
                "run_time": t1 - t0,
            })

            # Save results incrementally
            with open(result_file_path, "wb") as file:
                pickle.dump(results, file, pickle.HIGHEST_PROTOCOL)



def evaluate_classification(result_dir,plot_dir,hidden=False):

    os.makedirs(plot_dir,exist_ok=True)
    os.makedirs(result_dir,exist_ok=True)
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
            data = total_data["corrupt_data"]
            true_dag = total_data["dag"]
            
            predictions = run["predictions"]
            
            true_classes = np.zeros(data.shape[0],dtype=int) 

            for row_id in true_labels:
                
                gen_cols = true_labels[row_id]["generative"]
                meas_cols =  true_labels[row_id]["measurement"]
                col_ids = gen_cols + meas_cols

                # for col_id in col_ids:

                #     if not true_dag.successors(col_id):
                #         if col_id in gen_cols:
                        
                #             gen_cols.remove(col_id)
                #         if col_id not in meas_cols:
                #             meas_cols.append(col_id)

                if len(col_ids) > 1:
                    true_classes[row_id] = 2 
               
                else:

                    true_classes[row_id] = 1
            metrics = []
            pred_classes = predictions

            precision, recall, f1, support = precision_recall_fscore_support(
                true_classes,
                pred_classes,
                labels=[0,1,2],
                zero_division=0
            )   

    
            macro_f1 = f1_score(
                true_classes,
                pred_classes,
                labels=[0,1,2],
                average="macro",
                zero_division=0
            )

            micro_f1 = f1_score(
                true_classes,
                pred_classes,
                labels=[0,1],
                average="micro",
                zero_division=0
            )

            metrics.append((
                "micro_f1",micro_f1
            ))
            
            metrics.append((
                "macro_f1",macro_f1
            ))

            accuracy = accuracy_score(true_classes, pred_classes)

            metrics.append((
                "accuracy",accuracy
            ))


           
            for cls, p, r, f, s in zip([0,1,2,3], precision, recall, f1, support):
                
                metrics.append((
                    f"{cls}_precision",p
                ))
                metrics.append((
                    f"{cls}_recall",r
                ))
                metrics.append((
                f"{cls}_f1",f
                ))
                metrics.append((
                    f"{cls}_support",f
                ))
                

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

  