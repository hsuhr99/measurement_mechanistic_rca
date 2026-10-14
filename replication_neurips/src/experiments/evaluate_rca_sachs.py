import os 
import json 
import pickle 
import pandas as pd
import time 
from ..competitors.causal_discovery import dag_search_ae_anomalies, nogam_wrapper
import numpy as np
from .utils import evaluate_feature_level
from sklearn.metrics import f1_score,precision_score, recall_score
from ..cali.classifier_baseline import classify
from ..cali.cali import cali
def process_single_feature(method_name, function, config_file_name, data_dir_path):
    """Process a single (method, config_file) pair."""
    with open(f"{os.getcwd()}/{data_dir_path}/{config_file_name}", "rb") as file:
        total_data = pickle.load(file)

    corrupt_data = total_data["data"]
    dag = total_data["dag"]
    labels = total_data["labels"]

    t0 = time.perf_counter()

    # Oracle methods
    if "CALI" in method_name or "BASELINE" in method_name:
        if "LEARN" not in method_name:
            assignment, feature_scores, classifications, confidences = cali(
                data=corrupt_data, dag=dag
            )
        else:
            dag = dag_search_ae_anomalies(data=corrupt_data, cd_algorithm=nogam_wrapper)
            assignment, feature_scores, classifications, confidences = cali(
                data=corrupt_data, dag=dag
            )

        if "BASELINE" in method_name:
            root_causes = []
            
            label_mask = np.zeros(corrupt_data.shape[0],dtype=bool)
            label_mask[list(labels.keys())] = 1 
            for row_id in assignment:
                
                if row_id not in labels:
                    continue 

                ass_cols = set(assignment[row_id]["generative_attributions"].keys()) | set(assignment[row_id]["measurement_corrections"].keys()) 
                
                label_cols = set(labels[row_id]["generative"]) | set(labels[row_id]["measurement"])
                
                for col_id in (ass_cols & label_cols):

                    root_causes.append((row_id,col_id))
            
            params = {
                "data":corrupt_data,
                "dag":dag
            }
            params["root_causes"] = root_causes 
            params["anomaly_mask"] = label_mask
           
            classifications = function(**params)

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
      
        feature_scores = function(corrupt_data,dag=dag,eval_indices=np.array(list(labels.keys()))  )

        result = {
            "method": method_name,
            "config_file": config_file_name,
            "feature_scores": feature_scores,
            "run_time": time.perf_counter() - t0,
        }

    return result



def run_rca(methods, data_dir_path,result_dir, max_workers=None):
    
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
        f for f in os.listdir(f"{os.getcwd()}/{data_dir_path}") if ("pkl" in f)
    ]

    # print(data_dir_path)
    # print(os.getcwd())
    # print(matching_config_files)


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
            results.append(new_results)

            # Save incrementally after each run
            with open(result_file_path, "wb") as file:
                pickle.dump(results, file, pickle.HIGHEST_PROTOCOL)



    return results



def evaluate_rca(result_dir,plot_dir,hidden=False):

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
            
            data_file_path = f'src/datasets/sachs/{run["config_file"]}'
    
            with open(data_file_path, "rb") as file:
                total_data = pickle.load(file)
      
           
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

            y_pred = np.zeros_like(feature_scores,dtype=bool)
            y_true = np.zeros_like(feature_scores,dtype=bool)

            all_errors = set()

            for row_id in true_labels:
                
                for col_id in true_labels[row_id]["measurement"] + true_labels[row_id]["generative"]:
                    
                    all_errors.add((row_id,col_id))
                    y_true[row_id,col_id] = 1

            feature_alpha = len(true_labels)/len(all_errors)

            if not (("CALI" in method) or ("BASELINE" in method)):

                print("Feature alpha")
                pred_threshold =np.quantile(feature_scores,1-feature_alpha) 

                y_pred = feature_scores >= pred_threshold

            else:
                
                
                
                assignment = run["assignment"]
                
                print(assignment)
                
                for row_id in assignment:

                    for col_id in list(assignment[row_id]["generative_attributions"]) + list(assignment[row_id]["measurement_corrections"]) +list(assignment[row_id]["uncertain_origin"]):
                        y_pred[row_id,col_id] =  1


            print("Method",method)
            # print(feature_scores)

            feature_eval = evaluate_feature_level(feature_scores,true_labels,hidden,dag=true_dag)

            # y = np.zeros_like(feature_scores)

            # for row_id in true_labels:
            #     for col_id in list(true_labels[row_id]["generative"]) + list(true_labels[row_id]["measurement"]):
            #         y[row_id,col_id] = 1 


                # print("True root causes",y[row_id])
                # print("Predicted scores",feature_scores[row_id])
            # plot_file = f"{plot_dir}/accumulated_ap_{method}_{id}.png"
            # sample_wise_localization(feature_scores,y,True,plot_file)



            #meas_ap, gen_ap = evaluate_feature_ap_by_type(feature_scores)
            # residual_feature_scores = run["residual_feature_scores"]
            metrics = [ ]
            # # Evaluate
            # label_array = np.array([i in true_labels for i in range(corrupt_data.shape[0])])
            # auc = roc_auc_score(label_array,sample_scores)
            # ap = average_precision_score(label_array,sample_scores)

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
            
            
            for name, val in feature_eval.items():
                metrics.append((name,val))
              
         

            metrics.append(
                ("run_time", run["run_time"])
            )
            
            predicted_classes = []
            true_classes = []
            
            
            if (("CALI" in method) or ("BASELINE" in method)):
                classifications = run["classifications"]
                classified = 0
                correct_classifications = 0
                
                if assignment is not None:


                    for row_id in assignment:

                        if row_id not in true_labels:
                            continue 

                        assignment_cols = list(assignment[row_id]["generative_attributions"].keys()) + list(assignment[row_id]["measurement_corrections"].keys())

                        for col_id in true_labels[row_id]["measurement"]:



                            if col_id in assignment_cols:
                                predicted_classes.append(classifications[row_id,col_id])
                                true_classes.append(1)
                                classified += 1 
                               
                            else:
                                continue

                            if classifications[row_id,col_id] ==1:
                              
                                correct_classifications += 1 
                          


                        for col_id in true_labels[row_id]["generative"]:

                            if col_id in assignment_cols:
                                predicted_classes.append(classifications[row_id,col_id])
                                true_classes.append(0)
                                classified += 1 

                              
                                
                            else:
                                continue

                            if classifications[row_id,col_id] == 0:
                                
                                correct_classifications += 1 
                
                metrics.append(("macro_f1",f1_score(true_classes,predicted_classes,average="macro")))
                metrics.append(("tp_accuracy",correct_classifications/classified if classified else 0))

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
