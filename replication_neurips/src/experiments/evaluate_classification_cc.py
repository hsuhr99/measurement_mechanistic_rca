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
from ..cali.cali import cali

from scipy.stats import spearmanr, weightedtau

def process_single_run(name, function, config_file_name, data_dir_path):
    """Runs a single (method, config_file) pair and returns results."""
    import pickle  # Needed inside subprocess
    import time

    data_file_path = f"{os.getcwd()}/{data_dir_path}/{config_file_name}"
    with open(f"{os.getcwd()}/{data_dir_path}/{config_file_name}", "rb") as file:
        total_data = pickle.load(file)

    corrupt_data = total_data["data"]
    dag = total_data["dag"]
    labels = total_data["labels"]

    params = {}
    t0 = time.perf_counter()
    params["dag"] = dag
    params["data"] = corrupt_data 
    
    if not "LEARN" in name:
        
        assignment, feature_scores, classifications, confidences = cali(**params)
    
    else:
        
        dag = dag_search_ae_anomalies(corrupt_data,nogam_wrapper)
        params["dag"] = dag
        assignment, feature_scores, classifications, confidences = cali(**params)

    if "BASELINE" in name:
        
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
        
        params["root_causes"] = root_causes 
        params["anomaly_mask"] = label_mask
        print(params.keys())
        classifications = function(**params)

    t1 = time.perf_counter()

    return [{
        "method": name,
        "config_file": config_file_name,
        "data_file_name": data_file_path,
        "classifications": classifications,
        "assignment": assignment,
        "feature_scores": feature_scores,
        "confidences":confidences,
        "run_time": t1 - t0,
    }]
def _classification_worker(args):
    name, function, config_file_name, data_dir_path = args
    print(f"[INFO] Running {name} on {config_file_name} (parallel mode)")
    return process_single_run(name, function, config_file_name, data_dir_path)


import os
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed

def run_classification(methods, data_dir_path, dataset_name, result_dir):
    max_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
    os.makedirs(result_dir, exist_ok=True)
    result_file_path = f"{result_dir}/result.pkl"

    # Load existing results
    if os.path.exists(result_file_path):
        with open(result_file_path, "rb") as file:
            results = pickle.load(file)
    else:
        results = []

    # Already computed pairs
    existing_pairs = {(r["method"], r["config_file"]) for r in results}

    # Get matching files
    matching_config_files = [
        f for f in os.listdir(os.path.join(os.getcwd(), data_dir_path))
        if dataset_name in f and f.endswith(".pkl")
    ]
    print(f"Matching config files: {matching_config_files}")

    # --------------------------------------------------
    # Build task list
    # --------------------------------------------------
    tasks = []

    for name, function in methods.items():
        for config_file_name in matching_config_files:
            if (name, config_file_name) in existing_pairs:
                continue
            if ("CALI" in name) and ((f"{name}_max", config_file_name) in existing_pairs):
                continue

            tasks.append((name, function, config_file_name, data_dir_path))

    # --------------------------------------------------
    # Parallel execution
    # --------------------------------------------------
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {
            executor.submit(_classification_worker, task): task
            for task in tasks
        }

        for future in as_completed(future_to_task):
            try:
                new_results = future.result()   # ← list of results
                results.extend(new_results)

                # Incremental save (safe)
                with open(result_file_path, "wb") as file:
                    pickle.dump(results, file, pickle.HIGHEST_PROTOCOL)

            except Exception as e:
                name, _, config_file_name, _ = future_to_task[future]
                print(f"[ERROR] {name} on {config_file_name} failed: {e}")

    return results



def evaluate_classification(result_dir,plot_dir,hidden=False):

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
            data = total_data["data"]
            true_dag = total_data["dag"]

         
            assignment = run["assignment"]
            feature_scores = run["feature_scores"]
            classifications = run["classifications"]
            confidences = run["confidences"]
            
          

            # # Evaluate
            # max_sample_eval_auc, max_sample_eval_ap = evaluate_sample_level(feature_scores, true_labels, "max")
            # mean_sample_eval_auc, mean_sample_eval_ap = evaluate_sample_level(feature_scores, true_labels, "mean")
            # fl = evaluate_feature_level(feature_scores, true_labels,hidden=hidden,dag=true_dag)
            
            # feature_auc = fl["global_auc"]
            # feature_ap = fl["global_ap"]
            
            # per_sample_feature_auc = fl["per_sample_auc"]
            # per_sample_feature_ap = fl["per_sample_ap"]
      
            
            # metrics = [
            #    ("max_sample_auc", max_sample_eval_auc),
            #     ("max_sample_ap", max_sample_eval_ap),
            #     ("mean_sample_auc", mean_sample_eval_auc),
            #     ("mean_sample_ap", mean_sample_eval_ap),
            #     ("feature_auc", feature_auc),
            #     ("feature_ap", feature_ap),
            #     ("per_sample_feature_auc",per_sample_feature_auc),
            #     ("per_sample_feature_ap",per_sample_feature_ap)
            # ]
             # ensure defined
            metrics = []
            
            classified = 0
            correct_classifications = 0
            
            if assignment is not None:


                for row_id in assignment:

                    if row_id not in true_labels:
                        continue 

                    assignment_cols = list(assignment[row_id]["generative_attributions"].keys()) + list(assignment[row_id]["measurement_corrections"].keys())

                    for col_id in true_labels[row_id]["measurement"]:


                        if col_id in assignment_cols:
                            classified += 1 
                        else:
                            continue

                        if classifications[row_id,col_id] ==1:
                            correct_classifications += 1 


                    for col_id in true_labels[row_id]["generative"]:

                        if col_id in assignment_cols:
                            classified += 1 

                            
                        else:
                            continue

                        if classifications[row_id,col_id] ==0:
                            correct_classifications += 1 
                
                
                metrics.append(("tp_accuracy",correct_classifications/classified if classified else 0))
                


                sample_f1_score, sample_precision, sample_recall, feature_f1_score, feature_precision, feature_recall, accuracy, evaluated_run, confusion = evaluate_assignment(true_labels, assignment,true_dag,data.shape[0],data.shape[1])
                classification_confidences += evaluated
                #evaluated.extend(evaluated_run)
                CLASSES = ["none", "generative", "measurement", "both"]

                precision_c = {}
                recall_c = {}
                f1_c = {}
                present_classes = []

                for c in CLASSES:
                    TP = confusion[c][c]
                    FP = sum(confusion[true][c] for true in CLASSES) - TP
                    FN = sum(confusion[c][pred] for pred in CLASSES) - TP

                    precision_c[c] = TP / (TP + FP) if (TP + FP) > 0 else 0
                    recall_c[c]    = TP / (TP + FN) if (TP + FN) > 0 else 0
                    f1_c[c]        = (
                        2 * precision_c[c] * recall_c[c] / (precision_c[c] + recall_c[c])
                        if (precision_c[c] + recall_c[c]) > 0 else 0
                    )

                    # Class occurs in ground truth?
                    if TP + FN > 0:
                        present_classes.append(c)

                # Safe macro averages (only over present classes)
                if present_classes:
                    macro_precision = sum(precision_c[c] for c in present_classes) / len(present_classes)
                    macro_recall    = sum(recall_c[c]    for c in present_classes) / len(present_classes)
                    macro_f1        = sum(f1_c[c]        for c in present_classes) / len(present_classes)
                else:
                    macro_precision = macro_recall = macro_f1 = 0


                # ------------------------
                # Append all metrics
                # ------------------------
                metrics.extend([
                    # ("tp_accuracy", accuracy),
                    ("macro_precision", macro_precision),
                    ("macro_recall", macro_recall),
                    ("macro_f1", macro_f1),
                    ])

                # Add per-class metrics
                for c in CLASSES:
                    metrics.extend([
                        (f"{c}_precision", precision_c[c]),
                        (f"{c}_recall", recall_c[c]),
                        (f"{c}_f1", f1_c[c]),
                    ])




               
            else:
                sample_f1_score = sample_precision = sample_recall = None
                feature_f1_score = feature_precision = feature_recall = None
                accuracy = None
            
            # Calculate full meas/ gen accuracy:
            counter = 0
            accuracy = 0
            confs_total = []
            confs_tp = []
            
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
       

      
        # Only plot if evaluated is not empty
      
#         if evaluated:
#             # Sort by confidence descending
#             evaluated_sorted = sorted(evaluated, key=lambda x: -x["confidence"])

#             bin_size = 1  # adjust as needed
#             binned_accuracies = []
#             binned_confidences = []

#             for i in range(0, len(evaluated_sorted), bin_size):
#                 bin_slice = evaluated_sorted[i:i+bin_size]
#                 if not bin_slice:
#                     continue

#                 # Average confidence in this bin
#                 avg_conf = np.mean([e["confidence"][0] if isinstance(e["confidence"], np.ndarray) and e["confidence"].size else e["confidence"]
#                                     for e in bin_slice])
#                 binned_confidences.append(avg_conf)

#                 # Accuracy in this bin
#                 bin_acc = sum(e["correct"] for e in bin_slice) / len(bin_slice)
#                 binned_accuracies.append(bin_acc)
        
# # Now binned_confidences and binned_accuracies can be used for plotting
# # e.g., plt.plot(binned_confidences, binned_accuracies)

#             print("Smallest confidence, ",min(binned_confidences))
#             plt.plot(binned_confidences, binned_accuracies)
#             plt.xlabel("Top-k Predictions (by gain)")
#             plt.ylabel("Cumulative Accuracy")
#             plt.title(f"Accuracy vs. Number of Predictions (Run {run['config_file']}, {method})")
#             plt.grid(True)
#             plt.savefig(f"{plot_dir}/{method}_conf_acc.png")
#             plt.close()

        # classification_results.append({"method":method,"confidences":binned_confidences,"accuracies":binned_accuracies})

    # Save evaluation results
    # classification_result_filepath = f"{result_dir}/class_result.pkl"
    # print(classification_results)
    # with open(classification_result_filepath,"wb") as file:
    #     pickle.dump(classification_results,file,pickle.HIGHEST_PROTOCOL)

    eval_result_df = pd.DataFrame(evaluation_results)
    eval_result_df.to_csv(eval_result_file_path, index=False)

    return

       # data_file_path = 
       

        # Evaluate score based sample level and feature level AP, and AUC
        # Evaluate Assignment precision, recall F1 on sample and feature level if assignment is provided
        # Evaluate Classification Accuracy of all true positves on feature level 

