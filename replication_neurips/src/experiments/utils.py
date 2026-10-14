# --- Evaluation utilities ---
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np


def evaluate_sample_level(scores: np.ndarray, labels: dict, strategy='max'):
    
    y_true = np.zeros(scores.shape[0])
    for idx in labels:
        # if labels[idx]['measurement'] or labels[idx]['generative']:
            y_true[idx] = 1
    y_scores = scores.max(axis=1) if strategy == 'max' else scores.sum(axis=1)
    #print(y_scores.shape)
   
    auc = roc_auc_score(y_true, y_scores)
    ap = average_precision_score(y_true, y_scores)
    return auc, ap

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

def topk_recall_at_k(scores_row: np.ndarray, y_true_row: np.ndarray, k: int):
    """
    Recall@k for a single sample.
    """
    true_idx = np.where(y_true_row == 1)[0]
    if len(true_idx) == 0:
        return np.nan

    # k = min(k, scores_row.shape[0])
    topk_idx = np.argsort(scores_row)[-k:]

    print("True root causes",len(true_idx))

    print(f"Found ids: {len(set(topk_idx) & set(true_idx))}")
    return len(set(topk_idx) & set(true_idx)) / len(true_idx)


def topk_recall(scores_row: np.ndarray, y_true_row: np.ndarray):
    k = int(y_true_row.sum())
    if k == 0:
        return np.nan
    return topk_recall_at_k(scores_row, y_true_row, k)


def evaluate_feature_level(scores: np.ndarray, labels: dict, hidden:bool,dag):
    """
    Computes global AUC/AP across all samples,
    per-sample AUC/AP for samples containing anomalies,
    and AUC/AP restricted to the set of anomalous rows.
    
    Args:
        scores (np.ndarray): (n_samples, n_features) anomaly scores
        labels (dict): {row_id: {"measurement": set, "generative": set}}
    
    Returns:
        dict: {
            "global_auc": float,
            "global_ap": float,
            "per_sample_auc": float,
            "per_sample_ap": float,
            "anomalous_rows_auc": float,
            "anomalous_rows_ap": float
        }
    """
    y_true = np.zeros_like(scores)

    # Construct ground truth matrix
    for row_id in labels:
        if labels[row_id]['measurement'] or labels[row_id]['generative']:
            for col_id in list(labels[row_id]['measurement']) + list(labels[row_id]['generative']):
                

                # # No conditional outliers created by changing aroot node and propagating through
                # if ((hidden == True) and (col_id in list(labels[row_id]['generative']))) and (not list(dag.predecessors(col_id))) :
                #     continue

                y_true[row_id, col_id] = 1

    # meas_ap, gen_ap, meas_ap_mean, gen_ap_mean = evaluate_feature_ap_by_type(scores[list(labels.keys())],y_true[list(labels.keys())],labels)
    # meas_auc, gen_auc , meas_auc_mean, gen_auc_mean = evaluate_feature_auc_by_type(scores[list(labels.keys())],y_true[list(labels.keys())],labels)
    # meas_top_k_recall, gen_top_k_recall = evaluate_feature_topk_recall_by_type(scores[list(labels.keys())],y_true[list(labels.keys())],labels)

    # # Global metrics
    # global_auc = roc_auc_score(y_true.ravel(), scores.ravel())
    # global_ap = average_precision_score(y_true.ravel(), scores.ravel())

    # Per-sample metrics (averaged)
    per_sample_ap = 0
    per_sample_auc = 0
    count = 0

    top_k_recalls = []
    top_3_recalls = []
    top_5_recalls = []

    
    for row_id in labels:
        # Only compute if the row has at least one anomaly (otherwise metrics undefined)
        if y_true[row_id].sum() > 0:
            y_row = y_true[row_id]
            print(y_row)
            s_row = scores[row_id]
            r = topk_recall(s_row, y_row)
            top_k_recalls.append(r)

            top_3_recalls.append(topk_recall_at_k(s_row, y_row, k=3))
            top_5_recalls.append(topk_recall_at_k(s_row, y_row, k=5))
            auc = roc_auc_score(y_true[row_id], scores[row_id])
            ap = average_precision_score(y_true[row_id], scores[row_id])
            per_sample_auc += auc
            per_sample_ap += ap
            count += 1

    if top_k_recalls:
        avg_top_k_recall = np.nanmean(top_k_recalls)

    if top_3_recalls:
        avg_top_3_recall = np.nanmean(top_3_recalls)

    if top_5_recalls:
        avg_top_5_recall = np.nanmean(top_5_recalls)
    if count > 0:
        per_sample_ap /= count
        per_sample_auc /= count
    else:
        per_sample_ap, per_sample_auc = np.nan, np.nan

    # # Anomalous rows metrics (aggregate)
    # anomalous_rows = [row_id for row_id in labels if y_true[row_id].sum() > 0]
    # if anomalous_rows:
    #     y_true_anom = y_true[anomalous_rows].ravel()
    #     scores_anom = scores[anomalous_rows].ravel()
    #     anomalous_rows_auc = roc_auc_score(y_true_anom, scores_anom)
    #     anomalous_rows_ap = average_precision_score(y_true_anom, scores_anom)
    # else:
    #     anomalous_rows_auc, anomalous_rows_ap = np.nan, np.nan

    return {
      
        "total_auc": per_sample_auc,
        "total_ap": per_sample_ap,
        # "meas_ap":meas_ap_mean,
        # "gen_ap":gen_ap_mean,
        # "meas_auc":meas_auc_mean,
        # "gen_auc":gen_auc_mean,
        "top_k_recall": avg_top_k_recall if top_k_recalls else np.nan,
        "top_3_recall":avg_top_3_recall if top_3_recalls else np.nan,
        "top_5_recall":avg_top_5_recall if top_5_recalls else np.nan,
        # "meas_top_k_recall": meas_top_k_recall,
        # "gen_top_k_recall": gen_top_k_recall
    }




from collections import defaultdict

def evaluate_assignment(labels, assignment, dag,N,D):

    CLASSES = ["none", "generative", "measurement", "both"]

    # --------------------------
    # SAMPLE LEVEL
    # --------------------------
    true_positives = set(labels.keys())
    predicted_positives = set(assignment.keys())
    true_positives_predicted = true_positives & predicted_positives

    sample_precision = len(true_positives_predicted) / len(predicted_positives) if predicted_positives else 0   
    sample_recall = len(true_positives_predicted) / len(true_positives) if true_positives else 0
    sample_f1_score = 2 * (sample_precision * sample_recall) / (sample_precision + sample_recall) if (sample_precision + sample_recall) > 0 else 0

    # --------------------------
    # FEATURE LEVEL
    # --------------------------
    true_positives = set()
    for row_id in labels:
        if labels[row_id]['measurement'] or labels[row_id]['generative']:
            for col_id in list(labels[row_id]['measurement']) + list(labels[row_id]['generative']):
                true_positives.add((row_id, col_id))

    predicted_positives = set()
    for row_id in assignment:
        if assignment[row_id]['measurement_corrections'] or assignment[row_id]['generative_attributions'] or assignment[row_id]["uncertain_origin"]:
            for col_id in list(assignment[row_id]['measurement_corrections'].keys()) + \
                           list(assignment[row_id]['generative_attributions'].keys()) + \
                           list(assignment[row_id]["uncertain_origin"].keys()):
                predicted_positives.add((row_id, col_id))

    true_positives_predicted = true_positives & predicted_positives
    feature_precision = len(true_positives_predicted) / len(predicted_positives) if predicted_positives else 0   
    feature_recall = len(true_positives_predicted) / len(true_positives) if true_positives else 0
    feature_f1_score = 2 * (feature_precision * feature_recall) / (feature_precision + feature_recall) if (feature_precision + feature_recall) > 0 else 0

    # --------------------------
    # CLASSIFICATION ACCURACY (TRUE POSITIVES ONLY)
    # --------------------------
    true_meas_errors = set()
    true_gen_errors = set()

    for idx in labels:
        for child in labels[idx]['measurement']:
            if len(list(dag.successors(child))) > 0:  # Only non-sink nodes
                true_meas_errors.add((idx, child))
        for child in labels[idx]['generative']:
            if len(list(dag.successors(child))) > 0:
                true_gen_errors.add((idx, child))

    all_errors = true_gen_errors | true_meas_errors
    evaluated = []

    for idx in assignment:
        for child, attr in assignment[idx].get("measurement_corrections", {}).items():
            if (idx, child) not in all_errors:
                continue
            true_type = "measurement" if (idx, child) in true_meas_errors else (
                        "generative" if (idx, child) in true_gen_errors else "none")
            pred_type = "measurement"
            confidence = attr.get("confidence", 0)
            evaluated.append({"correct": true_type == pred_type, "confidence": confidence})

        for child, attr in assignment[idx].get("generative_attributions", {}).items():
            if (idx, child) not in all_errors:
                continue
            true_type = "generative" if (idx, child) in true_gen_errors else (
                        "measurement" if (idx, child) in true_meas_errors else "none")
            pred_type = "generative"
            confidence = attr.get("confidence", 0)
            evaluated.append({"correct": true_type == pred_type, "confidence": confidence})

    accuracy = sum([e["correct"] for e in evaluated]) / len(evaluated) if evaluated else 0

    # --------------------------
    # FULL CONFUSION MATRIX (NON-SINK NODES ONLY)
    # --------------------------
    def true_class(idx, child):
        if idx not in labels:
            return "none"

        is_meas = child in labels[idx]["measurement"]
        is_gen  = child in labels[idx]["generative"]
        if is_meas and is_gen:
            return "both"
        elif is_meas:
            return "measurement"
        elif is_gen:
            return "generative"
        else:
            return "none"

    def predicted_class(idx, child):
        meas = child in assignment.get(idx, {}).get("measurement_corrections", {})
        gen  = child in assignment.get(idx, {}).get("generative_attributions", {})
        both = child in assignment.get(idx, {}).get("uncertain_origin", {})
        if both or (meas and gen):
            return "both"
        elif meas:
            return "measurement"
        elif gen:
            return "generative"
        else:
            return "none"

    D_LIST = [col for col in range(D) if len(list(dag.successors(col))) > 0 ]
    all_pairs = set((n,d) for n in range(N) for d in D_LIST)
    for idx in labels:
        for child in set(labels[idx]["measurement"]) | set(labels[idx]["generative"]):
            if len(list(dag.successors(child))) > 0:  # Only non-sink nodes
                all_pairs.add((idx, child))
    for idx in assignment:
        for child in set(assignment[idx].get("measurement_corrections", {})) | \
                     set(assignment[idx].get("generative_attributions", {})) | \
                     set(assignment[idx].get("uncertain_origin", {})):
            if len(list(dag.successors(child))) > 0:
                all_pairs.add((idx, child))

    confusion = {true: {pred: 0 for pred in CLASSES} for true in CLASSES}
    for idx, child in all_pairs:
        t = true_class(idx, child)
        p = predicted_class(idx, child)
        confusion[t][p] += 1

    return sample_f1_score, sample_precision, sample_recall, \
           feature_f1_score, feature_precision, feature_recall, \
           accuracy, evaluated, confusion


        
    

def evaluate_feature_localization(locs: dict, labels: dict):
    precision_list, recall_list = [], []
    for i in labels:
        if i in locs:
            true_feats = set(labels[i].get('measurement', []) + labels[i].get('generative', []))
            pred_feats = locs[i]
            tp = len(pred_feats & true_feats)
            precision = tp / len(pred_feats) if pred_feats else 0
            recall = tp / len(true_feats) if true_feats else 0
            precision_list.append(precision)
            recall_list.append(recall)
    return {
        "precision": np.mean(precision_list),
        "recall": np.mean(recall_list)
    }

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

def evaluate_feature_topk_recall_by_type(y_score, y_true, labels: dict):
    """
    Computes mean Top-k Recall for measurement and generative outliers,
    where k is set per-sample to the true number of outliers of that type.
    """

    label_reverse_map = {v: k for k, v in enumerate(labels.keys())}

    idx_meas, idx_gen = [], []

    for idx in labels:
        if labels[idx]['measurement'] and not labels[idx]['generative']:
            idx_meas.append(label_reverse_map[idx])
        elif labels[idx]['generative'] and not labels[idx]['measurement']:
            idx_gen.append(label_reverse_map[idx])

    def mean_topk(scores, truth):
        recalls = []
        for i in range(scores.shape[0]):
            k = int(truth[i].sum())
            if k > 0:
                topk_idx = np.argsort(scores[i])[-k:]
                recalls.append(truth[i, topk_idx].sum() / k)
        return np.mean(recalls) if recalls else 0

    meas_topk = mean_topk(y_score[idx_meas], y_true[idx_meas]) if idx_meas else 0
    gen_topk  = mean_topk(y_score[idx_gen],  y_true[idx_gen])  if idx_gen  else 0

    return meas_topk, gen_topk


def evaluate_feature_ap_by_type(y_score, y_true, labels: dict):
    label_reverse_map = {v: k for k, v in enumerate(labels.keys())}   
    # Split scores by type 
    idx_meas = []
    idx_gen = []

    for idx in labels:
        if labels[idx]['measurement'] and not labels[idx]['generative']:
            idx_meas.append(label_reverse_map[idx])
        elif labels[idx]['generative'] and not labels[idx]['measurement']:
            idx_gen.append(label_reverse_map[idx])

    y_meas = y_score[idx_meas]
    y_gen = y_score[idx_gen]
    y_true_meas = y_true[idx_meas]
    y_true_gen = y_true[idx_gen]

    if np.any(np.isnan(y_meas)):
        raise ValueError("NAN value in measurement scores")

    # Global (all samples combined)
    meas_ap = average_precision_score(y_true_meas.ravel(), y_meas.ravel()) if len(idx_meas) > 0 else 0
    gen_ap  = average_precision_score(y_true_gen.ravel(), y_gen.ravel()) if len(idx_gen) > 0 else 0

    # Per-sample mean AP
    meas_ap_sample = np.mean([
        average_precision_score(y_true_meas[:, i], y_meas[:, i])
        for i in range(y_meas.shape[1])
        if len(np.unique(y_true_meas[:, i])) > 1  # need both classes present
    ]) if len(idx_meas) > 0 else 0

    gen_ap_sample = np.mean([
        average_precision_score(y_true_gen[:, i], y_gen[:, i])
        for i in range(y_gen.shape[1])
        if len(np.unique(y_true_gen[:, i])) > 1
    ]) if len(idx_gen) > 0 else 0

    return meas_ap, gen_ap, meas_ap_sample, gen_ap_sample


def evaluate_feature_auc_by_type(y_score, y_true, labels: dict):
    label_reverse_map = {v: k for k, v in enumerate(labels.keys())}   
    # Split scores by type 
    idx_meas = []
    idx_gen = []

    for idx in labels:
        if labels[idx]['measurement'] and not labels[idx]['generative']:
            idx_meas.append(label_reverse_map[idx])
        elif labels[idx]['generative'] and not labels[idx]['measurement']:
            idx_gen.append(label_reverse_map[idx])

    y_meas = y_score[idx_meas]
    y_gen = y_score[idx_gen]
    y_true_meas = y_true[idx_meas]
    y_true_gen = y_true[idx_gen]

    # Global
    meas_auc = roc_auc_score(y_true_meas.ravel(), y_meas.ravel()) if len(idx_meas) > 0 else 0
    gen_auc  = roc_auc_score(y_true_gen.ravel(), y_gen.ravel()) if len(idx_gen) > 0 else 0

    # Per-sample mean AUC
    meas_auc_sample = np.mean([
        roc_auc_score(y_true_meas[:, i], y_meas[:, i])
        for i in range(y_meas.shape[1])
        if len(np.unique(y_true_meas[:, i])) > 1
    ]) if len(idx_meas) > 0 else 0

    gen_auc_sample = np.mean([
        roc_auc_score(y_true_gen[:, i], y_gen[:, i])
        for i in range(y_gen.shape[1])
        if len(np.unique(y_true_gen[:, i])) > 1
    ]) if len(idx_gen) > 0 else 0

    return meas_auc, gen_auc, meas_auc_sample, gen_auc_sample




def evaluate_feature_localization_by_type(locs: dict, labels: dict):
    meas_precision_list, meas_recall_list = [], []
    gen_precision_list, gen_recall_list = [], []
    for i in labels:
        if not i in locs:
            continue
        true_meas_feats = set(labels[i].get('measurement', []))
        true_gen_feats = set(labels[i].get('generative', []))
        if not ((true_meas_feats and not true_gen_feats) or (not true_meas_feats and true_gen_feats)):
            continue
        pred_feats = locs[i]
        if true_meas_feats:
            tp = len(pred_feats & true_meas_feats)
            meas_precision_list.append(tp / len(pred_feats) if pred_feats else 0)
            meas_recall_list.append(tp / len(true_meas_feats))
        elif true_gen_feats:
            tp = len(pred_feats & true_gen_feats)
            gen_precision_list.append(tp / len(pred_feats) if pred_feats else 0)
            gen_recall_list.append(tp / len(true_gen_feats))
    return {
        "measurement_precision": np.mean(meas_precision_list) if meas_precision_list else None,
        "measurement_recall": np.mean(meas_recall_list) if meas_recall_list else None,
        "generative_precision": np.mean(gen_precision_list) if gen_precision_list else None,
        "generative_recall": np.mean(gen_recall_list) if gen_recall_list else None,
    }

def evaluate_reconstruction(clean, recon):
    from sklearn.metrics import mean_squared_error
    rmse_global = np.sqrt(mean_squared_error(clean, recon))
    return rmse_global, None

# def evaluate_by_anomaly_type(clean, recon, labels):
#     from sklearn.metrics import mean_squared_error
#     meas_idxs = [i for i, d in labels.items() if d['measurement'] and not d['generative']]
#     gen_idxs = [i for i, d in labels.items() if d['generative'] and not d['measurement']]
#     return {
#         #"measurement_only_rmse": np.sqrt(mean_squared_error(clean[meas_idxs], recon[meas_idxs])) if meas_idxs else None,
#        # "generative_only_rmse": np.sqrt(mean_squared_error(clean[gen_idxs], recon[gen_idxs])) if gen_idxs else None,
#         #"measurement_only_count": len(meas_idxs),
#         #"generative_only_count": len(gen_idxs)
#     }

# --- General Evaluation Wrappers ---
def evaluate_unsupervised_method(method, corrupt, labels):
    scores, _, _ = method(corrupt)
    auc, ap = evaluate_sample_level(scores[:, None], labels)
    return {"auc": auc, "ap": ap}

def run_ae_feature(method,corrupt, labels):
    scores, _, locs = method(corrupt,labels)
    sample_scores = scores.max(axis=1)

    return {"sample_score":sample_scores, "feature_score":scores[list(labels.keys())]}
def run_unsupervised_method(method, corrupt):
    scores, _, _ = method(corrupt)
  
    return {"sample_score":scores}

def evaluate_ae_method(method, corrupt, labels):
    scores, _, locs = method(corrupt,labels)
    auc, ap = evaluate_sample_level(scores, labels)
    loc = evaluate_feature_localization(locs, labels)
    loc_by_type = evaluate_feature_localization_by_type(locs, labels)
    return {"auc": auc, "ap": ap,"precision": loc["precision"], "recall": loc["recall"], **loc_by_type}

def run_explainer(method,corrupt,labels,dag):
    scores = method(corrupt,labels,dag)
    print(scores.shape)
    print(scores.dtype)
    return {"feature_score":scores}

def evaluate_reconstruction_method(method, corrupt, clean, labels):
    _, recon, locs = method(corrupt, labels)
    loc = evaluate_feature_localization(locs, labels)
    loc_by_type = evaluate_feature_localization_by_type(locs, labels)
    #rmse, _ = evaluate_reconstruction(clean, recon)
    #split = evaluate_by_anomaly_type(clean, recon, labels)
    return {"precision": loc["precision"], "recall": loc["recall"], **loc_by_type}# "rmse": rmse, **split}

def evaluate_localization_only_method(method, corrupt, labels, dag):
    try:
        locs = method(corrupt, dag, labels)
    except Exception:
        return {}
    loc = evaluate_feature_localization(locs, labels)
    loc_by_type = evaluate_feature_localization_by_type(locs, labels)
    return {"precision": loc["precision"], "recall": loc["recall"], **loc_by_type}