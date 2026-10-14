import pandas as pd
import numpy as np
import random
# from src.cali.robust_dag_scoring import score_dag
import networkx as nx
import cdt
# from src.competitors.causal_discovery import dag_search_ae_anomalies, cam_wrapper, topic_wrapper
# from src.competitors.feature_attribution import shap_feature_explainer_per_row
# import os
# from src.plots.utils import plot_from_dataframe
import matplotlib.pyplot as plt
import pickle
# -----------------import pickle------------
# Data & DAG prep
# -----------------------------

# # -----------------------------
# # Params
# # -----------------------------
# n_real_samples = 10
# n_runs = 5

# n_outliers = 5

# print(f"MAIN DATA SHAPE {sachs_main.shape}")

# # -----------------------------
# # Main loop
# # -----------------------------
# all_runs = []

def generate_dataset(data_dir,seed:int,n_outliers):
    sachs_main = pd.read_csv(f"{data_dir}/data/cd3cd28.csv")
    columns = list(sachs_main.columns)
    d = len(columns)

    INTERVENTION_TARGETS = {
        'cd3cd28_aktinhib.csv': 'Akt',
        'cd3cd28_g0076.csv': 'PKC',
        'cd3cd28_psitect.csv': 'P38',
        'cd3cd28_u0126.csv': 'Mek',
        'cd3cd28_ly.csv': 'PIP3',
        'pma.csv': 'PKC',
        'b2camp.csv': 'PKA',
    }

    true_dag = nx.DiGraph()
    node_mapping = {
        'PIP2': 'PIP2','PKC': 'PKC','plcg': 'Plcg','PIP3': 'PIP3',
        'pjnk': 'Jnk','P38': 'P38','PKA': 'PKA','praf': 'Raf',
        'pmek': 'Mek','p44/42': 'Erk','pakts473': 'Akt'
    }
    _, cdt_dag = cdt.data.load_dataset("sachs")

    # Remove to make acyclic
    cdt_dag.remove_edge('PIP3', 'plcg')
    
    for i,_ in enumerate(cdt_dag.nodes):
        true_dag.add_node(i)
    
    for c,e in cdt_dag.edges:
        c_i = columns.index(node_mapping[c])
        e_j = columns.index(node_mapping[e])
        true_dag.add_edge(c_i, e_j)
    
    
    
    assert(nx.is_directed_acyclic_graph(true_dag))
    
    
    rng = np.random.default_rng(seed)

    data = sachs_main.copy()
    y_true_row = np.zeros(len(data), dtype=int)
    y_feat = np.zeros((len(data), d), dtype=int)
    y_class = np.zeros_like(y_feat)
    intervention_labels = ["none"] * len(data)
    
    # # Synthetic outliers
    # synth_rows = rng.choice(len(data), size=n_outliers*data.shape[1], replace=False)
    
    
    # for row_id in synth_rows:
        
    #     j = rng.integers(0,data.shape[1])
    #     col = data.columns[j]
    #     data.iloc[row_id, j] = rng.uniform(data[col].min(), data[col].max()) + mean_shift*data[col].std()

    #     y_feat[row_id, j] = 1
    #     y_class[row_id,j] = 1
    # y_true_row[synth_rows] = 1
   
    
    # for idx in synth_rows:
    #     intervention_labels[idx] = "synthetic"

    # Real interventions
    for target_file, target_col in INTERVENTION_TARGETS.items():
        interv_data = pd.read_csv(f"{data_dir}/data/{target_file}")
        n_real = min(n_outliers, len(interv_data))
        interv_indices = rng.choice(len(interv_data), size=n_real, replace=False)
        for idx in interv_indices:
            # base_idx = rng.integers(low=0, high=len(data))
            # new_row = data.iloc[base_idx].copy()
            # new_row[target_col] = interv_data[target_col].iloc[idx]
            data = pd.concat([data, interv_data.iloc[idx].to_frame().T], ignore_index=True)
            y_true_row = np.append(y_true_row, 1)
            new_feat_row = np.zeros(d, dtype=int)
            new_feat_row[columns.index(target_col)] = 1
            y_feat = np.vstack([y_feat, new_feat_row])
            y_class =  np.vstack([y_class, np.zeros_like(new_feat_row)])
            intervention_labels.append(target_col)

        #
        n_real = min(n_outliers, len(interv_data))
        interv_indices = rng.choice(len(interv_data), size=n_real, replace=False)
        data_indices = rng.choice(data.shape[0],size=n_real,replace=False)
        col_id = columns.index(target_col)

        for i, data_idx in enumerate(data_indices):
            # base_idx = rng.integers(low=0, high=len(data))
            # new_row = data.iloc[base_idx].copy()
            # new_row[target_col] = interv_data[target_col].iloc[idx]
            
            interv_idx = interv_indices[i]
            print(target_col)
            print(data.iloc[data_idx])
            data.iloc[data_idx,col_id] = np.array(interv_data.iloc[interv_idx].values)[col_id]
            print(data.iloc[data_idx])
            # raise ValueError("stop")
            # data = pd.concat([data, interv_data.iloc[idx].to_frame().T], ignore_index=True)
            y_true_row[data_idx] = 1
         
            y_feat[data_idx,col_id] = 1
            y_class[data_idx,col_id] = 1
            # intervention_labels.append(target_col)



        
    print(f"Data size: {data.shape}")
    print(F"Outlier count: {y_feat.sum()}")

    labels = {}

    rc_rows, rc_cols = np.nonzero(y_feat)


    for row_id,col_id in zip  (rc_rows,rc_cols):
        if not row_id in labels:
            labels[row_id] = {"generative":[],"measurement":[]}

        

        if y_class[row_id,col_id] == 1:
            labels[row_id]["measurement"].append(col_id)

        else:
            labels[row_id]["generative"].append(col_id)



    return data, labels, true_dag



def process_sachs_data(data_dir,seed,N_RUNS,N_OUTLIERS=5):
    
    
    
    
    for r in range(N_RUNS):

        data, labels, dag = generate_dataset(data_dir,r+seed,n_outliers=N_OUTLIERS)

        data_file = {
            "dag":dag,
            "data":data,
            "labels":labels
        }

        with open(f"{data_dir}/{r}.pkl","wb") as file:
            pickle.dump(data_file,file,protocol=pickle.HIGHEST_PROTOCOL)
