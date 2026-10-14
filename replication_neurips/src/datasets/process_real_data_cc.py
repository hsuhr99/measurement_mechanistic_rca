	
import causalchamber
from causalchamber.utils import graph_to_tikz
from causalchamber.ground_truth import latex_name

import numpy as np
import networkx as nx
from causalchamber.datasets import Dataset
import os 
import pickle
import random



def adj_to_nx(data:np.array,names=None):

    nx_dag = nx.DiGraph()
    row_ids, col_ids = np.nonzero(data)

    for i,j in zip(row_ids,col_ids):

        if names:
            e_i = names[i]
            e_j = names[j]

        else:
            e_i = i 
            e_j = j 
        
        for node in [e_i,e_j]:
            if node not in nx_dag.nodes:
                nx_dag.add_node(node)
        nx_dag.add_edge(e_i,e_j)
    return nx_dag        

def process_real_data(DOWNLOAD_DIR,DATA_DIR,SAMPLE_ANOM_RATIO,N_RUNS,MEAN_SHIFT,dataset_name,seed=None,outlier_type="both",N_SAMPLES=2000,strength="strong"):
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    
    os.makedirs(DOWNLOAD_DIR,exist_ok=True)
    dataset = Dataset(name="lt_interventions_standard_v1", root=DOWNLOAD_DIR)

    
    variables = ['red', 'green', 'blue', 'current', 'ir_1', 'ir_2', 'ir_3', 'vis_1', 'vis_2', 'vis_3', 'pol_1', 'pol_2', 'angle_1', 'angle_2', 'l_11', 'l_12', 'l_21', 'l_22', 'l_31', 'l_32']


    # Ground truth graph
    true_dag = causalchamber.ground_truth.graph('lt', 'standard').loc[variables, variables].values

    nx_dag = adj_to_nx(true_dag)
   
    strong_intervetions = {

        'uniform_red_strong': 'red',
        'uniform_green_strong': 'green',
        'uniform_blue_strong': 'blue',
        'uniform_t_ir_1_strong': 'ir_1',
        'uniform_t_ir_2_strong': 'ir_2',
        'uniform_t_ir_3_strong': 'ir_3',
        'uniform_t_vis_1_strong': 'vis_1',
        'uniform_t_vis_2_strong': 'vis_2',
        'uniform_t_vis_3_strong': 'vis_3',
        'uniform_pol_1_strong': 'pol_1',
        'uniform_pol_2_strong': 'pol_2',
        'uniform_v_angle_1_strong': 'angle_1',
        'uniform_v_angle_2_strong': 'angle_2',
        'uniform_v_c_strong':'current',
     
    }
        
    
    mid_interventions = {
        
        'uniform_red_mid': 'red',
        'uniform_green_mid': 'green',
        'uniform_blue_mid': 'blue',
        'uniform_t_ir_1_mid': 'ir_1',
        'uniform_t_ir_2_mid': 'ir_2',
        'uniform_t_ir_3_mid': 'ir_3',
        'uniform_t_vis_1_mid': 'vis_1',
        'uniform_t_vis_2_mid': 'vis_2',
        'uniform_t_vis_3_mid': 'vis_3',
        'uniform_pol_1_mid': 'pol_1',
        'uniform_pol_2_mid': 'pol_2',
        'uniform_v_angle_1_mid': 'angle_1',
        'uniform_v_angle_2_mid': 'angle_2',
        'uniform_v_c_mid':'current'
    }

    interventions_to_variables = mid_interventions if strength == "mid" else strong_intervetions


    experiments = list(interventions_to_variables.keys())


    variables_to_interventions = {v:k for k,v in interventions_to_variables.items()}
    observational_data = dataset.get_experiment("uniform_reference").as_pandas_dataframe()[variables].sample(N_SAMPLES).values



    observational_data = observational_data#[:,[i for i in range(len(variables)) if i not in drop_col_indices]]


    interventional_data = {e:dataset.get_experiment(e).as_pandas_dataframe()[variables].values for e in experiments}

    for interv,data in interventional_data.items():
        interventional_data[interv] = data[:,[i for i in range(len(variables))]] #if i not in drop_col_indices]]

    # variables = [v for v in variables if v not in drop_cols]

    idx_to_vars = variables
    vars_to_idx = {v:i for i,v in enumerate(idx_to_vars)}
    n_obs, d_obs = observational_data.shape 
    per_intervention_count = int((n_obs /len(interventions_to_variables)) *(SAMPLE_ANOM_RATIO/(1- SAMPLE_ANOM_RATIO))) 



    for run in range(N_RUNS):
        rng = np.random.default_rng(seed=run)
    
        data = observational_data.copy()
        clean_data_mins = data.min(axis=0)
        clean_data_maxs = data.max(axis=0)
        config = {}
        labels = {}
    
        if outlier_type in ["both","mech"]:
            for intervention, interv_data in interventional_data.items():

                print(intervention)

                if not (strength in intervention):
                    continue

                # randomsample from interventional_data 
                n_interv = interv_data.shape[0]
                interventional_ids = rng.choice(n_interv, size=per_intervention_count, replace=False)

                old_data_shape = data.shape[0]
                print(data.shape)
                print(interv_data.shape)
                data = np.concatenate([data,interv_data[interventional_ids]],axis=0)

                

                for row_id in range(old_data_shape,old_data_shape + len(interventional_ids)):
                    if row_id not in labels:
                        labels[row_id] = {"measurement":set(),"generative":set()}

                    labels[row_id]["generative"].add(vars_to_idx[interventions_to_variables[intervention]])


        # create synthetic outliers in observational data with mean shifted uniform
        if outlier_type in ["both","meas"]:
            for intervention, interv_data in interventional_data.items():

                if not (strength in intervention):
                    continue
                n_interv = interv_data.shape[0]
                # Select random indices
                interventional_ids = rng.choice(n_interv, size=per_intervention_count, replace=False)
                meas_ids = rng.choice(data.shape[0], size=per_intervention_count, replace=False)
                col_id = vars_to_idx[interventions_to_variables[intervention]]

                for i,row_id in enumerate(meas_ids):
                    
                    # Only select column that received intervention
                    data[row_id,col_id] = interv_data[interventional_ids[i],col_id]
                    
                    if row_id not in labels:
                        labels[row_id] = {"measurement":set(),"generative":set()}

                    labels[row_id]["measurement"].add(col_id)

        config["labels"] = labels
        config["data"] = data
        config["variables"] = variables
        config["dag"] = nx_dag
        config["interventions"] = interventions_to_variables
        config["n_outliers"] = per_intervention_count
        config["mean_shift"] = MEAN_SHIFT


        with open(f"{DATA_DIR}/{dataset_name}_{run}.pkl","wb") as file:
            pickle.dump(config,file,pickle.HIGHEST_PROTOCOL)



        # for every intervention sample












