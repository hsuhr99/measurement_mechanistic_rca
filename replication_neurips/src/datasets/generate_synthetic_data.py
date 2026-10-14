
from .synthetic_dags import generate_data_with_anomalies, generate_experiment_fixed_root_causes
import pickle
import os
import random
import json
import networkx as nx


def _project_to_observed_variables(
    clean_data,
    corrupt_data,
    labels,
    dag,
    observed_dag=None,
    hidden_var_ratio: float = 0.0,
    hidden_var_count: int = 0,
):
    if hidden_var_ratio <= 0 and hidden_var_count <= 0:
        return {
            "clean_data": clean_data,
            "corrupt_data": corrupt_data,
            "labels": labels,
            "true_dag": dag,
            "observed_dag": observed_dag if observed_dag is not None else dag,
            "hidden_nodes": [],
            "observed_nodes": list(range(clean_data.shape[1])),
        }

    num_nodes = clean_data.shape[1]
    if hidden_var_count > 0:
        num_hidden = min(max(1, hidden_var_count), max(0, num_nodes - 2))
    else:
        num_hidden = min(max(1, int(round(num_nodes * hidden_var_ratio))), max(0, num_nodes - 2))
    hidden_nodes = sorted(random.sample(range(num_nodes), num_hidden))
    observed_nodes = [node for node in range(num_nodes) if node not in hidden_nodes]
    node_map = {old: new for new, old in enumerate(observed_nodes)}

    projected_labels = {}
    for row_id, row_labels in labels.items():
        measurement = [node_map[col] for col in row_labels["measurement"] if col in node_map]
        generative = [node_map[col] for col in row_labels["generative"] if col in node_map]
        if measurement or generative:
            projected_labels[row_id] = {
                "measurement": measurement,
                "generative": generative,
            }

    return {
        "clean_data": clean_data[:, observed_nodes],
        "corrupt_data": corrupt_data[:, observed_nodes],
        "labels": projected_labels,
        "true_dag": nx.relabel_nodes(dag.subgraph(observed_nodes).copy(), node_map),
        "observed_dag": nx.relabel_nodes((observed_dag if observed_dag is not None else dag).subgraph(observed_nodes).copy(), node_map),
        "hidden_nodes": hidden_nodes,
        "observed_nodes": observed_nodes,
    }


def generate_dataset(
        n:int,
        d:int,
        edge_density: float,
        outlier_setting,
        mean_shift:int,
        sample_ratio:float,
        seed:float,
        file_name:str,
        outlier_type:str,
        dag_misspecification_type: str = None,
        dag_misspecification_ratio: float = 0.0,
        soft_intervention_type: str = None,
        hidden_var_ratio: float = 0.0,
        hidden_var_count: int = 0,
        correlated_measurement_group_size: int = 1,
        heteroscedasticity_strength: float = 0.0
        
        ):
    
    clean_data, corrupt_data, labels, dag, functions ,corrupted_dag = generate_data_with_anomalies(
        num_samples=n,
        num_nodes=d,
        edge_prob=edge_density,
        sample_anom_ratio=sample_ratio,
        outlier_setting=  outlier_setting,
        mean_shift=mean_shift,
        seed=seed,
        outlier_type=outlier_type,
        dag_misspecification_type=dag_misspecification_type,
        dag_misspecification_ratio=dag_misspecification_ratio,
        soft_intervention_type=soft_intervention_type,
        correlated_measurement_group_size=correlated_measurement_group_size,
        heteroscedasticity_strength=heteroscedasticity_strength,
        )

    projected_data = _project_to_observed_variables(
        clean_data=clean_data,
        corrupt_data=corrupt_data,
        labels=labels,
        dag=dag,
        observed_dag=corrupted_dag if corrupted_dag is not None else dag,
        hidden_var_ratio=hidden_var_ratio,
        hidden_var_count=hidden_var_count,
    )
    
   
    
    total_data = {
        "clean_data": projected_data["clean_data"],
        "corrupt_data": projected_data["corrupt_data"],
        "dag": projected_data["true_dag"],
        "true_dag": projected_data["true_dag"],
        "observed_dag": projected_data["observed_dag"],
        "corrupt_dag": projected_data["observed_dag"],
        "latent_true_dag": dag,
        "latent_observed_dag": corrupted_dag if corrupted_dag is not None else dag,
        "labels": projected_data["labels"],
        "hidden_nodes": projected_data["hidden_nodes"],
        "observed_nodes": projected_data["observed_nodes"],
        "edge_prob":edge_density,
        "seed":seed,
    }

    with open(f"{file_name}_data.pkl","wb") as file:
        pickle.dump(total_data,file,pickle.HIGHEST_PROTOCOL)

    config = {
        "n":n,
        "d":d,
        "edge_prob":edge_density,
        "sample_ratio":sample_ratio,
        "outlier_setting":outlier_setting,
        "mean_shift":mean_shift,
        "seed":seed,
        "outlier_type":outlier_type,
        "dag_misspecification_type": dag_misspecification_type,
        "dag_misspecification_ratio": dag_misspecification_ratio,
        "soft_intervention_type": soft_intervention_type,
        "hidden_var_ratio": hidden_var_ratio,
        "hidden_var_count": hidden_var_count,
        "correlated_measurement_group_size": correlated_measurement_group_size,
        "heteroscedasticity_strength": heteroscedasticity_strength,
        "data_file_path":f"{os.getcwd()}/{file_name}_data.pkl"
    }

    with open(f"{file_name}_config.json","w") as file:
        json.dump(config,file)


# Create ablation data_sets

def generate_experiment_data(data_dir=None,experiment_name="",N_RUNS=1,seeds=[],config=None,specific=None):
    os.makedirs(data_dir,exist_ok=True)

    n = config["N"]
    d = config["D"]
    edge_density = config["EDGE_DENSITY"]
    sample_ratio = config["SAMPLE_RATIO"]
    mean_shift = config["MEAN_SHIFT"]
    outlier_type = config["OUTLIER_TYPE"]
    dag_misspecification_type = config.get("DAG_MISSPECIFICATION_TYPE")
    dag_misspecification_ratio = config.get("DAG_MISSPECIFICATION_RATIO", 0)
    soft_intervention_type = config.get("SOFT_INTERVENTION_TYPE")
    hidden_var_ratio = config.get("HIDDEN_VAR_RATIO", 0)
    hidden_var_count = config.get("HIDDEN_VAR_COUNT", 0)
    correlated_measurement_group_size = config.get("CORRELATED_MEASUREMENT_GROUP_SIZE", 1)
    heteroscedasticity_strength = config.get("HETEROSCEDASTICITY_STRENGTH", 0.0)

    if specific == "fixed_rc_count":
        
        ROOT_CAUSES = config["RC_COUNT"]

        for i in range(N_RUNS):
            if data_dir:
                file_name = f"{data_dir}/{experiment_name}_{i}"
            else:
                file_name = f"{experiment_name}_{i}"
            seed = seeds[i]
            clean_data, corrupt_data, labels, dag, functions ,corrupted_dag = generate_experiment_fixed_root_causes(
                num_samples=n,
                num_nodes=d,
                edge_prob=edge_density,
                sample_anom_ratio=sample_ratio,
                mean_shift=mean_shift,
                seed=seed,
                rc_count=ROOT_CAUSES,
                dag_misspecification_type=dag_misspecification_type,
                dag_misspecification_ratio=dag_misspecification_ratio,
                soft_intervention_type=soft_intervention_type,
                correlated_measurement_group_size=correlated_measurement_group_size,
                heteroscedasticity_strength=heteroscedasticity_strength,
            )

            projected_data = _project_to_observed_variables(
                clean_data=clean_data,
                corrupt_data=corrupt_data,
                labels=labels,
                dag=dag,
                observed_dag=corrupted_dag if corrupted_dag is not None else dag,
                hidden_var_ratio=hidden_var_ratio,
                hidden_var_count=hidden_var_count,
            )
    
   
    
            total_data = {
                "clean_data": projected_data["clean_data"],
                "corrupt_data": projected_data["corrupt_data"],
                "dag": projected_data["true_dag"],
                "true_dag": projected_data["true_dag"],
                "observed_dag": projected_data["observed_dag"],
                "corrupt_dag": projected_data["observed_dag"],
                "latent_true_dag": dag,
                "latent_observed_dag": corrupted_dag if corrupted_dag is not None else dag,
                "labels": projected_data["labels"],
                "hidden_nodes": projected_data["hidden_nodes"],
                "observed_nodes": projected_data["observed_nodes"],
                "edge_prob":edge_density,
                "seed":seed,
            }

            with open(f"{file_name}_data.pkl","wb") as file:
                pickle.dump(total_data,file,pickle.HIGHEST_PROTOCOL)

            config = {
                "n":n,
                "d":d,
                "edge_prob":edge_density,
                "sample_ratio":sample_ratio,
          
                "mean_shift":mean_shift,
          
                "seed":seed,
                "dag_misspecification_type": dag_misspecification_type,
                "dag_misspecification_ratio": dag_misspecification_ratio,
                "soft_intervention_type": soft_intervention_type,
                "hidden_var_ratio": hidden_var_ratio,
                "hidden_var_count": hidden_var_count,
                "correlated_measurement_group_size": correlated_measurement_group_size,
                "heteroscedasticity_strength": heteroscedasticity_strength,
             
                "data_file_path":f"{os.getcwd()}/{file_name}_data.pkl"
            }

            with open(f"{file_name}_config.json","w") as file:
                json.dump(config,file)

        return
   
    
    
    n = config["N"]
    d = config["D"]

    edge_density = config["EDGE_DENSITY"]
    sample_ratio = config["SAMPLE_RATIO"]
   

    outlier_setting = config["OUTLIER_SETTING"]
    mean_shift = config["MEAN_SHIFT"]
  



   
    for i in range(N_RUNS):

        if data_dir:
            file_name = f"{data_dir}/{experiment_name}_{i}"
        else:
            file_name = f"{experiment_name}_{i}"
        
        
        generate_dataset(
            n,
            d,
            edge_density=edge_density,
            outlier_setting=outlier_setting,
            mean_shift = mean_shift,
            sample_ratio=sample_ratio,
            seed=seeds[i],
            file_name=file_name,
            outlier_type=outlier_type,
            dag_misspecification_type=dag_misspecification_type,
            dag_misspecification_ratio=dag_misspecification_ratio,
            soft_intervention_type=soft_intervention_type,
            hidden_var_ratio=hidden_var_ratio,
            hidden_var_count=hidden_var_count,
            correlated_measurement_group_size=correlated_measurement_group_size,
            heteroscedasticity_strength=heteroscedasticity_strength,
        )



# generate_ablation()
