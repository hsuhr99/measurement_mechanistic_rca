import numpy as np
import networkx as nx
import random
from typing import Callable, Dict, List, Tuple

class NormalizedFunction:
    def __init__(self, raw_function: Callable, min_val: float, range_val: float):
        self.raw_function = raw_function
        self.min_val = min_val
        self.range_val = max(range_val, 1e-6)  # avoid division by zero

    def __call__(self, X: np.ndarray) -> np.ndarray:
        raw_output = self.raw_function(X)
        return (raw_output - self.min_val) / self.range_val

    def denormalize(self, norm_output: np.ndarray) -> np.ndarray:
        return norm_output * self.range_val + self.min_val

# ----- Function Definitions for DAG Generation -----
def generate_random_dag(num_nodes: int, edge_prob: float = 0.4, seed: int = None):
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    dag = nx.DiGraph()
    nodes = list(range(num_nodes))
    dag.add_nodes_from(nodes)

    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if np.random.rand() < edge_prob:
                dag.add_edge(i, j)

    assert nx.is_directed_acyclic_graph(dag), "Generated graph is not a DAG!"
    return dag


# ----- Functional Generators -----
def generate_polynomial_univ(l: int):
    a = np.array([random.choice([np.random.uniform(-1, -0.1), np.random.uniform(0.1, 1)]) for _ in range(l)])
    b = np.array([np.random.uniform(-3, 3) for _ in range(l)])
    exp = np.array([np.random.choice([2,3]) for _ in range(l)])
    
    def f(A):
        total = np.zeros_like(A)
        for i in range(l):
            term = a[i] * (A - b[i]) ** exp[i]
            total += term
        return total
    return f


def generate_linear_univ():
    a = random.choice([np.random.uniform(0.01, 1), np.random.uniform(1, 10)])
    b = np.random.uniform(-3, 3)
    return lambda A: (a * A + b)


def generate_polynomial(X: np.ndarray, l: int):
    funcs = [generate_polynomial_univ(l) for _ in range(X.shape[1])]
    return lambda A: (np.hstack([funcs[i](A[:,i]).reshape(-1,1) for i in range(X.shape[1])]).sum(axis=1))


def generate_linear(X: np.ndarray):
    funcs = [generate_linear_univ() for _ in range(X.shape[1])]
    return lambda A: np.hstack([funcs[i](A[:,i]).reshape(-1,1) for i in range(X.shape[1])]).sum(axis=1)



# def generate_root_node_func():
#     return lambda x: np.random.normal(scale=1.0, size=(x.shape[0],))

def generate_root_node_func(num_samples):
    noise_vector = np.random.normal(scale=1.0, size=num_samples)
    return lambda x: noise_vector


def _normalize_function(raw_f: Callable, parent_data: np.ndarray) -> NormalizedFunction:
    raw_vals = raw_f(parent_data)
    min_val, max_val = raw_vals.min(), raw_vals.max()
    raw_std = np.std(raw_vals)
    min_val -= 3 * raw_std
    max_val += 3 * raw_std
    range_val = max(max_val - min_val, 1e-6)
    return NormalizedFunction(raw_f, min_val, range_val)


def _compute_noise_scale(
    base_noise_scale: float,
    heteroscedasticity_strength: float,
    signal: np.ndarray,
    min_scale_ratio: float = 0.25,
) -> np.ndarray:
    if heteroscedasticity_strength <= 0:
        return np.full(signal.shape, base_noise_scale, dtype=float)

    centered_signal = np.abs(signal - np.mean(signal))
    signal_scale = np.std(signal)
    if signal_scale < 1e-6:
        return np.full(signal.shape, base_noise_scale, dtype=float)

    hetero_profile = centered_signal / signal_scale
    hetero_profile = hetero_profile + min_scale_ratio
    hetero_profile = hetero_profile / np.mean(hetero_profile)

    return base_noise_scale * (
        (1.0 - heteroscedasticity_strength) + heteroscedasticity_strength * hetero_profile
    )



def _sample_node_noise(base_noise_scale: float, heteroscedasticity_strength: float, signal: np.ndarray) -> np.ndarray:
    return np.random.normal(
        loc=0.0,
        scale=_compute_noise_scale(base_noise_scale, heteroscedasticity_strength, signal),
    )


def _group_nodes_for_correlated_measurement(num_nodes: int, group_size: int) -> List[List[int]]:
    if group_size <= 1:
        return [[node] for node in range(num_nodes)]

    shuffled_nodes = list(range(num_nodes))
    random.shuffle(shuffled_nodes)
    return [shuffled_nodes[i:i + group_size] for i in range(0, len(shuffled_nodes), group_size)]


def _build_structural_functions(
    dag: nx.DiGraph,
    clean_data: np.ndarray,
    noise_scale: float,
    num_samples: int,
    heteroscedasticity_strength: float = 0.0,
):
    topo_order = list(nx.topological_sort(dag))
    functions_per_node = {}

    for node in topo_order:
        parents = list(dag.predecessors(node))
        if not parents:
            f = generate_root_node_func(num_samples)
            functions_per_node[node] = f
            clean_data[:, node] = f(np.empty((num_samples, 1)))
            continue

        parent_data = clean_data[:, parents]
        if parent_data.ndim == 1:
            parent_data = parent_data.reshape(-1, 1)

        normalized_f = _normalize_function(generate_polynomial(parent_data, 2), parent_data)
        functions_per_node[node] = normalized_f
        signal = normalized_f(parent_data)
        clean_data[:, node] = signal + _sample_node_noise(noise_scale, heteroscedasticity_strength, signal)

    return topo_order, functions_per_node


def _build_mechanism_interventions(
    dag: nx.DiGraph,
    clean_data: np.ndarray,
):
    intervention_functions = {}
    for node in dag.nodes:
        parents = list(dag.predecessors(node))
        if not parents:
            intervention_functions[node] = None
            continue

        parent_data = clean_data[:, parents]
        if parent_data.ndim == 1:
            parent_data = parent_data.reshape(-1, 1)
        intervention_functions[node] = _normalize_function(generate_polynomial(parent_data, 3), parent_data)

    return intervention_functions


def _build_measurement_interventions(clean_data: np.ndarray):
    observation_functions = {}
    for node in range(clean_data.shape[1]):
        node_data = clean_data[:, node].reshape(-1, 1)
        observation_functions[node] = _normalize_function(generate_polynomial(node_data, 3), node_data)
    return observation_functions


def corrupt_dag(dag: nx.DiGraph, ratio: float, mode: str = "add") -> nx.DiGraph:
    if ratio <= 0:
        return dag.copy()

    corrupted_dag = dag.copy()
    edges = list(dag.edges())
    n_target = min(len(edges), max(1, int(round(len(edges) * ratio)))) if edges else 0

    if mode == "remove":
        if n_target == 0:
            return corrupted_dag
        for edge in random.sample(edges, n_target):
            corrupted_dag.remove_edge(*edge)
        return corrupted_dag

    if mode == "reverse":
        if n_target == 0:
            return corrupted_dag
        random.shuffle(edges)
        reversed_edges = 0
        for u, v in edges:
            if reversed_edges >= n_target:
                break
            if corrupted_dag.has_edge(v, u):
                continue
            corrupted_dag.remove_edge(u, v)
            corrupted_dag.add_edge(v, u)
            if nx.is_directed_acyclic_graph(corrupted_dag):
                reversed_edges += 1
            else:
                corrupted_dag.remove_edge(v, u)
                corrupted_dag.add_edge(u, v)
        return corrupted_dag

    if mode == "add":
        nodes = list(dag.nodes)
        candidate_edges = []
        for u in nodes:
            for v in nodes:
                if u == v or corrupted_dag.has_edge(u, v):
                    continue
                candidate_edges.append((u, v))

        random.shuffle(candidate_edges)
        added = 0
        for u, v in candidate_edges:
            if added >= n_target:
                break
            corrupted_dag.add_edge(u, v)
            if nx.is_directed_acyclic_graph(corrupted_dag):
                added += 1
            else:
                corrupted_dag.remove_edge(u, v)
        return corrupted_dag

    raise ValueError(f"Unknown DAG corruption mode: {mode}")

def generate_data_with_anomalies(
    num_samples: int,
    num_nodes: int,
    edge_prob: float = 0.3,
    sample_anom_ratio = 0.1,
    noise_scale: float = 0.1,
    seed: int = None,
    outlier_setting = "MEAN_SHIFT",
    mean_shift=0,
    outlier_type = "both",
    dag_misspecification_type: str = None,
    dag_misspecification_ratio: float = 0.0,
    soft_intervention_type: str = None,
    correlated_measurement_group_size: int = 1,
    heteroscedasticity_strength: float = 0.0,
    
) -> Tuple[np.ndarray, np.ndarray, Dict[int, Dict[str, List[int]]], nx.DiGraph, Dict[int, Callable]]:

    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    dag = generate_random_dag(num_nodes, edge_prob=edge_prob, seed=seed)
   
    clean_data = np.zeros((num_samples, num_nodes))
    topo_order, functions_per_node = _build_structural_functions(
        dag,
        clean_data,
        noise_scale,
        num_samples,
        heteroscedasticity_strength=heteroscedasticity_strength,
    )
    mechanism_intervention_functions = _build_mechanism_interventions(dag, clean_data)
    measurement_intervention_functions = _build_measurement_interventions(clean_data)
    corrupted_dag = (
        corrupt_dag(dag, dag_misspecification_ratio, dag_misspecification_type)
        if dag_misspecification_type
        else dag.copy()
    )
    corrupted_data = clean_data.copy()
    anomaly_labels: Dict[int, Dict[str, List[int]]] = {}

    

    # Step 3: Inject generative anomalies
    if outlier_type == "both":
        
        num_generative = int(num_samples *( 1 -(1-sample_anom_ratio)**(1/(2*num_nodes))))
    
    elif outlier_type == "mech":

          
        num_generative = int(num_samples *( 1 -(1-sample_anom_ratio)**(1/(num_nodes))))

    
    if outlier_type != "meas":
        generative_groups = _group_nodes_for_correlated_measurement(num_nodes, correlated_measurement_group_size)
        for node_group in generative_groups:
            gen_samples = np.random.choice(num_samples, num_generative, replace=False)
            for idx in gen_samples:
                for node in node_group:
            
                    #     np.random.uniform(clean_data[:, node].min() - 3, clean_data[:, node].min() - 1),
                    #     np.random.uniform(clean_data[:, node].max() + 1, clean_data[:, node].max() + 3)
                    # ])
                        
                    if outlier_setting == "IN_RANGE_UNIFROM":
                        new_val =  np.random.uniform(np.median(clean_data[:,node]), clean_data[:, node].max())
                    elif outlier_setting == "OUT_OF_RANGE":
                        k1,k2 = off_set_in_std
                        clean_std = np.std(clean_data[:,node])
                        new_val = np.random.uniform(clean_data[:, node].max() + k1*clean_std, clean_data[:, node].max() + k2*clean_std)
                        
                    elif outlier_setting == "MEAN_SHIFT":
                        clean_std = np.std(clean_data[:, node])
                        parents = list(dag.predecessors(node))

                        if soft_intervention_type == "soft_noise":
                            if not parents:
                                new_val = np.random.normal(loc=mean_shift * clean_std, scale=1.0)
                            else:
                                parent_vals = corrupted_data[idx, parents].reshape(1, -1)
                                new_val = functions_per_node[node](parent_vals)[0] + np.random.normal(
                                    loc=mean_shift * clean_std,
                                    scale=_compute_noise_scale(
                                        noise_scale,
                                        heteroscedasticity_strength,
                                        np.array([functions_per_node[node](parent_vals)[0]]),
                                    )[0],
                                )
                        elif soft_intervention_type == "mechanism" and parents:
                            parent_vals = corrupted_data[idx, parents].reshape(1, -1)
                            mechanism_signal = mechanism_intervention_functions[node](parent_vals)[0]
                            new_val = mechanism_signal + np.random.normal(
                                scale=_compute_noise_scale(
                                    noise_scale,
                                    heteroscedasticity_strength,
                                    np.array([mechanism_signal]),
                                )[0]
                            )
                        else:
                            new_sample = np.zeros(num_nodes)
                            for n in topo_order:
                                parent_nodes = list(dag.predecessors(n))
                                if not parent_nodes:
                                    new_sample[n] = np.random.normal(scale=1)
                                else:
                                    parent_vals = new_sample[parent_nodes].reshape(1, -1)
                                    signal = functions_per_node[n](parent_vals)[0]
                                    new_sample[n] = signal + np.random.normal(
                                        scale=_compute_noise_scale(
                                            noise_scale,
                                            heteroscedasticity_strength,
                                            np.array([signal]),
                                        )[0]
                                    )
                            new_val = new_sample[node] + mean_shift * clean_std


                    corrupted_data[idx, node] = new_val

                    # Propagate corruption downstream
                    affected = list(nx.descendants(dag, node))
                    for child in nx.topological_sort(dag):
                        if child not in affected:
                            continue
                        parents = list(dag.predecessors(child))
                        if not parents or child not in functions_per_node:
                            continue
                        parent_vals = corrupted_data[idx, parents]
                        if parent_vals.ndim == 1:
                            parent_vals = parent_vals.reshape(1, -1)
                        signal = functions_per_node[child](parent_vals)[0]
                        corrupted_data[idx, child] = signal + np.random.normal(
                            scale=_compute_noise_scale(
                                noise_scale,
                                heteroscedasticity_strength,
                                np.array([signal]),
                            )[0]
                        )

                    anomaly_labels.setdefault(idx, {"measurement": [], "generative": []})
                    anomaly_labels[idx]["generative"].append(node)

    # Step 2: Inject measurement anomalies

    if outlier_type == "both":
        
        num_measurement =  int(num_samples *( 1 -(1-sample_anom_ratio)**(1/(2*num_nodes))))
      
    elif outlier_type == "meas":
        
        num_measurement =  int(num_samples *( 1 -(1-sample_anom_ratio)**(1/(num_nodes))))
    
    if outlier_type != "mech":
        measurement_groups = _group_nodes_for_correlated_measurement(num_nodes, correlated_measurement_group_size)
        for node_group in measurement_groups:
            meas_samples = np.random.choice(num_samples, num_measurement, replace=False)
            for idx in meas_samples:
                joint_shift = np.random.normal(loc=mean_shift, scale=noise_scale)
                for node in node_group:
            
                
                    if outlier_setting == "IN_RANGE_UNIFROM":
                        new_val =  np.random.uniform(np.median(clean_data[:,node]), clean_data[:, node].max())
                    elif outlier_setting == "OUT_OF_RANGE":
                        k1,k2 = off_set_in_std
                        clean_std = np.std(clean_data[:,node])
                        new_val = np.random.uniform(clean_data[:, node].max() + k1*clean_std, clean_data[:, node].max() + k2*clean_std)
                        
                    elif outlier_setting == "MEAN_SHIFT":
                        clean_std = np.std(clean_data[:, node])
                        current_val = corrupted_data[idx, node]

                        if soft_intervention_type == "soft_noise":
                            new_val = current_val + joint_shift * clean_std
                        elif soft_intervention_type == "mechanism":
                            node_input = np.array([[current_val]])
                            mechanism_signal = measurement_intervention_functions[node](node_input)[0]
                            new_val = mechanism_signal + np.random.normal(
                                scale=_compute_noise_scale(
                                    noise_scale,
                                    heteroscedasticity_strength,
                                    np.array([mechanism_signal]),
                                )[0]
                            )
                        else:
                            new_sample = np.zeros(num_nodes)
                            for n in topo_order:
                                parents = list(dag.predecessors(n))
                                if not parents:
                                    root_signal = np.array([0.0])
                                    new_sample[n] = np.random.normal(scale=1) + np.random.normal(
                                        scale=_compute_noise_scale(
                                            noise_scale,
                                            heteroscedasticity_strength,
                                            root_signal,
                                        )[0]
                                    )
                                else:
                                    parent_vals = new_sample[parents].reshape(1, -1)
                                    signal = functions_per_node[n](parent_vals)[0]
                                    new_sample[n] = signal + np.random.normal(
                                        scale=_compute_noise_scale(
                                            noise_scale,
                                            heteroscedasticity_strength,
                                            np.array([signal]),
                                        )[0]
                                    )
                            new_val = new_sample[node] + joint_shift * clean_std
                
                    corrupted_data[idx,node] = new_val
                    anomaly_labels.setdefault(idx, {"measurement": [], "generative": []})
                    anomaly_labels[idx]["measurement"].append(node)
    # print(F"Clean data: max {clean_data.max()}min: {clean_data.min()}")
    # print(F"Corrupt data: max {corrupted_data.max()}min: {corrupted_data.min()}")
    # Sanity check
    
    assert set(functions_per_node.keys()) == set(dag.nodes), "Function map missing nodesp."
    print(F"SAMPLE RATIO {len(anomaly_labels)/num_samples}")
    return clean_data, corrupted_data, anomaly_labels, dag, functions_per_node, corrupted_dag






def generate_experiment_fixed_root_causes(
    num_samples: int,
    num_nodes: int,
    edge_prob: float = 0.3,
    sample_anom_ratio = 0.1,
    noise_scale: float = 0.1,
    seed: int = None,
    outlier_setting = "MEAN_SHIFT",
    mean_shift=0,
    rc_count = 0,
    dag_misspecification_type: str = None,
    dag_misspecification_ratio: float = 0.0,
    soft_intervention_type: str = None,
    correlated_measurement_group_size: int = 1,
    heteroscedasticity_strength: float = 0.0
      
) -> Tuple[np.ndarray, np.ndarray, Dict[int, Dict[str, List[int]]], nx.DiGraph, Dict[int, Callable]]:

    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    dag = generate_random_dag(num_nodes, edge_prob=edge_prob, seed=seed)

    
    clean_data = np.zeros((num_samples, num_nodes))
    topo_order, functions_per_node = _build_structural_functions(
        dag,
        clean_data,
        noise_scale,
        num_samples,
        heteroscedasticity_strength=heteroscedasticity_strength,
    )
    mechanism_intervention_functions = _build_mechanism_interventions(dag, clean_data)
    measurement_intervention_functions = _build_measurement_interventions(clean_data)
    corrupted_dag = (
        corrupt_dag(dag, dag_misspecification_ratio, dag_misspecification_type)
        if dag_misspecification_type
        else dag.copy()
    )
    corrupted_data = clean_data.copy()
    anomaly_labels: Dict[int, Dict[str, List[int]]] = {}

    

    # Step 3: Inject generative anomalies
    num_total = int(sample_anom_ratio  * num_samples) 
    #gen_samples = np.random.choice(num_samples, num_generative, replace=False)
    total_samples = np.random.choice(num_samples, num_total, replace=False)
    # if rc_count <= 1:
    #     gen_samples = total_samples[:int(sample_anom_ratio  * num_samples*0.5) ]
    #     meas_samples = total_samples[int(sample_anom_ratio  * num_samples*0.5): ]

    # else:
    #     gen_samples = total_samples
    #     meas_samples = total_samples
    gen_samples = set()
    meas_samples = set()
    measurement_groups = _group_nodes_for_correlated_measurement(num_nodes, correlated_measurement_group_size)
    
    for row_idx in total_samples:

        for _ in range(rc_count):
            node = random.choice(list(dag.nodes))
            if random.random() <= 0.5:
                gen_samples.add((row_idx,node))
            else:
                node_group = random.choice(measurement_groups)
                for grouped_node in node_group:
                    meas_samples.add((row_idx, grouped_node))
    
    print("Amount of outliers")
    print("MECH",len(gen_samples))
    print("MEAS",len(meas_samples))
    
    for idx, node in gen_samples:

            
            if outlier_setting == "MEAN_SHIFT":
                clean_std = np.std(clean_data[:, node])
                parents = list(dag.predecessors(node))
                if soft_intervention_type == "soft_noise":
                    if not parents:
                        new_val = np.random.normal(loc=mean_shift * clean_std, scale=1.0)
                    else:
                        parent_vals = corrupted_data[idx, parents].reshape(1, -1)
                        new_val = functions_per_node[node](parent_vals)[0] + np.random.normal(
                            loc=mean_shift * clean_std,
                            scale=_compute_noise_scale(
                                noise_scale,
                                heteroscedasticity_strength,
                                np.array([functions_per_node[node](parent_vals)[0]]),
                            )[0],
                        )
                elif soft_intervention_type == "mechanism" and parents:
                    parent_vals = corrupted_data[idx, parents].reshape(1, -1)
                    mechanism_signal = mechanism_intervention_functions[node](parent_vals)[0]
                    new_val = mechanism_signal + np.random.normal(
                        scale=_compute_noise_scale(
                            noise_scale,
                            heteroscedasticity_strength,
                            np.array([mechanism_signal]),
                        )[0]
                    )
                else:
                    new_sample = np.zeros(num_nodes)
                    for n in topo_order:
                        parent_nodes = list(dag.predecessors(n))
                        if not parent_nodes:
                            new_sample[n] = np.random.normal(scale=1)
                        else:
                            parent_vals = new_sample[parent_nodes].reshape(1, -1)
                            signal = functions_per_node[n](parent_vals)[0]
                            new_sample[n] = signal + np.random.normal(
                                scale=_compute_noise_scale(
                                    noise_scale,
                                    heteroscedasticity_strength,
                                    np.array([signal]),
                                )[0]
                            )
                    new_val = new_sample[node] + mean_shift * clean_std


            corrupted_data[idx, node] = new_val

            # Propagate corruption downstream
            affected = list(nx.descendants(dag, node))
            for child in nx.topological_sort(dag):
                if child not in affected:
                    continue
                parents = list(dag.predecessors(child))
                if not parents or child not in functions_per_node:
                    continue
                parent_vals = corrupted_data[idx, parents]
                if parent_vals.ndim == 1:
                    parent_vals = parent_vals.reshape(1, -1)
                signal = functions_per_node[child](parent_vals)[0]
                corrupted_data[idx, child] = signal + np.random.normal(
                    scale=_compute_noise_scale(
                        noise_scale,
                        heteroscedasticity_strength,
                        np.array([signal]),
                    )[0]
                )

            anomaly_labels.setdefault(idx, {"measurement": [], "generative": []})
            anomaly_labels[idx]["generative"].append(node)

    # Step 2: Inject measurement anomalies

    
   
       
    sampled_joint_shifts = {}
    for idx,node in meas_samples:
      
            
            if outlier_setting == "MEAN_SHIFT":
                clean_std = np.std(clean_data[:, node])
                current_val = corrupted_data[idx, node]
                group_id = (idx, tuple(sorted(group for group in measurement_groups if node in group)[0]))
                joint_shift = sampled_joint_shifts.setdefault(
                    group_id,
                    np.random.normal(loc=mean_shift, scale=noise_scale),
                )
                if soft_intervention_type == "soft_noise":
                    new_val = current_val + joint_shift * clean_std
                elif soft_intervention_type == "mechanism":
                    node_input = np.array([[current_val]])
                    mechanism_signal = measurement_intervention_functions[node](node_input)[0]
                    new_val = mechanism_signal + np.random.normal(
                        scale=_compute_noise_scale(
                            noise_scale,
                            heteroscedasticity_strength,
                            np.array([mechanism_signal]),
                        )[0]
                    )
                else:
                    new_sample = np.zeros(num_nodes)
                    for n in topo_order:
                        parents = list(dag.predecessors(n))
                        if not parents:
                            root_signal = np.array([0.0])
                            new_sample[n] = np.random.normal(scale=1) + np.random.normal(
                                scale=_compute_noise_scale(
                                    noise_scale,
                                    heteroscedasticity_strength,
                                    root_signal,
                                )[0]
                            )
                        else:
                            parent_vals = new_sample[parents].reshape(1, -1)
                            signal = functions_per_node[n](parent_vals)[0]
                            new_sample[n] = signal + np.random.normal(
                                scale=_compute_noise_scale(
                                    noise_scale,
                                    heteroscedasticity_strength,
                                    np.array([signal]),
                                )[0]
                            )
                    new_val = new_sample[node] + joint_shift * clean_std
        
            corrupted_data[idx,node] = new_val
            anomaly_labels.setdefault(idx, {"measurement": [], "generative": []})
            anomaly_labels[idx]["measurement"].append(node)
    # print(F"Clean data: max {clean_data.max()}min: {clean_data.min()}")
    # print(F"Corrupt data: max {corrupted_data.max()}min: {corrupted_data.min()}")
    # Sanity check
    
    assert set(functions_per_node.keys()) == set(dag.nodes), "Function map missing nodesp."
    print(F"SAMPLE RATIO {len(anomaly_labels)/num_samples}")
    return clean_data, corrupted_data, anomaly_labels, dag, functions_per_node, corrupted_dag





