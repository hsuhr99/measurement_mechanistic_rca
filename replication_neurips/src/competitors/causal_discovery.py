from src.competitors.topic.src.topic.topic import Topic

from dodiscover.toporder.cam import CAM 
from dodiscover.toporder.score import SCORE
from dodiscover.toporder.nogam import NoGAM
from dodiscover import make_context

from castle.algorithms import Notears as Notears, NotearsNonlinear as NotearsMLP

import numpy as np 
import networkx as nx
import pandas as pd
from pyod.models.auto_encoder import AutoEncoder
import torch
from scipy.stats import norm
import torch

from sklearn.preprocessing import StandardScaler, RobustScaler
torch.set_default_dtype(torch.float32) 
import lingam
# --- Wrapper for CAM ---
def cam_wrapper(data):
    """
    Input: data as a pandas DataFrame or numpy array
    Output: NetworkX DAG
    """
    cam = CAM()
    data = pd.DataFrame(data)
    context = make_context().variables(data=data).build()
    cam.learn_graph(data,context)
    G_est = cam.graph_ 
    # Ensure it's a NetworkX DiGraph
    if not isinstance(G_est, nx.DiGraph):
        print(G_est)
        G_est = nx.DiGraph(G_est)
    if set(G_est.nodes) != set(list(range(data.shape[1]))):
        raise ValueError(f"CAM did not return a complete graph with all nodes present in the data columns. Expected nodes: {set(list(range(data.shape[1])))}, found nodes: {set(G_est)}")
    return G_est


# --- Wrapper for SCORE ---
def score_wrapper(data):
    """
    Input: data as a pandas DataFrame or numpy array
    Output: NetworkX DAG
    """
    score = SCORE()
    data = pd.DataFrame(data)
    context = make_context().variables(data=data).build()
    score.learn_graph(data,context)
    G_est = score.graph_
    if not isinstance(G_est, nx.DiGraph):
        print(G_est)
        G_est = nx.DiGraph(G_est)

    if set(G_est.nodes) != set(list(range(data.shape[1]))):
        raise ValueError(f"SCORE did not return a complete graph with all nodes present in the data columns. Expected nodes: {set(list(range(data.shape[1])))}, found nodes: {set(G_est)}")
    return G_est

# --- Wrapper for NoGAM ---
def nogam_wrapper(data):
    """
    Input: data as a pandas DataFrame or numpy array
    Output: NetworkX DAG
    """

    data = pd.DataFrame(data)
    context = make_context().variables(data=data).build()
    nogam = NoGAM()
    nogam.learn_graph(data,context)
    G_est = nogam.graph_
    if not isinstance(G_est, nx.DiGraph):
        G_est = nx.DiGraph(G_est)
    if set(G_est.nodes) != set(list(range(data.shape[1]))):
        raise ValueError(f"NoGAM did not return a complete graph with all nodes present in the data columns. Expected nodes: {set(list(range(data.shape[1])))}, found nodes: {set(G_est)}")
    return G_est



def dag_search_oracle_anomalies(data,labels,cd_algorithm):
    
    data = RobustScaler().fit_transform(data)
    # clf = AutoEncoder()
    clean_indices = list (set(range(data.shape[0])).difference(set(labels.keys())))

    clean_data = data[clean_indices]

    return cd_algorithm(clean_data)


# --- Wrapper for Topic --- #
def topic_wrapper(data):
    dag,_ = Topic().fit(data)

    return dag


def notears_mlp_wrapper(data, lambda1=0.01,max_iter=1000,hidden_layers=(10,1),w_threshold=0.1):
    model = NotearsMLP(
    lambda1=lambda1,
    max_iter=max_iter,
    hidden_layers=hidden_layers,
    w_threshold=w_threshold,
    model_type="mlp",
    expansions=10,
    bias=True,
    device_type="cpu"
)   
    data = data.astype(np.float32)
    model.learn(data)
    W_est = model.causal_matrix

    W_bin = (np.abs(W_est) > w_threshold).astype(int)

    row_indices, col_indices = np.nonzero(W_bin)

    dag_nx = nx.DiGraph()

    for i in range(data.shape[1]):
        dag_nx.add_node(i)

    for row_id,col_id in zip(row_indices, col_indices):
        dag_nx.add_edge(row_id, col_id)

        if not nx.is_directed_acyclic_graph(dag_nx):
            dag_nx.remove_edge(row_id, col_id)
    
    if set(dag_nx.nodes) != set(list(range(data.shape[1]))):
        raise ValueError(f"Notears did not return a complete graph with all nodes present in the data columns. Expected nodes: {set(list(range(data.shape[1])))}, found nodes: {set(dag_nx.nodes)}")

    return dag_nx

# import lingam


def lingam_wrapper(data, method='DirectLiNGAM', **kwargs):
    """
    Runs a LiNGAM algorithm on data and returns a DAG as a NetworkX DiGraph.

    Parameters:
    - data: pandas DataFrame or numpy array (observations x variables)
    - method: str, 'DirectLiNGAM' or 'ICA-LiNGAM'
    - kwargs: additional parameters passed to the LiNGAM model
    
    Returns:
    - nx.DiGraph representing the causal DAG
    """
    # Convert data to numpy if it is a DataFrame
    if isinstance(data, pd.DataFrame):
        columns = data.columns
        data_np = data.values
    else:
        data_np = data
        columns = [f'X{i}' for i in range(data_np.shape[1])]

    # Select model
    if method == 'DirectLiNGAM':
        model = lingam.DirectLiNGAM(**kwargs)
    elif method == 'ICALiNGAM':
        model = lingam.ICALiNGAM(**kwargs)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Fit model
    model.fit(data_np)

    # Get adjacency matrix
    adjacency_matrix = model.adjacency_matrix_

    # Convert to NetworkX DiGraph
    G = nx.DiGraph()
    for i, _ in enumerate(columns):
        G.add_node(i)
    for i, _ in enumerate(columns):
        for j, _ in enumerate(columns):
            if adjacency_matrix[i, j] != 0:
                G.add_edge(i ,j, weight=adjacency_matrix[i, j])
    G_est = G
    if set(G_est.nodes) != set(list(range(data.shape[1]))):
        raise ValueError(f"LinGAM did not return a complete graph with all nodes present in the data columns. Expected nodes: {set(list(range(data.shape[1])))}, found nodes: {set(G_est)}")
    return G_est

def clean_data_ae(cleaned_data, remove_quantile):
    # Train the autoencoder
    # print(f"CLEANED DATA DTYPE: {cleaned_data.dtype}")
    
  
    clf = AutoEncoder()
    # clf.model = clf.model.float()
    cleaned_data = np.array(cleaned_data, dtype=np.float32)
    # cleaned_data = torch.tensor(cleaned_data.astype('double') ) # convert NumPy array to float32
# clf.fit(cleaned_data)

    clf.fit(cleaned_data)
    
    # Get reconstruction error scores
    scores = clf.decision_scores_
    
    # Determine the threshold for the top remove_quantile
    threshold = np.quantile(scores, 1 - remove_quantile)
    
    # Keep only points with scores below the threshold
    cleaned_data_filtered = cleaned_data[scores < threshold]
    
    return cleaned_data_filtered

    
def multivariate_robust_outlier_mask(X, alpha=0.05):
    """
    Marks as outlier if any feature is a robust outlier. Applies Bonferroni correction.
    """
    X = np.asarray(X)

    n, d = X.shape
    if d == 0:
        d+= 1
        
    corrected_alpha = 1 - (1 - alpha)**(1 / (2*n * d))  # Conservative Bonferroni-style correction
    z_thresh = norm.ppf(1 - corrected_alpha )

    z_scores = np.zeros_like(X)

    for j in range(d):
        feature = X[:, j]
        med = np.median(feature)
        mad = np.median(np.abs(feature - med))
        mad = max(mad, 1e-6)
        z_scores[:, j] = np.abs(feature - med) / (1.4826 * mad)

    return np.all(z_scores < z_thresh, axis=1)  # True for inliers (safe points)


def clean_data_z(data,remove_quantile):

    cleaned_data = data[multivariate_robust_outlier_mask(data,alpha=remove_quantile)]
    return cleaned_data

    
def dag_search_ae_anomalies(data,cd_algorithm,alpha = 0.05):
    data = RobustScaler().fit_transform(data)
    clean_data = clean_data_ae(data,alpha)

    return cd_algorithm(clean_data )
    
def dag_search_z_anomalies(data,cd_algorithm,alpha = 0.05):
    
    data = RobustScaler().fit_transform(data)
    clean_data = clean_data_z(data,alpha)

    return cd_algorithm(clean_data)

