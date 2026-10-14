"""Small causal-discovery helpers for examples and lightweight workflows."""

from __future__ import annotations

import networkx as nx
import numpy as np
from sklearn.linear_model import LinearRegression

from ._replication import ensure_dodiscover_on_path, ensure_topic_on_path


def dag_from_adjacency(adjacency: np.ndarray, threshold: float = 0.0) -> nx.DiGraph:
    """Convert a weighted adjacency matrix into a ``networkx.DiGraph``."""
    weights = np.asarray(adjacency, dtype=float)
    if weights.ndim != 2 or weights.shape[0] != weights.shape[1]:
        raise ValueError("adjacency must be a square matrix")

    dag = nx.DiGraph()
    dag.add_nodes_from(range(weights.shape[0]))
    rows, cols = np.where(np.abs(weights) > threshold)
    for parent, child in zip(rows, cols):
        if parent == child:
            continue
        dag.add_edge(int(parent), int(child), weight=float(weights[parent, child]))

    if not nx.is_directed_acyclic_graph(dag):
        raise ValueError("The thresholded adjacency matrix contains a directed cycle.")
    return dag


def discover_dag_from_order(
    data: np.ndarray,
    variable_order: list[int] | tuple[int, ...],
    threshold: float = 0.05,
) -> nx.DiGraph:
    """Learn a simple DAG when a causal/topological order is known.

    Each variable is regressed on earlier variables in ``variable_order``. Edges
    whose absolute linear coefficient exceeds ``threshold`` are kept. This is a
    lightweight teaching helper, not a replacement for a full causal discovery
    package.
    """
    data = np.asarray(data, dtype=float)
    if data.ndim != 2:
        raise ValueError("data must be a two-dimensional array")

    n_features = data.shape[1]
    order = list(variable_order)
    if sorted(order) != list(range(n_features)):
        raise ValueError("variable_order must contain every feature index exactly once")

    dag = nx.DiGraph()
    dag.add_nodes_from(range(n_features))

    for child_position, child in enumerate(order):
        parents = order[:child_position]
        if not parents:
            continue
        model = LinearRegression().fit(data[:, parents], data[:, child])
        for parent, coefficient in zip(parents, model.coef_):
            if abs(coefficient) > threshold:
                dag.add_edge(int(parent), int(child), weight=float(coefficient))

    return dag


def discover_dag_topic(data: np.ndarray, **topic_kwargs) -> nx.DiGraph:
    """Learn a DAG with the bundled Python TOPIC implementation.

    TOPIC is included in the replication package. Its continuous-data path is
    Python-based and can be used when the optional causal-discovery dependencies
    are installed.
    """
    ensure_topic_on_path()
    try:
        from topic.topic import Topic
    except ImportError as exc:
        raise ImportError(
            "TOPIC is available from a full repository checkout with the "
            "bundled replication package, but it is not included in minimal "
            "wheel/PyPI-style distributions."
        ) from exc

    data = np.asarray(data, dtype=float)
    if data.ndim != 2:
        raise ValueError("data must be a two-dimensional array")

    dag, _ = Topic(**topic_kwargs).fit(data)
    if not isinstance(dag, nx.DiGraph):
        dag = nx.DiGraph(dag)
    dag.add_nodes_from(range(data.shape[1]))
    return dag


def discover_dag_nogam(data: np.ndarray, **nogam_kwargs) -> nx.DiGraph:
    """Learn a DAG with the bundled Python NoGAM implementation.

    This wrapper uses the NoGAM implementation included in the replication
    package via ``dodiscover.toporder.NoGAM``. It has no R dependency.
    """
    ensure_dodiscover_on_path()
    try:
        import pandas as pd
        from dodiscover import make_context
        from dodiscover.toporder.nogam import NoGAM
    except ImportError as exc:
        raise ImportError(
            "NoGAM dependencies are not installed. Install them with "
            "`pip install measurement-mechanistic-rca[causal-discovery]`."
        ) from exc

    data = np.asarray(data, dtype=float)
    if data.ndim != 2:
        raise ValueError("data must be a two-dimensional array")

    data_df = pd.DataFrame(data, columns=list(range(data.shape[1])))
    context = make_context().variables(data=data_df).build()
    model = NoGAM(**nogam_kwargs)
    model.learn_graph(data_df, context)
    dag = model.graph_
    if not isinstance(dag, nx.DiGraph):
        dag = nx.DiGraph(dag)
    dag.add_nodes_from(range(data.shape[1]))
    return dag
