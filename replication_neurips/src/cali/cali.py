import networkx as nx
import numpy as np
from .robust_regression.utils import log_binom
from .robust_regression.qgam_regression import robustly_regress_qgam
from joblib import Parallel, delayed
from .utils import ConstantFunction, fit_dists

from sklearn.preprocessing import RobustScaler
RUN_PARALLEL = False


from .sampling import RandomSampler, KNNSampler

from .utils import markov_blanket


from scipy.special import logsumexp


def uniform_cost(a,b):

    return np.log2(b-a)


def _scalar(value):
    return float(np.asarray(value).reshape(-1)[0])


def _regress_single(child, parents, data,regression_alg="QGAM",regression_settings=None):
    if len(parents) == 0:
        y = data[:, child]
        return ((tuple(), child), ConstantFunction(y))
    else:
        X = data[:, parents]
        y = data[:, child].reshape((-1, 1))
        if regression_alg in {"QGAM", "SKLEARN_QUANTILE_SPLINE", "PYGAM_EXPECTILE"}:
            settings = regression_settings or {}
            backend = settings.get("backend")
            if regression_alg == "SKLEARN_QUANTILE_SPLINE":
                backend = "sklearn_quantile_spline"
            elif regression_alg == "PYGAM_EXPECTILE":
                backend = "pygam_expectile"

            func = robustly_regress_qgam(
                X,
                y,
                mode=settings.get("mode", "default"),
                loss=settings.get("loss", "L1"),
                scale=settings.get("scale", False),
                discrete=settings.get("discrete", False),
                backend=backend or "r_qgam",
                qu=settings.get("qu", 0.5),
            )
        else:
            raise ValueError("Not supported")
        return ((tuple(sorted(parents)), child), func)

def regress_dag(parent_child_pairs, data, n_jobs=-1,regression_alg="QGAM",regression_settings=None):
   

    if np.any(np.isnan(data)):
        raise ValueError("Data contains nan")
   
    if RUN_PARALLEL == True:   
        with  Parallel(n_jobs=n_jobs, backend="loky") as parallel:  
            results = parallel(
                delayed(_regress_single)(child, parents, data,regression_alg,regression_settings)
                for child, parents in parent_child_pairs
            )

    else:
        
        results = [_regress_single(child,parents,data,regression_alg,regression_settings) for child,parents in parent_child_pairs]


    return dict(results)



def sample_cost(row_id, data, data_min, data_max, dag, functions,
                    residuals, residual_distributions,
                    meas_indices, gen_indices, sampler,
                    setting="MARGINALIZE", N_MC=100):

    
    cost = 0.0
    D = data.shape[1]
    all_indices = set(range(D))
    clean_residuals = residuals[row_id].copy()

    # --- Generative outliers (treated as uniform priors on those coords) ---
    for idx in gen_indices:
        if not idx in meas_indices:
            cost += uniform_cost(a=data_min[idx], b=data_max[idx])
        all_indices.discard(idx)

    # --- Measurement outliers (we will marginalize over these) ---
    if meas_indices:
        # Build set of factors (parents tuple, child) we need to re-evaluate.
        # We want:
        #  - every marginalized node itself (we won't include its likelihood factor)
        #  - every child of a marginalized node (we *will* evaluate the child's likelihood
        #    under sampled parent values) — immediate children only.
        parent_child_set = set()
        for col_id in meas_indices:
            cost += uniform_cost(a=data_min[col_id], b=data_max[col_id])
            parents = list(dag.predecessors(col_id))
            parent_child_set.add((tuple(sorted(parents)), col_id))

            if col_id in all_indices:
                all_indices.discard(col_id)

            # immediate children: we'll need to evaluate their likelihood under sampled parents
            for child in dag.successors(col_id):
                if child in gen_indices:
                    continue
                if child in all_indices:
                    all_indices.discard(child)
                child_parents = list(dag.predecessors(child))
                parent_child_set.add((tuple(sorted(child_parents)), child))

        # --- 1) Draw MC samples for each measurement noise (N_MC x |meas_indices|) ---
        mc_noise = dict()
        for col_id in meas_indices:
            if col_id not in gen_indices:
                mc_noise[col_id] = residual_distributions[col_id].sample(N_MC).reshape(N_MC,)  # shape (N_MC,)

        # --- 2) Build mc_data: each row is a full-data vector for one MC sample ---
        mc_data = np.zeros((N_MC, D))
        # fill non-marginalized columns with observed values (broadcast)
        for col_id in range(D):
            if col_id not in meas_indices:
                mc_data[:, col_id] = data[row_id, col_id]
        
        # fill marginalized cols by using structural prediction + sampled noise
        meas_order = [
            i for i in nx.topological_sort(dag)
            if i in meas_indices
        ]

        for col_id in meas_order:
            parents = list(dag.predecessors(col_id))
            parents_tuple = tuple(sorted(parents))
            fun = functions[(parents_tuple, col_id)]
            # prepare X (N_MC x n_parents). If no parents, X is ignored by predict; create zeros
            if parents:
                # parent values: for parents not in meas_indices, these are observed (broadcast)
                # but if any parent is itself marginalized, its value is in mc_data already
                X = np.column_stack([mc_data[:, p] if p in meas_indices else np.full(N_MC, data[row_id, p])
                                     for p in parents])
                pred = fun.predict(X)[0]  # should return shape (N_MC,) or (N_MC,1)
            else:
                X = np.zeros((N_MC, 1))
                pred = fun.predict(X)
            pred = pred.reshape(N_MC,)

            if col_id not in gen_indices:
                mc_data[:, col_id] = pred + mc_noise[col_id]
            else:
                alpha = 0.5  # U-shaped
                X = np.random.beta(alpha, alpha, size=N_MC)
                mc_data[:, col_id] = data_min[col_id] + X * (data_max[col_id] - data_min[col_id])

        # --- 3) For each factor (parents, child) in parent_child_set where child NOT marginalized,
        # evaluate the log-likelihood of the observed child value under sampled parents ---
        # We'll collect per-sample log-likelihood contributions and then combine them.
        # Number of factors we will evaluate:
        factors = [(parents, child) for (parents, child) in parent_child_set if child not in meas_indices]
        
        if len(factors) == 0:
            # nothing to evaluate: degenerate case
            sample_ll = np.zeros(N_MC)
        else:
            mc_residuals = np.zeros((N_MC, len(factors)))  # rows: samples, cols: factors
            
            for j, (parents, child) in enumerate(factors):
                fun = functions[(parents, child)]
                # y_obs is scalar (observed child value), replicate to length N_MC
                y_obs = np.full(N_MC, data[row_id, child])
                if parents:
                    X = np.column_stack([mc_data[:, p] if p in meas_indices else np.full(N_MC, data[row_id, p])
                                         for p in parents])
                else:
                    X = np.zeros((N_MC, 1))
                try:
                    res = fun.get_scaled_residuals(X, y_obs)  # should return shape (N_MC,)
                except Exception:
                    pred = fun.predict(X).reshape(N_MC,)
                    res = y_obs - pred
                # Evaluate log-likelihood for each sample
                ll_vals = residual_distributions[child].ll(res)   # shape (N_MC,)
                mc_residuals[:, j] = ll_vals

            # Sum per-sample log-likelihood across factors to get sample-wise ll
            sample_ll = mc_residuals.sum(axis=1)   # shape (N_MC,)

        # --- 4) Monte Carlo marginalization: log(1/N sum exp(sample_ll)) ---
        log_marg = logsumexp(sample_ll) - np.log(N_MC)   # log p(data | marginalized nodes)
        cost += -log_marg   # cost is negative log-likelihood

        for child in meas_indices:
            
            clean_residuals[child] = residual_distributions[child].mean 

    # --- Clean residuals: include likelihood of nodes not marginalized or generative-outliers ---
    for col_id in sorted(all_indices):
        # For indices still in all_indices, we include their normal factor
        resid = residuals[row_id, col_id].reshape(-1,)   # scalar-like
        ll = residual_distributions[col_id].ll(resid)    # expects 1-d or scalar; returns scalar log-lik
        cost += -_scalar(ll)

    return cost, clean_residuals

def add_measurement_error(
    variable_idx: int,
    sample_idx: int,
    data: np.ndarray,
    dag,
    residuals: np.ndarray,
    functions: dict,
    residual_distributions: list,
    data_min,
    data_max,
    assignment,
    sampler,
    setting
):
    """
    Optimize a replacement value for a single variable in a sample,
    maximizing gain in outliers (ignoring nodes/functions already fixed by generative errors).
    """
   

    


    gen_ids = set()

    meas_ids = set()

    if sample_idx in assignment:
        for child_id in list(assignment[sample_idx]["generative_attributions"].keys()) + list(assignment[sample_idx]["uncertain_origin"].keys()):
            
            if child_id == variable_idx:
                
                raise ValueError("Trying already assigned error")
            
            gen_ids.add(child_id)

        
        for child_id in list(assignment[sample_idx]["measurement_corrections"].keys()):
            if child_id == variable_idx:

                raise ValueError("Trying already assigned error")
              
            meas_ids.add(child_id)


        current_cost = assignment[sample_idx]["sample_cost"]
    else:
        current_cost,_ =  sample_cost(
            row_id=sample_idx,
            data =data,
            data_min = data_min,
            data_max = data_max,
            dag = dag,
            functions = functions,
            residuals=residuals,
            residual_distributions=residual_distributions,
            meas_indices=meas_ids,
            gen_indices=gen_ids,
            sampler=sampler,
            setting=setting
        )

    new_cost, clean_residuals = sample_cost(
        row_id=sample_idx,
        data =data,
        data_min=data_min,
        data_max=data_max,
        dag = dag,
        functions = functions,
        residuals=residuals,
        residual_distributions=residual_distributions,
        meas_indices=list(meas_ids) + [variable_idx],
        gen_indices=list(gen_ids),
        sampler=sampler,
        setting=setting
    )

    return current_cost - new_cost, new_cost, clean_residuals



def add_generative_error(
    variable_idx: int,
    sample_idx: int,
    data: np.ndarray,
    data_min,
    data_max,
    dag,
    residuals: np.ndarray,
    functions: dict,
    residual_distributions: list,
    assignment,
    sampler,
    setting,
    
):
    """
    Evaluate gain from adding a generative outlier for `variable_idx` in row `sample_idx`.
    If no measurement outlier exists in its Markov blanket, use raw residual.
    Otherwise, re-evaluate with `sample_cost` (marginalization or MAP).
    Returns: relative gain (new_cost - current_cost).
    """

    if sample_idx in assignment:
        current_cost = assignment[sample_idx]["sample_cost"]

        gen_ids = set(assignment[sample_idx]["generative_attributions"].keys()) \
            | set(assignment[sample_idx]["uncertain_origin"].keys())
        meas_ids = set(assignment[sample_idx]["measurement_corrections"].keys())
    else:

        gen_ids = []
        meas_ids = []
        current_cost,_  = sample_cost(
            row_id=sample_idx,
            data=data,
            data_min=data_min,
            data_max=data_max,
            dag=dag,
            functions=functions,
            residuals=residuals,
            residual_distributions=residual_distributions,
            meas_indices=[],
            gen_indices=[],
            sampler=sampler,
            setting=setting,
        )

    blanket = set(markov_blanket(dag, variable_idx))
    if not (blanket & set(meas_ids)):
        # raw residual plus uniform prior
        res_val = residuals[sample_idx, variable_idx]
        uniform_val = uniform_cost(a=data_min[variable_idx], b=data_max[variable_idx])
        new_cost = current_cost + _scalar(residual_distributions[variable_idx].ll(res_val.reshape(1,-1))) + uniform_val
        return current_cost - new_cost, new_cost


    # --- Case 2: Connected to measurement error -> need full recomputation ---
    new_cost,_ = sample_cost(
        row_id=sample_idx,
        data=data,
        data_min=data_min,
        data_max=data_max,
        dag=dag,
        functions=functions,
        residuals=residuals,
        residual_distributions=residual_distributions,
        meas_indices=list(meas_ids),
        gen_indices=list(gen_ids) + [variable_idx],
        sampler=sampler,
        setting=setting,
    )

    return current_cost - new_cost,new_cost


def assign_errors( dag,
    data,
    functions,
    residuals,
    residual_distributions,
    sampler,
    setting="MARGINALIZE",
    labels = None
    
):
    
    # Calculate accumulated residuals for each sample
    n, d = data.shape
    data_min = data.min(axis=0)
    data_max = data.max(axis=0)
    successors = {j: list(dag.successors(j)) for j in dag.nodes()}
    
    u_cost = uniform_cost(a=data_min,b=data_max)
    accumulated_max_gains = np.zeros((n, d))

    cleaned_residuals = residuals.copy()

    # Precompute: optimal residual cost for zero residual (i.e., minimal encoding cost)
    optimal_residual_cost = np.zeros(d)

    # map gains to candidates ad candidate to gains
    
    for k in range(d):

     
        optimal_residual_cost[k] = -_scalar(residual_distributions[k].ll(np.array([[ residual_distributions[k].mean]])))

    # Compute max gain per (row i, feature j) from imputing j in row i
    for j in range(d):
        

        # Ensure that child nodes get zero gain since they are handled at a later pont
        if successors[j]:
            affected_nodes = [j] + successors[j] 

            for k in affected_nodes:
                current_cost = -residual_distributions[k].ll(residuals[:, k])  # shape (n,)
                gain = current_cost - optimal_residual_cost[k]
                accumulated_max_gains[:, j] += gain  # shape (n,)
            
            accumulated_max_gains[:,j] -= (u_cost[j] +log_binom(n,1))
    
    # Flatten into (row, feature) pairs
    row_idx, col_idx = np.indices((n, d))
    all_indices = np.stack((row_idx.ravel(), col_idx.ravel()), axis=1)

    all_gains = accumulated_max_gains.flatten()

    
    
    # Sort all (i, j) pairs by descending max gain
    flat_idx = np.argsort(-all_gains)  # vectorized
    sorted_candidates = [(divmod(idx, d), all_gains[idx]) for idx in flat_idx if all_gains[idx] >0]

    
    assignment = {}
    error_assignment_measurement = {i:set() for i in range(d)}
    error_assignment_generative = {i: set() for i in range(d)}
    error_assignment_uncertain_origin = {i:set() for i in range(d)}

    for (i, j), _ in sorted_candidates:
        
        # Check if optimal gain still positive
        affected_nodes = [j] + successors[j] 
        
        optimal_gain = 0
        
        for k in affected_nodes:

            current_cost = -residual_distributions[k].ll(cleaned_residuals[i, k].reshape(1,-1))  # shape (n,)
            gain = _scalar(current_cost - optimal_residual_cost[k])
            optimal_gain += gain
        delta = log_binom(n,len(error_assignment_measurement[j])) -  log_binom(n,len(error_assignment_measurement[j])+1) 
        
        optimal_gain += delta
        optimal_gain -= u_cost[j] 

        if optimal_gain <= 0:
            continue
        
        measurement_gain, new_cost, new_residual = add_measurement_error(
            variable_idx=j,
            sample_idx=i,
            data=data,
            dag=dag,
            data_min=data_min,
            data_max=data_max,
            residual_distributions=residual_distributions,
            residuals=residuals,
            sampler=sampler,
            assignment=assignment,
            setting=setting,
            functions=functions

        )
        if measurement_gain + delta > 0:
            
            # Make sure that measurement gain, higher than generative gain for highest affected residual in that row to avoid spurious discoveries, we compare to the cost, of encoding all reduced residuals with generative encodings.

            affected_nodes = [j] + successors[j]
            gen_gain = 0

            for col_id in affected_nodes:
                
                if (i in assignment) and (col_id in assignment[i]["measurement_corrections"]):
                    continue
        
                additional_gain, _ =  add_generative_error(
                    variable_idx=col_id,
                    sample_idx=i,
                    data=data,
                    dag=dag,
                    data_min=data_min,
                    data_max=data_max,
                    residual_distributions=residual_distributions,
                    residuals=residuals,
                    sampler=sampler,
                    assignment=assignment,
                    setting=setting,
                    functions=functions,

                ) 

               
                if additional_gain - log_binom(n,1) > 0:
                    gen_gain += additional_gain - log_binom(n,1)
            
            if measurement_gain + delta < gen_gain:
                continue
            
            else:
                
                measurement_correction = {
                    "base_gain":measurement_gain.copy(),
                    "gain":measurement_gain + delta
                }
            if i not in assignment:
                assignment[i] = {"measurement_corrections":{},"uncertain_origin":{},"generative_attributions":{}}
            
            if len(successors[j]) == 0:
                raise ValueError("No imputation for child nodes expected")
                
            else:

               
                assignment[i]["measurement_corrections"][j] = measurement_correction

            assignment[i]["sample_cost"] = new_cost
            cleaned_residuals[i] = new_residual 

            error_assignment_measurement[j].add(i)
    
    

        # Compute all per-sample, per-variable costs in one go
    all_costs = np.column_stack([
        -dist.ll(cleaned_residuals[:, j])
        for j, dist in enumerate(residual_distributions)
    ])   # shape (n, d)

    # Flatten into vector of gains
    flat_costs = all_costs.ravel()   # shape (n*d,)

    # Sort by descending cost
    sorted_idx = np.argsort(-flat_costs)
    sorted_generative_candidates = [
        (tuple(all_indices[k]), flat_costs[k]) for k in sorted_idx
    ]

    for (i, j), gain in sorted_generative_candidates:
        
        is_child_node = len(successors[j]) == 0

        if i in assignment:
            if j in assignment[i]["measurement_corrections"]:
                continue

        positional_delta = 0
        
        if is_child_node:
            positional_delta = log_binom(n,len(error_assignment_uncertain_origin[j])) - log_binom(n,len(error_assignment_uncertain_origin[j])+1)
        else:
            positional_delta  = log_binom(n,len(error_assignment_generative[j])) - log_binom(n,len(error_assignment_generative[j])+1)
        

        generative_gain, new_cost = add_generative_error(
            variable_idx=j,
            sample_idx=i,
            data=data,
            data_min=data_min,
            data_max=data_max,
            dag=dag,
            residuals=residuals,
            functions=functions,
            residual_distributions=residual_distributions,
            assignment=assignment,
            sampler=sampler,
            setting=setting            
        )
        if generative_gain + positional_delta <= 0:
            continue
        else:
             
            if len(successors[j]) == 0:
                if i not in assignment:
                    assignment[i] = {"measurement_corrections":{},"uncertain_origin":{},"generative_attributions":{}}
                
                
                    
                assignment[i]["uncertain_origin"][j]  =  {
                    
                    "gain": generative_gain + positional_delta,
                    "base_gain":generative_gain.copy(),
                    
                }

                
                error_assignment_uncertain_origin[j].add(i)
                
                    
            else:

               
                if i not in assignment: 
                    assignment[i] = {"measurement_corrections":{},"uncertain_origin":{},"generative_attributions":{}}
                assignment[i]["generative_attributions"][j] =  {
                    
                    "gain":generative_gain + positional_delta,
                    "base_gain":generative_gain.copy(),
                }

                error_assignment_generative[j].add(i)   

            assignment[i]["sample_cost"] = new_cost

        
    return assignment, cleaned_residuals, error_assignment_generative, error_assignment_measurement, error_assignment_uncertain_origin


def calculate_residuals(functions,data):
    residuals = np.zeros_like(data)
    for (parents,child),fun in functions.items():

        y = data[:,child]
        X = data[:,list(parents)] if parents else np.zeros_like(y)

        residuals[:,child] = fun.get_scaled_residuals(X,y)#y.flatten() - fun.predict(X).flatten()

    return residuals


def cali(
        
        dag: nx.DiGraph, 
        data: np.array, 
        precision: float = 1,
        functions = None,
        imputation_functions = None,
        regression_alg="QGAM",
        regression_settings= {
            "mode":"default",
            "scale":False,
            "loss":"L1",
            "discrete":False
        },
        sampler_setting = "KNN",
        meas_setting="MARGINALIZE",
        labels=None
       
        ):
    """
    Scores a DAG using robust regression and Minimum Description Length (MDL) with outlier detection
    """
    data = RobustScaler().fit_transform(data)

    n, d = data.shape
    data = data.copy()
    data_max = data.max(axis=0)
    data_min = data.min(axis=0)

    feature_scores = np.zeros_like(data)

    parent_child_pairs = [(child, list(dag.predecessors(child))) for child in dag.nodes]
    if functions is None:
        functions = regress_dag(parent_child_pairs, data,regression_alg=regression_alg,regression_settings=regression_settings)
    
    
    if sampler_setting == "KNN":

        random_sampler = KNNSampler(data,dag)
    elif sampler_setting == "DEFAULT":
        random_sampler = RandomSampler(data,dag)


    # 4. Calculate residuals from DAG regressions
    residuals = calculate_residuals(functions, data)
    
    # 5. Fit distributions to residuals and wrap
    residual_distributions = fit_dists(residuals)

    # 6. Assign errors based on residual outlier detection
    assignment, _, _, _, _ = assign_errors(
        
        dag=dag,
        data=data, 
        functions=functions, 
        residuals=residuals,
        residual_distributions=residual_distributions,
        sampler = random_sampler,
        setting=meas_setting,
        labels=labels
        
    )
    classifications = np.zeros_like(feature_scores)
    confidences = np.zeros_like(feature_scores)

    for sample_idx, fit_result in assignment.items():
       
        generative_errors = fit_result["generative_attributions"]
        measurement_errors = fit_result["measurement_corrections"]
        uncertain_origin =  fit_result["uncertain_origin"]

        for child in generative_errors:
            feature_scores[sample_idx, child] = generative_errors[child]["base_gain"]

            flip_sample_cost,_ = sample_cost(
                row_id = sample_idx,
                data = data,
                data_min = data_min,
                data_max = data_max,
                functions=functions,
                residuals=residuals,
                sampler=random_sampler,
                setting=meas_setting,
                dag = dag,
                residual_distributions = residual_distributions,
                meas_indices =  list(measurement_errors.keys()) + [child],
                gen_indices = [col_id for col_id in  list(generative_errors.keys()) + list(uncertain_origin.keys()) if col_id != child]
            )
            generative_errors[child]["confidence"] =  (flip_sample_cost - assignment[sample_idx]["sample_cost"] ) 
            
          
         
            confidences[sample_idx,child] =  generative_errors[child]["confidence"]

        for child in measurement_errors:
            
            feature_scores[sample_idx, child] = measurement_errors[child]["base_gain"]

            flip_sample_cost,_ = sample_cost(
                row_id = sample_idx,
                data = data,
                data_min = data_min,
                data_max = data_max,
                dag = dag,
                residuals=residuals,
                functions = functions,
                residual_distributions = residual_distributions,
                meas_indices =  list(col_id for col_id  in measurement_errors if col_id != child),
                gen_indices = list(generative_errors.keys()) + list(uncertain_origin.keys()) + [child],
                sampler = random_sampler,
                setting=meas_setting
            )

            measurement_errors[child]["confidence"] =  (flip_sample_cost - assignment[sample_idx]["sample_cost"] )
            classifications [sample_idx,child] = 1
            confidences[sample_idx,child] =  measurement_errors[child]["confidence"]

        for child in uncertain_origin:
            
            feature_scores[sample_idx, child] = uncertain_origin[child]["base_gain"]
            uncertain_origin[child]["confidence"] = 0

        col_ids = np.nonzero(feature_scores[sample_idx]==0)[0]
        for col_id in col_ids:
            
            gen_gain, gen_cost = add_generative_error(
                variable_idx = col_id,
                sample_idx = sample_idx,
                data = data,
                dag = dag,
                residual_distributions=residual_distributions,
                residuals = residuals,
                data_max=data_max,
                data_min = data_min,
                assignment=assignment,
                functions=functions,
                sampler=random_sampler,
                setting=meas_setting,
            )

            if not list(dag.successors(col_id)):
                feature_scores[sample_idx,col_id] = gen_gain
                continue

            meas_gain, meas_cost, _ = add_measurement_error(
                variable_idx = col_id,
                sample_idx = sample_idx,
                data = data,
                dag = dag,
                residual_distributions=residual_distributions,
                residuals = residuals,
                data_max=data_max,
                data_min = data_min,
                assignment=assignment,
                functions=functions,
                sampler=random_sampler,
                setting = meas_setting
            )

           

            if meas_cost < gen_cost:
                classifications[sample_idx,col_id] = 1

            confidences[sample_idx,col_id] = abs(meas_cost-gen_cost)
            

            feature_scores[sample_idx,col_id] = max(gen_gain,meas_gain)

    
    if labels is not None:
        for sample_idx in labels:
                if sample_idx not in assignment:
                   
                    for col_id in range(d):

                        gen_gain, gen_cost = add_generative_error(
                            variable_idx = col_id,
                            sample_idx = sample_idx,
                            data = data,
                            dag = dag,
                            residual_distributions=residual_distributions,
                            data_max=data_max,
                            data_min = data_min,
                            assignment=assignment,
                            functions=functions,
                            sampler=random_sampler,
                            residuals=residuals,
                            setting=meas_setting
                        )

                        
                        if not list(dag.successors(col_id)):
                            feature_scores[sample_idx,col_id] = gen_gain
                            continue
                        meas_gain, meas_cost, _ = add_measurement_error(
                            variable_idx = col_id,
                            sample_idx = sample_idx,
                            data = data,
                            dag = dag,
                            residual_distributions=residual_distributions,
                            data_max=data_max,
                            data_min = data_min,
                            assignment=assignment,
                            functions=functions,
                            sampler=random_sampler,
                            residuals=residuals,
                            setting = meas_setting
                        )

                       
                        if meas_cost < gen_cost:
                            classifications[sample_idx,col_id] = 1

                        confidences[sample_idx,col_id] = abs(meas_cost-gen_cost)
                        

                        feature_scores[sample_idx,col_id] = max(gen_gain,meas_gain)
    # For the remaining scores we only calculate generative gains 
    final_mdl_scores = np.zeros_like(feature_scores)


    residuals = np.nan_to_num(
        residuals,
        nan=0.0,
        posinf=np.finfo(residuals.dtype).max,
        neginf=np.finfo(residuals.dtype).min
    )

    for i in range(d):
        
        final_mdl_scores[:,i] = -residual_distributions[i].ll(residuals[:,i]) - uniform_cost(a=data_min[i],b=data_max[i])
      

    unassigned_mask = feature_scores == 0
    
    feature_scores[unassigned_mask] = final_mdl_scores[unassigned_mask]

    return assignment, feature_scores, classifications, confidences 
 

 
