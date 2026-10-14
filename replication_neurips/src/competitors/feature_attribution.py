from .M2OE.explainers.AETabularMMmd_ae import TabularMM as M2OEExplainer
import numpy as np
import tensorflow as tf
from dice_ml import Dice
import dice_ml

from pyod.models.auto_encoder import AutoEncoder
import numpy as np
import tensorflow as tf
from sklearn.neighbors import NearestNeighbors
from pzflow import Flow
import torch 
import numpy as np
from pyod.models.auto_encoder import AutoEncoder

# from .RootCLAM.data_modules.het_scm import HeterogeneousSCMDataModule
# from .RootCLAM.utils.utils import prepare_vaca
from sklearn.preprocessing import RobustScaler

args = {}
config = {
    "early_stopping":True,
    "trainer":{
        "max_epochs": 100,  # Maximum number of epochs to train
        "min_epochs": 100,  # Min number of epochs to train
        "limit_train_batches": 1.0,
        "limit_val_batches": 1.0,
        "limit_test_batches": 1.0,
        "check_val_every_n_epoch": 2,
        #"progress_bar_refresh_rate": 1,
        #"flush_logs_every_n_steps": 100,
        "log_every_n_steps": 2,  # How often to add logging rows (does not write to disk)
        "precision": 32,
        #"terminate_on_nan": True,
        #"auto_select_gpus": True,
        "deterministic": True,
        #"weights_summary": None,
        #"gpus": None,  # number of gpus to train on (int) or which GPUs to train on (list or str) applied per node
        "num_sanity_val_steps": 2,
        #"track_grad_norm": -1,  # Otherwise tracks that norm (2 for 2-norm)
        "gradient_clip_val": 0.0
    },
    "seed":0,
    "optimizer":{
        "name": "adam",
        "params":
            {
                "lr": 0.0005,
                "betas": [0.9, 0.999],
                "weight_decay": 1.2e-6, # L2 regularization
            }
        },
    "scheduler":{
    "name": "exp_lr",
    "params":
        {"gamma": 0.99},
    },
    "model":{
        "name": "vaca",
        "params":
            {
                "architecture": "dgnn",
                "estimator":"elbo", #multi elbo_iwae, simple iwaedreg, elbo, iwae
                "h_dim_list_dec": [8, 8, 8, 8],
                "h_dim_list_enc": [16, 16],
                "z_dim": 4,
                "distr_z": "normal",
                "dropout_adj_rate": 0.0,
                "dropout_adj_pa_rate": 0.2,
                "dropout_adj_pa_prob_keep_self": 0.0,
                "residual": 1.0,
                "norm_categorical": 0 # False, True
        }
    },
    "root_dir":"test",
    "dataset":{
        "name":"test",
        "params2":{


        }
    }


}

import copy



def get_feature_level_errors(clf, X):
    """
    Compute per-feature reconstruction errors from a trained PyOD AutoEncoder.

    Parameters:
    - clf: fitted PyOD AutoEncoder (PyTorch-based)
    - X: input data (numpy array)

    Returns:
    - feature_errors: numpy array of shape (n_samples, n_features)
    """
    # Convert input to torch tensor and move to device
    X_torch = torch.tensor(X, dtype=torch.float32).to(clf.device)

    # Disable gradient tracking
    with torch.no_grad():
        X_pred = clf.model(X_torch).cpu().numpy()

    feature_errors = np.abs(X - X_pred)  # or (X - X_pred)**2
    
    return feature_errors


def run_autoencoder_feature(train_data: np.ndarray,epochs=10,batch_size=128, true_labels=None):
    
    clf = AutoEncoder(epoch_num=epochs,batch_size=batch_size)
    clf.fit(train_data)

    feature_errors = get_feature_level_errors(clf, train_data)
   
    
    return feature_errors



def run_vaca(data,labels,dag,z=4,enc =[16, 16],dec=  [8, 8, 8, 8], seed=0,threshold=0.01,batch_size=128,use_max=True,max_epochs=500):
    # print(labels)

    n,d = data.shape
    data = RobustScaler().fit_transform(data)
    test_data_indices = np.array(list(labels))
    normal_idx = np.setdiff1d(np.arange(n), test_data_indices, assume_unique=True)
    normal_data = data#data[normal_idx]
    
    
    training_tensor = torch.tensor(normal_data,dtype=torch.float32 ).float()

    test_tensor = torch.tensor(data,dtype=torch.float32).float()
    cfg = copy.deepcopy(config)
    cfg["model"]["params"]["z_dim"] = z
    cfg["model"]["params"]["h_dim_list_dec"] = dec
    cfg["model"]["params"]["h_dim_list_enc"] = enc
    cfg["seed"] = seed
    cfg["trainer"]["max_epochs"] = max_epochs
   
    data_module = HeterogeneousSCMDataModule(dataset_name="default",num_samples_tr= int(0.9 * normal_data.shape[0]),data=training_tensor.float(),dag=dag,normalize="std",batch_size=batch_size)
    data_module.prepare_data()
    
    model = prepare_vaca(args,cfg,data_module)


    #training_latent =  -np.log(model.get_distribution(training_tensor, data_module, device="cpu"))
    # print(training_latent.shape)
    # threshold_low = np.quantile(training_latent,threshold,axis=0)
    # threshold_high = np.quantile(training_latent,1-threshold,axis=0)
    model.eval()
    with torch.no_grad():
        log_probability_scores = model.get_distribution(test_tensor, data_module, device="cpu")

    
    if np.any(np.isnan(log_probability_scores)):
        print("NaN in porbability scores")
        raise ValueError("NaN in porbability scores")
        
    test_latent = -log_probability_scores
    #outlier_mask = (test_latent <= threshold_low) | (test_latent >= threshold_high)
    #return {test_data_indices[i]:set(np.where(outlier_mask.any(axis=2)[i])[0]) for i in range(len(test_data_indices))}
    if use_max:
        return test_latent.max(axis=2)
    else:
        return test_latent.mean(axis=2)
    # test_mask = 
    return score, train_data, labels



def run_vaca_realistic(data,sample_alpha,feature_alpha,dag,z=4,enc =[16, 16],dec=  [8, 8, 8, 8], seed=0,threshold=0.01,batch_size=128,use_max=True,max_epochs=500):
    # print(labels)

    n,d = data.shape
    
    X = data
    clf = AutoEncoder()
    train_data  = X.astype(np.float32)
    # print(train_data.shape)
    # print(train_data.dtype)
    # print("Training")
    clf.fit(train_data)
    scores = clf.decision_function(train_data)
    threshold = np.quantile(scores,1-sample_alpha)

    y_pred = scores >= threshold
    
    normal_data = data[scores < threshold]
    
    
    training_tensor = torch.tensor(normal_data,dtype=torch.float32 ).float()

    test_tensor = torch.tensor(data[y_pred],dtype=torch.float32).float()
    cfg = copy.deepcopy(config)
    cfg["model"]["params"]["z_dim"] = z
    cfg["model"]["params"]["h_dim_list_dec"] = dec
    cfg["model"]["params"]["h_dim_list_enc"] = enc
    cfg["seed"] = seed
    cfg["trainer"]["max_epochs"] = max_epochs
   
    data_module = HeterogeneousSCMDataModule(dataset_name="default",num_samples_tr= int(0.9 * normal_data.shape[0]),data=training_tensor.float(),dag=dag,normalize="std",batch_size=batch_size)
    data_module.prepare_data()
    
    model = prepare_vaca(args,cfg,data_module)


    #training_latent =  -np.log(model.get_distribution(training_tensor, data_module, device="cpu"))
    # print(training_latent.shape)
    # threshold_low = np.quantile(training_latent,threshold,axis=0)
    # threshold_high = np.quantile(training_latent,1-threshold,axis=0)
    model.eval()
    with torch.no_grad():
        log_probability_scores = model.get_distribution(test_tensor, data_module, device="cpu")

    
    if np.any(np.isnan(log_probability_scores)):
        print("NaN in porbability scores")
        raise ValueError("NaN in porbability scores")
        
    test_latent = -log_probability_scores
    #outlier_mask = (test_latent <= threshold_low) | (test_latent >= threshold_high)
    #return {test_data_indices[i]:set(np.where(outlier_mask.any(axis=2)[i])[0]) for i in range(len(test_data_indices))}
    if use_max:
        feature_scores = test_latent.max(axis=2)
    else:
        feature_scores = test_latent.mean(axis=2)
    # test_mask = 

    feature_threshold = np.quantile(feature_scores,1-feature_alpha)
    return y_pred, feature_scores



def run_m2oe_method(data: np.ndarray, labels, k_neighbors=100, loss_weights=[1.0, 1.0, 0.3], lr=1e-4, epochs=500, batch_size=32):
    """
    Run M2OE (Tabular MM method) in line with the published experimental setup.

    Parameters
    ----------
    data : np.ndarray
        Input feature matrix (n x d).
    labels : dict
        Dictionary mapping anomalous indices to labels.
    k_neighbors : int
        Number of neighbors to sample for local training.
    loss_weights : list
        Weights for loss components.
    lr : float
        Learning rate for optimizer.
    epochs : int
        Training epochs per point.
    batch_size : int
        Training batch size.

    Returns
    -------
    choices : np.ndarray
        Continuous importance scores for each label (no thresholding).
    """

    n, d = data.shape
    train_data = data#data[list(set(range(n)).difference(set(labels.keys())))]  # "normal" points

    in_shape = (data.shape[1],)

    # Container for feature scores
    choices = np.zeros((len(labels), d))

    # Fit NN on "normal" data
    nn = NearestNeighbors(n_neighbors=k_neighbors)
    nn.fit(train_data)
    print("Running M2oe")
    for i, idx in enumerate(labels):
        x = data[idx]

        # Local neighborhood
        print(f"Reached")
        _, neigh_idx = nn.kneighbors(x.reshape(1, -1))
        x_normal = train_data[neigh_idx[0]]

        # Build mm_data
        print(f"Reached")
        mm_data_o = np.full_like(x_normal, fill_value=x)
        mm_data_i = x_normal
        print(f"Reached")
        # Build explainer for this sample
        explainer = M2OEExplainer(in_shape, train_data, loss_weights)
        
        explainer.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0),
            run_eagerly=True
        )
       
        # Train explainer locally
        explainer.fit([mm_data_o, mm_data_i], mm_data_i, batch_size=batch_size, epochs=epochs, verbose=0)

        # Run explanation
        choose = explainer.explain([mm_data_o, mm_data_i], threshold=0.0, combine=False)

        # IMPORTANT: keep continuous scores instead of thresholded binary mask
        if tf.is_tensor(choose):
            choose = choose.numpy().astype(np.float32)

        choices[i] = choose.squeeze()

    return choices



def run_dice(data, labels,  dag=None, n_thresholds=100):
    """
    Generate DiCE counterfactuals for anomalies using an AE + optimal F1 threshold classifier.

    Args:
        data: np.array of shape (n_samples, n_features)
        labels: list/array of anomaly indices
        ae_model: trained autoencoder with .predict() method
        dag: optional, not used here
        n_thresholds: number of thresholds to sweep for F1

    Returns:
        results: np.array of shape (len(labels), n_features) with absolute feature changes
        reconstructed: np.array with CF-reconstructed values
        threshold_opt: the F1-optimal threshold used
    """

    import pandas as pd
    import numpy as np
    import dice_ml
    from sklearn.metrics import f1_score
    if type(labels) == dict:
        labels = list(labels.keys())
    else:
        labels = list(labels)
    n, d = data.shape
    feature_names = [f"F{i}" for i in range(d)]
    X_df = pd.DataFrame(data, columns=feature_names, dtype=float)

    results = np.zeros((len(labels), d))

    ae_model = AutoEncoder()
    ae_model.fit(X=data)

    # 1. Compute reconstruction errors
    recon_errors = ae_model.decision_function(data)

    # 2. Sweep thresholds to maximize F1
    thresholds = np.linspace(recon_errors.min(), recon_errors.max(), n_thresholds)
    y_true = np.zeros(n, dtype=int)
    y_true[labels] = 1

    f1_scores = []
    for t in thresholds:
        y_pred = (recon_errors > t).astype(int)
        f1_scores.append(f1_score(y_true, y_pred))

    idx_best = np.argmax(f1_scores)
    threshold_opt = thresholds[idx_best]

    # 3. Wrap AE + threshold as binary classifier for DiCE
    class AEClassifierWrapper:
        def __init__(self, ae_model, threshold):
            self.ae = ae_model
            self.threshold = threshold
            self.classes_ = np.array([0, 1])

        def predict(self, X_input):
            X_array = X_input.values if hasattr(X_input, "values") else X_input
            errors = self.ae.decision_function(X_array)
            return (errors > self.threshold).astype(int)

        def predict_proba(self, X_input):
            preds = self.predict(X_input)
            proba = np.zeros((len(preds), 2))
            proba[np.arange(len(preds)), preds] = 1.0
            return proba

    model_wrapper = AEClassifierWrapper(ae_model, threshold_opt)

    # 4. Setup DiCE
    data_for_dice = X_df.copy()
    data_for_dice["target"] = y_true
    dice_data = dice_ml.Data(
        dataframe=data_for_dice,
        continuous_features=feature_names,
        outcome_name="target"
    )
    dice_model = dice_ml.Model(model=model_wrapper, backend="sklearn")
    explainer = dice_ml.Dice(dice_data, dice_model)

    # 5. Generate CFs for anomalies
    for i, idx in enumerate(labels):
        query_instance = X_df.loc[[idx]]
        try:
            explanation = explainer.generate_counterfactuals(
                query_instance,
                total_CFs=1,
                desired_class=0  # flip anomaly -> normal
            )
            cf_df = explanation.cf_examples_list[0].final_cfs_df.drop(columns=["target"])
            differences = np.abs(query_instance.values - cf_df.values).flatten()
            results[i] = differences
            

        except Exception as e:
            print(f"DICE EXCEPTION {e}")
            results[i] = np.zeros(d)
           

    return results


import shap
import numpy as np
import pandas as pd
from typing import Callable, Optional



def shap_feature_explainer_per_row(
    data: np.ndarray,
    
    nsamples: int = 100,
    labels = None
) -> pd.DataFrame:
    """
    Explain anomaly scores using SHAP for tabular data per row.

    Parameters
    ----------
    X : np.ndarray
        Input data, shape (n_samples, n_features)
    anomaly_method : Callable
        Black-box function that takes X and returns anomaly scores
    indices : np.ndarray, optional
        Row indices to explain. If None, explain all rows
    nsamples : int
        Number of SHAP samples for KernelExplainer

    Returns
    -------
    pd.DataFrame
        Feature importance per row (absolute SHAP values). Index corresponds
        to rows of X_subset (all rows if indices=None), columns to features.
    """
    # if indices is None:
    #     indices = np.arange(X.shape[0])
    
    X = data
    clf = AutoEncoder()
    train_data  = X.astype(np.float32)
    print(train_data.shape)
    print(train_data.dtype)
    print("Training")
    clf.fit(train_data)
    # print("Fitted")
    # clf.decision_function(train_data)

    class ModelWrapper():
        def __init__(self,ae):
            self.ae = ae

        def predict(self,X_input):
            return self.ae.decision_function(X_input)
    
    
    anomaly_method = ModelWrapper(clf)
    
    if type(labels) == dict:
        X_subset = X[list(labels.keys())]
    else:
        X_subset = X[list(labels)]
    
    # Background sample for KernelExplainer (use small subset for efficiency)
    background = shap.sample(X, min(100, X.shape[0]))
    
    explainer = shap.KernelExplainer(anomaly_method.predict, background)
    shap_values = explainer.shap_values(X_subset, nsamples=nsamples)
    
    # Convert to DataFrame for readability
    
    return np.abs(shap_values)


def shap_feature_explainer_per_row_realistic(
    data: np.ndarray,
    
    nsamples: int = 100,
    sample_alpha=0.05,
    feature_alpha=0.05,
) -> pd.DataFrame:
    """
    Explain anomaly scores using SHAP for tabular data per row.

    Parameters
    ----------
    X : np.ndarray
        Input data, shape (n_samples, n_features)
    anomaly_method : Callable
        Black-box function that takes X and returns anomaly scores
    indices : np.ndarray, optional
        Row indices to explain. If None, explain all rows
    nsamples : int
        Number of SHAP samples for KernelExplainer

    Returns
    -------
    pd.DataFrame
        Feature importance per row (absolute SHAP values). Index corresponds
        to rows of X_subset (all rows if indices=None), columns to features.
    """
    # if indices is None:
    #     indices = np.arange(X.shape[0])
    
    X = data
    clf = AutoEncoder()
    train_data  = X.astype(np.float32)
    # print(train_data.shape)
    # print(train_data.dtype)
    # print("Training")
    clf.fit(train_data)
    scores = clf.decision_function(train_data)
    threshold = np.quantile(scores,1-sample_alpha)


    # Select threshold based on quantile 
    # print("Fitted")
    # clf.decision_function(train_data)

    class ModelWrapper():
        def __init__(self,ae):
            self.ae = ae

        def predict(self,X_input):
            return self.ae.decision_function(X_input)
    
    
    anomaly_method = ModelWrapper(clf)
    
    X_test = X[scores>=threshold]
    
    # Background sample for KernelExplainer (use small subset for efficiency)
    background = shap.sample(X[scores<threshold], min(100, X[scores<threshold].shape[0]))
    
    explainer = shap.KernelExplainer(anomaly_method.predict, background)
    shap_values = explainer.shap_values(X_test, nsamples=nsamples)

    feature_threshold = np.quantile(shap_values,1-feature_alpha)

    sample_y = scores >= threshold
    feature_y = shap_values >= feature_threshold
    
    # Convert to DataFrame for readability
    
    return sample_y, shap_values





def run_m2oe_method_realistic(data: np.ndarray,sample_alpha=0.05,feature_alpha=0.05,  k_neighbors=100, loss_weights=[1.0, 1.0, 0.3], lr=1e-4, epochs=500, batch_size=32):
    """
    Run M2OE (Tabular MM method) in line with the published experimental setup.

    Parameters
    ----------
    data : np.ndarray
        Input feature matrix (n x d).
    labels : dict
        Dictionary mapping anomalous indices to labels.
    k_neighbors : int
        Number of neighbors to sample for local training.
    loss_weights : list
        Weights for loss components.
    lr : float
        Learning rate for optimizer.
    epochs : int
        Training epochs per point.
    batch_size : int
        Training batch size.

    Returns
    -------
    choices : np.ndarray
        Continuous importance scores for each label (no thresholding).
    """

    n, d = data.shape
    train_data = data#data[list(set(range(n)).difference(set(labels.keys())))]  # "normal" points

   
     
    X = data
    clf = AutoEncoder()
    train_data  = X.astype(np.float32)
    # print(train_data.shape)
    # print(train_data.dtype)
    # print("Training")
    clf.fit(train_data)
    scores = clf.decision_function(train_data)
    threshold = np.quantile(scores,1-sample_alpha)

    clean_data = data[scores < threshold]
    in_shape = (data.shape[1],)
    # Container for feature scores
    choices = np.zeros(((scores >= threshold).sum(), d))

    # Fit NN on "normal" data
    nn = NearestNeighbors(n_neighbors=k_neighbors)
    nn.fit(clean_data)

    for i, idx in enumerate(np.where(scores >= threshold)[0]):
        x = data[idx]

        # Local neighborhood
        _, neigh_idx = nn.kneighbors(x.reshape(1, -1))
        x_normal = train_data[neigh_idx[0]]

        # Build mm_data
        mm_data_o = np.full_like(x_normal, fill_value=x)
        mm_data_i = x_normal

        # Build explainer for this sample
        explainer = M2OEExplainer(in_shape, train_data, loss_weights)
        explainer.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0),
            run_eagerly=True
        )

        # Train explainer locally
        explainer.fit([mm_data_o, mm_data_i], mm_data_i, batch_size=batch_size, epochs=epochs, verbose=0)

        # Run explanation
        choose = explainer.explain([mm_data_o, mm_data_i], threshold=0.0, combine=False)

        # IMPORTANT: keep continuous scores instead of thresholded binary mask
        if tf.is_tensor(choose):
            choose = choose.numpy().astype(np.float32)

        choices[i] = choose.squeeze()

    feature_threshold = np.quantile(choices,1-feature_alpha)

    sample_y = scores>=threshold
    feature_y = choices >= feature_threshold


    return sample_y, choices



def run_dice_realistic(data, sample_alpha=0.05,feature_alpha=0.05,  dag=None, n_thresholds=100):
    """
    Generate DiCE counterfactuals for anomalies using an AE + optimal F1 threshold classifier.

    Args:
        data: np.array of shape (n_samples, n_features)
        labels: list/array of anomaly indices
        ae_model: trained autoencoder with .predict() method
        dag: optional, not used here
        n_thresholds: number of thresholds to sweep for F1

    Returns:
        results: np.array of shape (len(labels), n_features) with absolute feature changes
        reconstructed: np.array with CF-reconstructed values
        threshold_opt: the F1-optimal threshold used
    """

    import pandas as pd
    import numpy as np
    import dice_ml
    from sklearn.metrics import f1_score
   
    n, d = data.shape
    feature_names = [f"F{i}" for i in range(d)]
    X_df = pd.DataFrame(data, columns=feature_names, dtype=float)


    X = data
    clf = AutoEncoder()
    train_data  = X.astype(np.float32)
    # print(train_data.shape)
    # print(train_data.dtype)
    # print("Training")
    clf.fit(train_data)
    scores = clf.decision_function(train_data)
    threshold = np.quantile(scores,1-sample_alpha)

    y_pred = scores >= threshold
    results = np.zeros((y_pred.sum(), d))

    


    # 3. Wrap AE + threshold as binary classifier for DiCE
    class AEClassifierWrapper:
        def __init__(self, ae_model, threshold):
            self.ae = ae_model
            self.threshold = threshold
            self.classes_ = np.array([0, 1])

        def predict(self, X_input):
            X_array = X_input.values if hasattr(X_input, "values") else X_input
            errors = self.ae.decision_function(X_array)
            return (errors > self.threshold).astype(int)

        def predict_proba(self, X_input):
            preds = self.predict(X_input)
            proba = np.zeros((len(preds), 2))
            proba[np.arange(len(preds)), preds] = 1.0
            return proba

    model_wrapper = AEClassifierWrapper(clf, threshold)

    # 4. Setup DiCE
    data_for_dice = X_df.copy()
    data_for_dice["target"] = y_pred
    dice_data = dice_ml.Data(
        dataframe=data_for_dice,
        continuous_features=feature_names,
        outcome_name="target"
    )
    dice_model = dice_ml.Model(model=model_wrapper, backend="sklearn")
    explainer = dice_ml.Dice(dice_data, dice_model)

    # 5. Generate CFs for anomalies
    for i, idx in enumerate(np.where(y_pred)[0]):
        query_instance = X_df.loc[[idx]]
        try:
            explanation = explainer.generate_counterfactuals(
                query_instance,
                total_CFs=1,
                desired_class=0  # flip anomaly -> normal
            )
            cf_df = explanation.cf_examples_list[0].final_cfs_df.drop(columns=["target"])
            differences = np.abs(query_instance.values - cf_df.values).flatten()
            results[i] = differences
            

        except Exception as e:
            print(f"DICE EXCEPTION {e}")
            results[i] = np.zeros(d)

    feature_threshold = np.quantile(results,1-feature_alpha)

    y_feature = results >= feature_threshold           

    return y_pred, results


import shap
import numpy as np
import pandas as pd
from typing import Callable, Optional



def shap_feature_explainer_per_row(
    data: np.ndarray,
    
    nsamples: int = 100,
    labels = None
) -> pd.DataFrame:
    """
    Explain anomaly scores using SHAP for tabular data per row.

    Parameters
    ----------
    X : np.ndarray
        Input data, shape (n_samples, n_features)
    anomaly_method : Callable
        Black-box function that takes X and returns anomaly scores
    indices : np.ndarray, optional
        Row indices to explain. If None, explain all rows
    nsamples : int
        Number of SHAP samples for KernelExplainer

    Returns
    -------
    pd.DataFrame
        Feature importance per row (absolute SHAP values). Index corresponds
        to rows of X_subset (all rows if indices=None), columns to features.
    """
    # if indices is None:
    #     indices = np.arange(X.shape[0])
    
    X = data
    clf = AutoEncoder()
    train_data  = X.astype(np.float32)
    print(train_data.shape)
    print(train_data.dtype)
    print("Training")
    clf.fit(train_data)
    # print("Fitted")
    # clf.decision_function(train_data)

    class ModelWrapper():
        def __init__(self,ae):
            self.ae = ae

        def predict(self,X_input):
            return self.ae.decision_function(X_input)
    
    
    anomaly_method = ModelWrapper(clf)
    
    if type(labels) == dict:
        X_subset = X[list(labels.keys())]
    else:
        X_subset = X[list(labels)]
    
    # Background sample for KernelExplainer (use small subset for efficiency)
    background = shap.sample(X, min(100, X.shape[0]))
    
    explainer = shap.KernelExplainer(anomaly_method.predict, background)
    shap_values = explainer.shap_values(X_subset, nsamples=nsamples)
    
    # Convert to DataFrame for readability
    
    return np.abs(shap_values)



import numpy as np
import pandas as pd
import shap

# ---- Wrappers for anomaly models ----
class AEWrapper:
    def __init__(self, ae):
        self.ae = ae

    def predict(self, X_input):
        return self.ae.decision_function(X_input.astype(np.float32))

class NFWrapper:
    def __init__(self, flow):
        self.flow = flow

    def predict(self, X_input):
        # NF returns log-probs, anomaly score = -log_prob
        X_df = pd.DataFrame(X_input.astype(np.float32))
        return -self.flow.log_prob(X_df)


# ---- Unified SHAP explainer ----
def shap_feature_explainer(
    data: np.ndarray,
    model_type: str = "AE",  # "AE" or "NF"
    model=None,              # Pretrained AE or NF model (optional)
    nsamples: int = 100,
    sample_alpha: float = 0.05,
    feature_alpha: float = 0.05,
    realistic: bool = False, 
    labels=None
):
    """
    Explain anomaly scores using SHAP for tabular data per row.

    Parameters
    ----------
    data : np.ndarray
        Input data, shape (n_samples, n_features)
    model_type : str
        "AE" (AutoEncoder) or "NF" (Normalizing Flow)
    model : object, optional
        Pretrained AE or NF model. If None, will train inside function.
    nsamples : int
        Number of SHAP samples for KernelExplainer
    sample_alpha : float
        Quantile for anomaly thresholding (used if realistic=True)
    feature_alpha : float
        Quantile for feature-level importance threshold (used if realistic=True)
    realistic : bool
        If True, only explains anomalies above threshold.

    Returns
    -------
    If realistic=False:
        np.ndarray of absolute SHAP values (n_samples, n_features)
    If realistic=True:
        tuple (sample_y, shap_values) where
            sample_y : boolean mask of anomalous samples
            shap_values : SHAP attribution array
    """

    X = data.astype(np.float32)

    # ---- Train/Load model ----
    if model_type == "AE":
        if model is None:
            from pyod.models.auto_encoder import AutoEncoder
            model = AutoEncoder()
            model.fit(X)
        wrapper = AEWrapper(model)
        scores = model.decision_function(X)

    elif model_type == "NF":
        if model is None:
            # Train new NF
            columns = list(range(X.shape[1]))
         
            model = Flow(columns)
            model.train(pd.DataFrame(X), epochs=10, batch_size=32, verbose=False)
        wrapper = NFWrapper(model)
        scores = wrapper.predict(X)

    else:
        raise ValueError("model_type must be 'AE' or 'NF'")

    # ---- SHAP explanation ----
    if realistic:
        threshold = np.quantile(scores, 1 - sample_alpha)
        X_test = X[scores >= threshold]
        background = shap.sample(X[scores < threshold], min(100, max(1, (scores < threshold).sum())))
        explainer = shap.KernelExplainer(wrapper.predict, background)
        shap_values = explainer.shap_values(X_test, nsamples=nsamples)

        feature_threshold = np.quantile(shap_values, 1 - feature_alpha)
        sample_y = scores >= threshold
        return sample_y, shap_values >= feature_threshold

    else:
        background = shap.sample(X, min(100, X.shape[0]))
        explainer = shap.KernelExplainer(wrapper.predict, background)
        shap_values = explainer.shap_values(X[list(labels)], nsamples=nsamples)
        return np.abs(shap_values)
