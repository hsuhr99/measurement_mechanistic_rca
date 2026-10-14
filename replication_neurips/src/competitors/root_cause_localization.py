import numpy as np
import shap
from typing import Callable, Optional
from dowhy.gcm import RescaledMedianCDFQuantileScorer
from dowhy.gcm.anomaly_scorer import AnomalyScorer
from dowhy.graph import node_connected_subgraph_view, is_root_node
# from .cali import cali
from ..cali.cali import regress_dag, calculate_residuals, fit_dists
# from .scm_shap import QGAM_SCM
from pyod.models.auto_encoder import AutoEncoder
import torch 
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.neighbors import NearestNeighbors
from .M2OE.explainers.AETabularMMmd_ae import TabularMM as M2OEExplainer
import tensorflow as tf
import pandas as pd
import dice_ml


class RCAExplainer:
    def __init__(self):
        pass 

    def fit(self,dag,data):
        pass 

    def localize(self,dag,data):
        pass



class ModelWrapper():
        def __init__(self,ae):
            self.ae = ae

        def predict(self,X_input):
            return self.ae.decision_function(X_input)
        


class SHAPExplainer(RCAExplainer):

    def __init__(self,model=None,n=100):
        self.n = n
        self.model =AutoEncoder() if model is None else model

    def fit(self,dag,data):
        
        self.model.fit(data)
        model_wrapper = ModelWrapper(self.model)  
        background = shap.sample(data, min(self.n, data.shape[0]))
    
        self.explainer = shap.KernelExplainer(model_wrapper.predict, background)

    def localize(self,data):
        return self.explainer.shap_values(data,nsamples=self.n)


class SCORE_ORDER(RCAExplainer):

    def __init__(self,it_anom_scorer: Callable[[], AnomalyScorer] = RescaledMedianCDFQuantileScorer):
        
        self.it_anom_scorer = it_anom_scorer

    def fit(self, data,dag=None, anomaly_scorer: Callable[[], AnomalyScorer] = RescaledMedianCDFQuantileScorer):
        
      
        
        self.scorers = []

        for node in range(data.shape[1]):
            
            tmp_anomaly_scorer = self.it_anom_scorer()
            tmp_anomaly_scorer.fit(data[:,node])
            self.scorers.append(tmp_anomaly_scorer)


    def localize(self,data):
        
        feature_scores = np.zeros_like(data)
        
        for node in range(data.shape[1]):
            
            feature_scores[:,node] =  self.scorers[node].score(data[:,node])
        
        return feature_scores


class SMOOTH_TRAVERSAL(RCAExplainer):

    def __init__(self,it_anom_scorer: Callable[[], AnomalyScorer] = RescaledMedianCDFQuantileScorer):
        self.it_anom_scorer = it_anom_scorer    

    def fit(self,data,dag):
        
        self.dag = dag 

        self.scorers = []


        for node in range(data.shape[1]):
            
            tmp_anomaly_scorer = self.it_anom_scorer()
            tmp_anomaly_scorer.fit(data[:,node])
            self.scorers.append(tmp_anomaly_scorer)

    
    def localize(self,data):
        all_scores = np.zeros_like(data)

        for node in self.dag.nodes:
            all_scores[:,node] = self.scorers[node].score(data[:,node])
        feature_scores = np.zeros_like(data)
        # print(labels)
        for node in self.dag.nodes:
            if is_root_node(self.dag, node):
                feature_scores[:,node] = all_scores[:,node]
            else:
                gaps = np.array([(all_scores[:,node] - all_scores[:,parent]) for parent in self.dag.predecessors(node)])
                # print(gaps.shape)

                feature_scores[:,node] = np.min(gaps,axis=0)
            # print(F"Recalculating max")
        feature_scores = np.maximum(feature_scores,0.0)
        
        return feature_scores


class SCM_LOCALIZER(RCAExplainer):

    def fit(self,data,dag,regression_settings={
            "mode":"default",
            "scale":False,
            "loss":"L1",
            "discrete":False
        }):
         
        parent_child_pairs = [(child, list(dag.predecessors(child))) for child in dag.nodes]
        self.functions = regress_dag(parent_child_pairs=parent_child_pairs,data=data,regression_settings=regression_settings)

        
        residuals = calculate_residuals(self.functions, data)
  
        # 5. Fit distributions to residuals and wrap
        self.residual_distributions = fit_dists(residuals)

    
    def localize(self,data):

        feature_scores = np.zeros_like(data)

        n, d = data.shape
        residuals = calculate_residuals(self.functions, data)
        
        for i in range(d):

            feature_scores[:,i] = -self.residual_distributions[i].ll(residuals[:,i])


        return feature_scores
    


class SCM_SHAP_LOCALIZER(RCAExplainer):
    
    def __init__(self, n=100):
        self.n = n
    
    def fit(self, data, dag,regression_settings={
            "mode":"default",
            "scale":True,
            "loss":"L1",
            "discrete":False
        }):


        self.model = QGAM_SCM(dag)
        self.model.fit(data)
        self.background = shap.sample(data, min(self.n, data.shape[0]))
        self.explainer = shap.KernelExplainer(self.model.predict, self.background)


    def localize(self,data):
        
        shap_values = self.explainer.shap_values(data, nsamples=self.n)
        return shap_values
        # score =  


class AE_REC(RCAExplainer):
    def __init__(self,epochs=100,batch_size=32):
        self.epochs = epochs
        self.batch_size = batch_size
      
    def fit(self,data,dag=None):
        
        self.clf = AutoEncoder(epoch_num=self.epochs,batch_size=self.batch_size)
        self.clf.fit(data)

    def localize(self,eval_data):

        X_torch = torch.tensor(eval_data, dtype=torch.float32).to(self.clf.device)

        # Disable gradient tracking
        self.clf.model.eval()
        with torch.no_grad():
            X_pred = self.clf.model(X_torch).cpu().numpy()

        feature_errors = np.abs(eval_data - X_pred)  # or (X - X_pred)**2
        
        return feature_errors

# class COUNTERFACTUAL:

#     def __init__(self):
#         pass 


class DICE(RCAExplainer):

    def __init__(self):
        pass 

    def fit(self,data,alpha):

        n, d = data.shape
        feature_names = [f"F{i}" for i in range(d)]
        X_df = pd.DataFrame(data, columns=feature_names, dtype=float)

       

        ae_model = AutoEncoder()
        ae_model.fit(X=data)

        # 1. Compute reconstruction errors
        recon_errors = ae_model.decision_function(data)

        # 2. Sweep thresholds to maximize F1
        threshold = np.quantile(recon_errors,1-alpha)

        y_true = (recon_errors >= threshold).astype(bool)

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

        model_wrapper = AEClassifierWrapper(ae_model, threshold)

        # 4. Setup DiCE
         # 4. Setup DiCE
        data_for_dice = X_df.copy()
        data_for_dice["target"] = y_true
        dice_data = dice_ml.Data(
            dataframe=data_for_dice,
            continuous_features=feature_names,
            outcome_name="target"
        )
        dice_model = dice_ml.Model(model=model_wrapper, backend="sklearn")
        self.explainer = dice_ml.Dice(dice_data, dice_model)

        return #data[y_true]
        

    def localize(self,data):
        n,d = data.shape
        feature_names = [f"F{i}" for i in range(d)]

        results = np.zeros_like(data)
        X_df = pd.DataFrame(data, columns=feature_names, dtype=float)
        
        for i, _ in enumerate(data):

            query_instance = X_df.iloc[[i]]
            
            try:
                explanation = self.explainer.generate_counterfactuals(
                    query_instance,
                    total_CFs=1,
                    desired_class=0  # flip anomaly -> normal
                )
                cf_df = explanation.cf_examples_list[0].final_cfs_df.drop(columns=["target"])
                differences = np.abs(query_instance.values - cf_df.values).flatten()
                results[i] = differences
                

            except Exception as e:
                print(f"DICE EXCEPTION {e}")
                results[i] = np.zeros(data.shape[1])

        return results

class M2OE(RCAExplainer):

    def __init__(self,k_neighbors=100, loss_weights=[1.0, 1.0, 0.3], lr=1e-4, epochs=500, batch_size=32):

        self.k_neighbors = k_neighbors 
        self.loss_weights = loss_weights 
        self.lr = lr 
        self.epochs = epochs 
        self.batch_size = batch_size
        
        pass 

    def fit(self,data,dag=None):
        n, d = data.shape
        train_data = data#data[list(set(range(n)).difference(set(labels.keys())))]  # "normal" points

       

        # Fit NN on "normal" data
        self.nn = NearestNeighbors(n_neighbors=self.k_neighbors)
        self.nn.fit(train_data)

        self.train_data  = data
        pass 

    def localize(self,eval_data):
        in_shape = (eval_data.shape[1],)
        choices = np.zeros_like(eval_data)
        for i, x in enumerate(eval_data):
       
            # Local neighborhood
          
            _, neigh_idx =self. nn.kneighbors(x.reshape(1, -1))
            x_normal = self.train_data[neigh_idx[0]]

            # Build mm_data
            # print(f"Reached")
            mm_data_o = np.full_like(x_normal, fill_value=x)
            mm_data_i = x_normal
            # print(f"Reached")
            # Build explainer for this sample
            explainer = M2OEExplainer(in_shape, self.train_data, self.loss_weights)
            
            explainer.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr),
                # run_eagerly=True
            )
        
            # Train explainer locally
            explainer.fit([mm_data_o, mm_data_i], mm_data_i, batch_size=self.batch_size, epochs=self.epochs, verbose=0)

            # Run explanation
            choose = explainer.explain([mm_data_o, mm_data_i], threshold=0.0, combine=False)

            # IMPORTANT: keep continuous scores instead of thresholded binary mask
            if tf.is_tensor(choose):
                choose = choose.numpy().astype(np.float32)

            choices[i] = choose.squeeze()

        return choices



def run_root_cause_localization(data = np.array, dag = None, rca_explainer = None, anoamly_labels = np.array, eval_indices = np.array,params={}):
    

    explainer = rca_explainer(**params)

    if rca_explainer.__name__ == "DICE":
        explainer.fit(data,alpha=len(eval_indices)/data.shape[0])
    else:
        
        normal_data = data[~anoamly_labels]
        # anomalous_data = data[anoamly_labels]==1
        print("Normal data shape:",normal_data.shape )
        explainer.fit(dag=dag,data=normal_data)
    
    eval_data = data[eval_indices]
    feature_scores = explainer.localize(eval_data)
    print("Feature scores shape:",feature_scores.shape )

    return feature_scores





def full_rca_run(data = np.array, dag = None, rca_explainer = None, eval_indices: np.array=None,ad_model=None,ad_model_params={},rca_params={}):
    
    # Step 1 standardize DATA: 
    if eval_indices is None:
        raise ValueError("eval_indices must be provided for RCA evaluation.")

    scaler =RobustScaler()
    data = scaler.fit_transform(data)

    #Step 2: Get anomaly labels
    ad_model_instance = ad_model(**ad_model_params)
    ad_model_instance.fit(data)
 
    anomaly_scores = ad_model_instance.decision_function(data)
    alpha =len(eval_indices) / data.shape[0]   # fraction of anomalies

    # alpha = 0.01
    print("Anomaly threshold alpha:",alpha)
    threshold = np.quantile(anomaly_scores,1-alpha)
    anoamly_labels = (anomaly_scores >= threshold).astype(bool)

    # Step 3: Run RCA
    feature_scores = run_root_cause_localization(data=data,dag=dag,rca_explainer=rca_explainer,anoamly_labels=anoamly_labels,eval_indices=eval_indices,params=rca_params)

    return feature_scores



def full_realistic_rca_run(data = np.array, dag = None, rca_explainer = None, sample_alpha =0,feature_alpha=0,ad_model=None,ad_model_params={},rca_params={}):
    
    # Step 1 standardize DATA: 
    # if eval_indices is None:
    #     raise ValueError("eval_indices must be provided for RCA evaluation.")

    scaler =RobustScaler()
    data = scaler.fit_transform(data)

    #Step 2: Get anomaly labels
    ad_model_instance = ad_model(**ad_model_params)
    ad_model_instance.fit(data)
 
    anomaly_scores = ad_model_instance.decision_function(data)
   

    # alpha = 0.01
    print("Anomaly threshold alpha:",sample_alpha)
    threshold = np.quantile(anomaly_scores,1-sample_alpha)
    anomaly_labels = (anomaly_scores >= threshold).astype(bool)

    # Step 3: Run RCA
    feature_scores = run_root_cause_localization(data=data,dag=dag,rca_explainer=rca_explainer,anoamly_labels=anomaly_labels,eval_indices=np.where(anomaly_labels)[0],params=rca_params)

    threshold = np.quantile(feature_scores,1-feature_alpha)

    feature_labels = (feature_scores >= threshold).astype(bool)
    global_predictions = np.zeros_like(data,dtype=bool)
    global_predictions[anomaly_labels] = feature_labels

    

    return global_predictions