from .cali import regress_dag, calculate_residuals, fit_dists
import numpy as np
import shap 
class QGAM_SCM:
    def __init__(self,dag):

        self.dag = dag 


    def fit(self,data,regression_settings={
            "mode":"default",
            "scale":True,
            "loss":"L1",
            "discrete":False
        }):

        self.parent_child_pairs = [(child, list(self.dag.predecessors(child))) for child in self.dag.nodes]
        self.functions = regress_dag(parent_child_pairs= self.parent_child_pairs,data=data,regression_settings=regression_settings)

        self.residuals = calculate_residuals(self.functions, data)
       
        
        # 5. Fit distributions to residuals and wrap
        self.residual_distributions = fit_dists(self.residuals)


    def predict(self,X):

        mdl_scores = np.zeros_like(X)
        X_resid = calculate_residuals(self.functions,X)
        
        for i,dist in enumerate(self.residual_distributions):
            mdl_scores[:,i] = dist.mdl_score_sample(X_resid[:,i])

        return np.mean(mdl_scores,axis=1)
    
def shap_qgam_scm(data,dag,labels=None,n_samples=100,sample_alpha=0.05,feature_alpha=0.05):

    

    model = QGAM_SCM(dag)
    model.fit(data)

    if labels is not None:
        test_data = data[list(labels)]
    else:
        scores = model.predict(data)
        outliers = scores >= np.quantile(scores,1-sample_alpha)
        test_data = data[outliers]
    if labels:
        background = shap.sample(data, min(100, data.shape[0]))
    else:
        background = shap.sample(data[~outliers], min(100, data.shape[0]))
    
    explainer = shap.KernelExplainer(model.predict, background)
    shap_values = explainer.shap_values(test_data, nsamples=n_samples)
    
    # Convert to DataFrame for readability
    if labels is not None:
        return np.abs(shap_values)
    else:
        return outliers,np.abs(shap_values)

    