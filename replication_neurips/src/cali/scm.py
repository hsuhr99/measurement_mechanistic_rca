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


    def predict(self,X,regression_settings={
            "mode":"default",
            "scale":True,
            "loss":"L1",
            "discrete":False
        }):

        mdl_scores = np.zeros_like(X)
        X_resid = calculate_residuals(self.functions,X)
        
        for i,dist in enumerate(self.residual_distributions):
            mdl_scores[:,i] = dist.mdl_score_sample(X_resid[:,i])

        return np.mean(mdl_scores,axis=1)
    
def scm_explain(data,dag,labels=None,n_samples=100,sample_alpha=0.05,feature_alpha=0.05,regression_settings={
            "mode":"default",
            "scale":True,
            "loss":"L1",
            "discrete":False
        }):

    
    parent_child_pairs = [(child, list(dag.predecessors(child))) for child in dag.nodes]
    functions = regress_dag(parent_child_pairs=parent_child_pairs,data=data,regression_settings=regression_settings)

    residuals = calculate_residuals(self.functions, data)
    
    
    # 5. Fit distributions to residuals and wrap
    residual_distributions = fit_dists(self.residuals)
    