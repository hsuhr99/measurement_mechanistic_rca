import networkx as nx
import numpy as np
from scipy.stats import norm
from sklearn.mixture import GaussianMixture

from .distributions import RobustKDE
from .robust_regression.utils import logg, robust_z_scores




def markov_blanket(G: nx.DiGraph, node):
    parents = set(G.predecessors(node))
    children = set(G.successors(node))
    co_parents = set()
    for child in children:
        co_parents.update(G.predecessors(child))
    co_parents.discard(node)
    return parents | children | co_parents

def uniform_cost(a,b,precision=1):
    return logg(b-a) - logg(precision)

def is_outlier_marg(data: np.array, alpha: float):
    n = data.shape[0]

    threshold = norm.ppf(1 - alpha)
    z_scores = robust_z_scores(data)
    return z_scores >= threshold

class ConstantFunction:
    def __init__(self, X):
        self.value = np.median(X)
        self.X = X

    def model_complexity(self):
        return 0
    
    def predict(self, X):
        return np.full(shape=(X.shape[0],), fill_value=self.value)
    def get_residuals(self,X,y):
        return y - self.predict(X)
    
    def get_scaled_residuals(self,X,y):
        return self.get_residuals(X,y)
   
    def corrected_alpha_global(alpha: float, test_count) -> float:
        corrected_alpha =  1 - (1 - alpha) ** (1 / test_count)
    
        return corrected_alpha

def calculate_samples_needed(alpha_corrected, min_tail_samples=10):
    """
    Calculate the number of samples needed to reliably estimate the distribution tail.

    Parameters:
    - alpha_corrected: float, corrected significance level (e.g., 0.001)
    - min_tail_samples: int, desired expected number of tail samples (default 10)

    Returns:
    - samples_per_node: int, recommended sample size per node/residual
    """
    samples = int(np.ceil(min_tail_samples / alpha_corrected))
    return max(samples, 100)  # Ensure a minimum to avoid too small samples

def compute_surprise_thresholds(gmm: GaussianMixture, sample_multiplier: int, alpha: float) -> float:
    """
    Computes the surprise threshold (-log₂ p(x)) for outlier detection from a fitted GMM.

    Args:
        gmm: A fitted sklearn GaussianMixture.
        sample_multiplier: Number of samples to draw from the GMM for estimating the surprise distribution.
        alpha: Corrected alpha value used to determine the (1 - alpha) quantile threshold.

    Returns:
        threshold_surprise: The quantile value of -log₂(p(x)) used as an outlier threshold.
    """
    samples = gmm.sample(sample_multiplier)[0]
    log_probs = gmm.score_samples(samples.reshape(-1, 1))  # log_e p(x)
    surprises = -log_probs / np.log(2)  # convert to log₂(p(x))
    return np.quantile(surprises, 1 - alpha)





def fit_dists(data:np.array):
    n,d = data.shape
    distributions = []
    for i in range(d):
        robust_dist = RobustKDE()
        robust_dist.fit(data[:, i][~np.isnan(data[:, i])].flatten())
        distributions.append(robust_dist)


    return distributions
