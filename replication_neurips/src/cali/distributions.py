import math

import numpy as np
from scipy.stats import norm
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KernelDensity

from .robust_regression.utils import arbitrary_threshold_cost, universalIntegerLength



class DistributionBase:
    def is_outlier(self, X: np.ndarray) -> np.ndarray:
        """Return boolean mask for outliers."""
        raise NotImplementedError
    
    def mdl_score(self, X: np.ndarray) -> float:
        """Compute MDL or penalized likelihood score."""
        raise NotImplementedError

class SurpriseDistribution(DistributionBase):
    def __init__(self, gmm: GaussianMixture):
        """
        Encapsulates tail-probability-based outlier detection using a univariate GMM.

        Args:
            gmm: A fitted sklearn GaussianMixture model.
            alpha: Significance level for outlier detection (e.g., 0.01 for 1%).
        """
        self.gmm = gmm
        self.compute_outlier_bounds()

    def compute_tail_probability(self, X: np.ndarray) -> np.ndarray:
        """
        Compute two-sided tail probability for each input value.
        Tail probability is defined as:
            min(P(X <= x), P(X >= x)) = min(CDF(x), 1 - CDF(x))

        Returns:
            Tail probabilities, one per input point.
        """
        X = X.reshape(-1, 1)
        cdf_values = self._gmm_cdf(X)
        tail_probs = 2*np.minimum(cdf_values, 1 - cdf_values)
        return tail_probs 

    def _gmm_cdf(self, X: np.ndarray) -> np.ndarray:
        """
        Compute the CDF of a Gaussian Mixture Model at given points.

        Returns:
            CDF values at input points.
        """
        weights = self.gmm.weights_
        means = self.gmm.means_.flatten()
        stds = np.sqrt(self.gmm.covariances_).flatten()

        # For each component, compute its weighted CDF contribution
        cdf_vals = np.zeros_like(X, dtype=np.float64)
        for w, mu, sigma in zip(weights, means, stds):
            cdf_vals += w * norm.cdf(X, loc=mu, scale=sigma)
        return cdf_vals.flatten()

    def mdl_score_sample(self,x: np.ndarray, precision=1) -> float:
        if (not 1 in x.shape) or x.shape[1] == 0:
            x = x.reshape((x.shape[0],1))
        log_likelihood_nat = self.gmm.score_samples(x)
        log_likelihood_bits = log_likelihood_nat / np.log(2) #+ np.log2(precision)
        score = -log_likelihood_bits

        return score

    def mdl_score(self, X: np.ndarray, precision=1) -> float:
        """
        Computes the MDL score (in bits) for the observed X under the GMM model.
        """
      
        n = X.shape[0]
    
        log_likelihood_nat = self.gmm.score_samples(X.reshape(-1, 1)).sum()
        log_likelihood_bits = log_likelihood_nat / np.log(2) #+ n * np.log2(precision)
        score = -log_likelihood_bits

        # Add parameter cost
        weights, means, variances = extract_gmm_params(self.gmm)
        score += universalIntegerLength(self.gmm.n_components)
        for i, (w, mu, sigma) in enumerate(zip(weights, means, variances)):
            if i > 0:
                score += arbitrary_threshold_cost(w, -np.log10(precision))
            score += arbitrary_threshold_cost(mu, -np.log10(precision))
            score += arbitrary_threshold_cost(sigma, -np.log10(precision))

        return score
    
class RobustGaussian(DistributionBase):
    """
    Robust Gaussian distribution estimated via median and MAD for residuals.
    No explicit fallback mixture — fallback handled externally.
    """

    def __init__(self):
       
        self.mean = None
        self.std = None
       

    def fit(self, residuals: np.ndarray):
        """
        Robustly estimate location and scale from data using median and MAD.
        """
        self.mean = np.median(residuals)
        mad = np.median(np.abs(residuals - self.mean))
        # Convert MAD to std estimate assuming normality
        self.std = mad * 1.4826  

        if self.std == 0:  # fallback to sample std if mad=0 (rare)
            self.std = np.std(residuals)

    def sample(self,n):

        return np.random.normal(loc=self.mean,scale=self.std,size=n)
    
    def ll(self,x:np.array):
        if len(x.shape) < 2  or (x.shape[1] == 0):
                x = x.reshape((x.shape[0],1))
        ll = norm.logpdf(x, loc=self.mean, scale=self.std) / np.log(2)

        return ll[:,0]
    
def kde_mode(kde, data, grid_size=2000, margin=3.0):
    """
    Estimate mode (argmax) of a univariate KDE.
    """
    xmin, xmax = np.min(data), np.max(data)
    span = xmax - xmin
    xs = np.linspace(xmin - margin * span,
                     xmax + margin * span,
                     grid_size)

    dens = kde(xs)
    return xs[np.argmax(dens)]

class RobustKDE(DistributionBase):
    """
    Robust univariate density estimated via Gaussian KDE
    with IQR-based bandwidth for outlier robustness.
    """

    def __init__(self):
        self.alpha = 0.01
        self.kde = None
        self.scale = None

    def fit(self, residuals: np.ndarray):
        """
        Fit KDE with robust bandwidth selection.
        """
        self.mean = np.median(residuals)
        self.kde = KernelDensity(kernel='gaussian',bandwidth="silverman")
       
        distances_r = np.abs(residuals - self.mean)
        keep_idx_r = np.argpartition(distances_r, int(len(residuals)*(1-self.alpha)))[:int(len(residuals)*(1-self.alpha))]
        res_trim = residuals[keep_idx_r]
        self.kde.fit(res_trim.reshape(-1, 1))

    def sample(self, n: int):
        """
        Draw samples from the fitted KDE.
        """
        if self.kde is None:
            raise RuntimeError("Distribution not fitted")

        return self.kde.sample(n)

    def ll(self, x: np.ndarray):
        """
        Log-likelihood in bits.
        """
        if self.kde is None:
            raise RuntimeError("Distribution not fitted")

        logp = self.kde.score_samples(x.reshape(x.shape[0],1)).flatten()

        # Convert nats -> bits
        return logp / np.log(2)
def mdl_score(gmm, X,precision=1):
    n = X.shape[0]
    # score returns average log likelihood per sample in natural log
    log_likelihood_nat = gmm.score(X) * n  
    log_likelihood_bits = log_likelihood_nat / np.log(2)  + n*np.log2(precision) # convert ln to log2
    k = gmm.n_components
    
    score = -log_likelihood_bits
    # Parameters count for 1D GMM
    weights, means, variances= extract_gmm_params(gmm)

    
    score += universalIntegerLength(k)

    for i,(w,mu,sigma) in enumerate(zip( weights, means, variances)):
        if i > 0:
            score += arbitrary_threshold_cost(w,-math.log10(precision))
        score += arbitrary_threshold_cost(mu,-math.log10(precision))
        score += arbitrary_threshold_cost(sigma,-math.log10(precision))
      

    
    return score  # MDL in bits

def extract_gmm_params(gmm):
    # Extract weights, means, variances explicitly
    weights = gmm.weights_           # shape (k,)
    means = gmm.means_.flatten()     # shape (k,)
    # For 'full' covariance type in 1D, covariances_ shape (k,1,1)
    variances = gmm.covariances_.flatten()  # shape (k,)
    return weights, means, variances

def fit_gmm_mdl(X, max_components=10):
    X = X.reshape(-1, 1)
    best_mdl = np.inf
    best_gmm = None
    best_params = None

    for k in range(1, max_components + 1):
        gmm = GaussianMixture(n_components=k, covariance_type='full', random_state=0)
        gmm.fit(X)

        current_mdl = mdl_score(gmm, X)

        if current_mdl < best_mdl:
            best_mdl = current_mdl
            best_gmm = gmm
            best_params = extract_gmm_params(gmm)
        else:
            break

    return best_mdl, best_gmm, best_params 

