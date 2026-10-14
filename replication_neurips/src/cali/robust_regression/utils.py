import numpy as np
from scipy.stats import norm
import math
from sklearn.neighbors import LocalOutlierFactor



class ModelWrapper:
    
    def __init__(self, bayesian_regression_model, basis_transform, x_scaler, y_scaler, basis_size, outlier_mask):
        self.model = bayesian_regression_model
        self.basis_transform = basis_transform if basis_transform else []
        self.x_scaler = x_scaler
        self.y_scaler = y_scaler
        self.basis_size = basis_size
        self.outlier_mask = outlier_mask
        self.transformation_order = []
        self.feature_map = self._build_feature_map()

    def _build_feature_map(self):
        feature_map = []
        n_features = len(self.basis_transform)
        for i in range(n_features):
            feature_map.append((i, None))  # identity
            for bf in self.basis_transform[i]:
                feature_map.append((i, bf))
        return feature_map

    def transform(self, X):
        transformed = X.copy()
        for i, j in self.transformation_order:
          
            transformed = np.hstack([transformed, self.basis_transform[j](X[:, i].reshape(-1, 1))])
        return self.x_scaler.transform(transformed)

    def inverse_transform_y(self, y):
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        return self.y_scaler.inverse_transform(y)

    def predict(self, X):
        X_trans = self.transform(X)
        preds, post_std = self.model.predict(X_trans,return_std=True)
        return self.inverse_transform_y(preds), post_std

    def get_residuals(self, X, y):
        return y.flatten() - self.predict(X)[0].flatten()

    def get_scaled_residuals(self, X, y):
        """
        Compute scaled residuals r_scaled = residual / posterior std dev at each x.
        """
      
        preds, post_std = self.predict(X)
        residuals = y.flatten() - preds.flatten() 

        if np.isnan(post_std).any() or np.isinf(post_std).any():
            raise ValueError("Post std contains NaNs or infs.")

        
        if hasattr(self.y_scaler, "scale_"):
            post_std *= self.y_scaler.scale_[0]
    
        r_scaled = residuals / post_std

        return r_scaled
    
    def evaluate(self, x):
        return self.predict(x.reshape(1, -1))[0]

    def gradient(self, x):
        x = x.reshape(1, -1)
        n_inputs = x.shape[1]

        # Multivariate output is not supported in gradient method yet
        if self.model.coef_.ndim > 1:
            raise NotImplementedError("Gradient computation not supported for multivariate outputs.")

        grads = np.zeros(n_inputs)

        # Apply basis transformations to a single input sample
        transformed = []
        for i in range(n_inputs):
            transformed.append(x[:, i].reshape(-1, 1))
            for bf in self.basis_transform[i]:
                transformed.append(bf(x[:, i].reshape(-1, 1)))
        X_aug = np.hstack(transformed)

        # Scale the transformed input
        X_scaled = self.x_scaler.transform(X_aug)

        # Model weights
        weights = self.model.coef_.flatten()

        # Compute gradient via chain rule
        for j, (input_idx, bf) in enumerate(self.feature_map):
            if bf is None:
                df_dx = 1.0
            else:
                df_dx = bf.grad(x[:, input_idx].reshape(-1, 1)).flatten()
            grads[input_idx] += weights[j] * df_dx[0]

        # Adjust for y scaling
        if hasattr(self.y_scaler, "scale_"):
            grads *= self.y_scaler.scale_[0]

        return grads

    def is_outlier(self):
        return self.outlier_mask

    def model_complexity(self):
        cost = 0
        for i, bf in enumerate(self.basis_transform):
            cost += math.log2(self.basis_size + 1) + log_binom(self.basis_size, len(bf))

        if hasattr(self.model, 'coef_'):
            weights = self.model.coef_
            if weights.ndim == 1:
                weights = weights.reshape(1, -1)
            intercept = (
                self.model.intercept_.flatten()
                if hasattr(self.model.intercept_, 'flatten')
                else np.array([self.model.intercept_])
            )

            for row in weights:
                for w in row:
                    cost += arbitrary_threshold_cost(w, 10)
            for b in intercept:
                cost += arbitrary_threshold_cost(b, 10)
        return cost

class BasisFunction:
    def __init__(self, func, derivative):
        self.func = func
        self.derivative = derivative

    def __call__(self, x):
        return self.func(x)

    def grad(self, x):
        return self.derivative(x)


BASIS_FUNCTIONS = []

for k in range(2, 6):
    BASIS_FUNCTIONS.append(
        BasisFunction(
            lambda X, k=k: X**k,
            lambda X, k=k: k * X**(k - 1)
        )
    )





BASIS_FUNCTIONS += [
    # BasisFunction(np.cos, lambda X: -np.sin(X)),
    BasisFunction(np.sin, lambda X: np.cos(X)),
]


def robust_z_scores(x,med=None,sigma = None):
    """Compute robust z-scores using median and MAD"""
    med = np.median(x)
    if med is None:
        med = np.median(x)
    if sigma is None:
        mad = np.median(np.abs(x - med))
        sigma = 1.4826 * mad
        sigma = max(sigma, 1e-6)  # prevent division by zero
    return np.abs(x - med) / (sigma)







def logN(z):
    z = np.ceil(z);
    
    if z < 1 :
        return 0;
    else :
        log_star = logg(z);
        sum = log_star;
        
        while log_star > 0:
            log_star = logg(log_star);
            sum = sum+log_star;
        
        return sum + logg(2.865064)
    
def gaussian_score_emp_sse(sse, n,min_diff,a=None,b=None,k=0):
    var = sse / n
    sigma = np.sqrt(var)

    # print(f"Sigma: {sigma}, sse: {sse}, n: {n}, min_diff: {min_diff}")
    return gaussian_score_sse(sigma, sse, n,min_diff,a,b,k), sigma

def gaussian_score_sse (sigma, sse, n,resolution=1e-10):
    sigmasq = sigma**2;
    if sse == 0.0 or sigmasq == 0.0:
        return np.array([0.0]);
    else:
        err = (sse / (2 * sigmasq * np.log(2))) + ((n/2) * logg(2 * np.pi * sigmasq)) - n * logg(resolution) 
      

        # print(f"{(sse / (2 * sigmasq * np.log(2)))}, {((n/2) * logg(2 * np.pi * sigmasq))}, {n * logg(resolution)}")
        return max(err,0);



# Helper function for binary logarithm



def logg(x):
    return np.where(x<=0,0,np.log2(x))
  

def universalIntegerLength(n):
    return logN(n)

def thresholdNCost( n):
   
    return math.log2(n+1)

def residualCost(r,n,epsilon):
    return gaussian_score_emp_sse(r,n,epsilon)

def arbitrary_threshold_cost(x,decimal_places):
    
    return 1 + universalIntegerLength(np.floor(np.abs(x)) + 1) + np.log2(10**decimal_places+1)

import numpy as np

def log_fac(arr):
    arr = np.asarray(arr)
    max_val = np.max(arr)
    
    # Precompute log2 of integers from 1 to max_val
    log2_values = np.log2(np.arange(1, max_val + 1))
    cumsum_log2 = np.cumsum(log2_values)

    # Pad with a zero at the beginning so that log2_factorial[0] = 0
    cumsum_log2 = np.concatenate(([0.0], cumsum_log2))

    # Use advanced indexing to get log2(n!) for each element
    return cumsum_log2[arr]



    
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


def log_binom(n,k):
    return log_fac(n) - log_fac(k) - log_fac(n-k)

def multivariate_lof_outlier_mask(X, contamination=0.05):
    lof = LocalOutlierFactor(n_neighbors=20, contamination=contamination)
    y_pred = lof.fit_predict(X)
    # y_pred = 1 for inliers, -1 for outliers
    return y_pred == 1



def robust_outlier_mask_bonferroni(x, alpha=0.05):
    """
    Return a boolean mask where True indicates an inlier,
    using robust z-scores with Bonferroni correction for multiple comparisons.

    Parameters:
        x (array-like): 1D array of observations
        alpha (float): Overall significance level (e.g., 0.05 for 95% family-wise confidence)

    Returns:
        mask (ndarray): Boolean array, True for inliers
    """
    x = np.asarray(x)
    n = len(x)
    
    # Bonferroni correction: divide alpha by the number of tests
    corrected_alpha = 1 - (1 - alpha)**(1/n)

    
    # Convert corrected alpha to z-score threshold (two-tailed)
    z_thresh = norm.ppf(1 - corrected_alpha)
    
    # Compute robust z-scores
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    mad = max(mad, 1e-6)
    z = np.abs(x - med) / (1.4826 * mad)
    
    return z < z_thresh

    
def gaussian_score_sse (sigma, sse, n,resolution=1e-10,a=None,b=None,k=0):
    sigmasq = sigma**2;
    if sse == 0.0 or sigmasq == 0.0:
        return np.array([0.0]);
    else:
        err = (sse / (2 * sigmasq * np.log(2))) + ((n/2) * logg(2 * np.pi * sigmasq)) - n * logg(resolution) 
       

        return max(err,0);

def gaussian_mdl(X,mu=None,alpha=0.05,precision=1e-10):

 
    n = X.shape[0]
    corrected_alpha = 1 - (1 - alpha)**(1/(2*n))
    threshold = norm.ppf(1 - corrected_alpha)  # Bonferroni corrected threshold
    
    inlier_mask = robust_z_scores(X,med=0) < threshold
    X_clean = X[inlier_mask]
    mad = np.median(np.abs(X))
    sigma = mad * 1.4826
    sigma = max(sigma,1e-3)
    sigma = min(sigma,1e3)
    length = gaussian_score_sse(sigma,np.sum((X_clean)**2),X_clean.shape[0],precision)
    length += log_binom(n,X_clean.shape[0]) + logg(n+1)

    for x in X[~inlier_mask]:
        length += arbitrary_threshold_cost(x,10)
    return length