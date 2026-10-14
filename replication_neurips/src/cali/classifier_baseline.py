import numpy as np

import numpy as np
from statsmodels.nonparametric.kde import KDEUnivariate
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import pandas as pd 

class MedianDistDetector:
    def fit(self, data):
        data = np.asarray(data)
        self.med = np.median(data)
        self.train_dist = np.abs(data - self.med)
        self.n = len(data)

    def predict(self, X):
        X = np.asarray(X)
        X_dist = np.abs(X - self.med)

        # Broadcast comparison: (n_train, n_test)
        frac = (self.train_dist[:, None] > X_dist).mean(axis=0)

        # Avoid log(0)
        frac = np.clip(frac, 1e-12, None)

        return frac




class KDEDetector:
    def __init__(self, bw="silverman"):
        self.bw = bw

    def fit(self, data):
        data = np.asarray(data)
        self.data = data

        self.kde = KDEUnivariate(data)
        self.kde.fit(bw=self.bw, fft=True)

        # Density at training points
        self.train_density = self.kde.evaluate(data)

    def predict(self, X):
        X = np.asarray(X)
        test_density = self.kde.evaluate(X)

        # Empirical p-values
        frac = (self.train_density[:, None] <= test_density).mean(axis=0)
        return np.clip(frac, 1e-12, 1.0)


def benjamini_hochberg(pvals, alpha):
    pvals = np.asarray(pvals)
    m = len(pvals)

    if m == 0:
        return np.zeros(0, dtype=bool)

    order = np.argsort(pvals)
    ranked = pvals[order]
    thresholds = alpha * (np.arange(1, m + 1) / m)

    passed = ranked <= thresholds
    if not np.any(passed):
        return np.zeros(m, dtype=bool)

    k = np.max(np.where(passed))
    rejected = np.zeros(m, dtype=bool)
    rejected[order[:k + 1]] = True
    return rejected



import networkx as nx

def classify(data,anomaly_mask=None, root_causes=None, dag=None, alpha=0.05,detector="kde",correction=True):

    if isinstance(data, pd.DataFrame):
        data = data.to_numpy()
        

    train_data = data[~anomaly_mask] 
    test_data = data
    n_features = train_data.shape[1]
    classes_no_correction = np.zeros_like(data)
    classes_correction = classes_no_correction.copy()
    # Fit detectors
    detector_list = [KDEDetector() if detector =="kde" else MedianDistDetector() for _ in range(n_features)]

    p_vals = np.zeros_like(test_data)
    print(n_features)
    print(detector_list)
    for i in range(n_features):
        detector_list[i].fit(train_data[:, i])
        p_vals[:,i] = detector_list[i].predict(test_data[:,i])
        

    # Marginal empirical p-values for test point
    # pvals = np.array([
    #     detector_list[i].predict(test_data[i])
    #     for i in [r for r,_ in root_causes]
    # ])
    # print(root_causes)
    for  (r,c) in root_causes:
        
        successors = list(dag.successors(c))
       
        # BH correction among descendants
        desc_pvals = p_vals[r,successors]
        significant_desc_no_cor = desc_pvals < (alpha)
        significant_desc_cor = desc_pvals < (alpha / len(successors))

        # Any significant descendant → class 0
        if not np.any(significant_desc_no_cor):
            # classes_no_correction[r,] = 0

            classes_no_correction[r,c] = 1

        if not np.any(significant_desc_cor):
            classes_correction[r,c] = 1
       

        # Root is marginal outlier and descendants are clean → class 1
  
    if correction:
        return classes_correction
    else:
        return classes_no_correction

    






    return classes 


import random

import numpy as np
import random
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor

def classify_lasosa(training_data: np.array, test_data: np.array, alpha=0.05, group_count=100, group_size=3):
    """
    i.i.d. tabular version of the LaRosa et al. method.
    
    Parameters
    ----------
    training_data : np.array, shape (n_samples, n_sensors)
        Historical clean data for training predictors.
    test_data : np.array, shape (m_samples, n_sensors)
        Data to classify anomalies on.
    alpha : float
        Quantile threshold for defining residual anomalies.
    group_count : int
        Number of random sensor groups to generate.
    group_size : int
        Number of sensors per group.
        
    Returns
    -------
    meta_labels : np.array, shape (n_sensors,)
        0 = normal, 1 = sensor anomaly, 2 = process anomaly
    """
    n, d = training_data.shape
    scaler = StandardScaler()
    training_data = scaler.fit_transform(training_data)
    test_data = scaler.transform(test_data)
    
    # 1️⃣ Create random groups
    sensor_groups = set()
    while len(sensor_groups) < group_count:
        group = tuple(sorted(random.sample(range(d), k=group_size)))
        sensor_groups.add(group)
    sensor_groups = list(sensor_groups)
    
    # 2️⃣ Train predictors and compute residuals
    # Store binary anomalies per (group, sensor)
    anomaly_flags = dict()  # key: (group, sensor), value: binary array for test data
    print("Fitting models")
    counter = 0
    for group in sensor_groups:
        for sensor in range(d):
            # Inputs = group minus sensor if sensor in group
            input_cols = [j for j in group if j != sensor]
            X_train = training_data[:, input_cols]
            y_train = training_data[:, sensor]
            
            # Train regressor
            model = GradientBoostingRegressor()
            model.fit(X_train, y_train)
            print(F"Fitted model {counter+1}/{group_count*d}")
            counter += 1
            # Compute training residuals to set threshold
            train_pred = model.predict(X_train)
            train_residuals = np.abs(y_train - train_pred)
            threshold = np.quantile(train_residuals, 1 - alpha)
            
            # Compute test residuals and anomalies
            X_test = test_data[:, input_cols]
            y_test = test_data[:, sensor]
            test_pred = model.predict(X_test)
            test_residuals = np.abs(y_test - test_pred)
            test_anomalies = (test_residuals >= threshold).astype(int)
            
            anomaly_flags[(group, sensor)] = test_anomalies
    
    # 3️⃣ Compute meta-predictor: r_i and c_i
    r = np.zeros_like(test_data)
    c = np.zeros_like(test_data)
    classes = np.zeros_like(test_data)

    for row_id in range(test_data.shape[0]):
        for sensor in range(d):
            # r_i = number of groups that flag this sensor
            r[row_id,sensor] = sum(anomaly_flags[(g, sensor)][row_id] > 0 for g in sensor_groups)
            
            # c_i = number of anomaly flags for other sensors in groups that exclude this sensor
            c_sum = 0
            for other_sensor in range(d):
                if other_sensor == sensor:
                    continue
                for g in sensor_groups:
                    if sensor not in g:
                        c_sum += int(anomaly_flags[(g, other_sensor)][row_id] > 0)
            c[row_id,sensor] = c_sum
    
    # 4️⃣ Classify sensors
    #  1 = sensor anomaly, else normal or process″
    classes = np.zeros(test_data.shape[0])
    for row_id in range(test_data.shape[0]):
        
        for sensor in range(d):
            if r[row_id,sensor] == 0:
                continue
            elif r[row_id,sensor] == len(sensor_groups) and c[row_id,sensor] == 0:
                classes[row_id] = 1  # sensor anomaly
            else:
                if classes[row_id] != 1:
                    classes[row_id] = 2


              
    
#     classes = np.where(classes.sum(axis=1, keepdims=True) > 0, 1, 0)

#     # Collapse per-sensor classes into per-sample indicator
# # 1 = at least one sensor anomaly in the sample, 0 = otherwise
#     sample_sensor_anomaly = (classes == 1).any(axis=1).astype(int)


    return classes

    
    
    # No classifiy according to rool
    



        








     
