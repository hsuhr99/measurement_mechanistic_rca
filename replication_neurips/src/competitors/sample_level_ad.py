import numpy as np
import torch

from pyod.models.auto_encoder import AutoEncoder
from pyod.models.deep_svdd import DeepSVDD
from pyod.models.copod import COPOD
from pyod.models.iforest import IForest
from pyod.models.lof import LOF
import pandas as pd
import pandas as pd
from pzflow import Flow
import multiprocessing as mp
from sklearn.preprocessing import RobustScaler

def ae_wrapper(train_data: np.ndarray,epochs=10,lr=1e-3, true_labels=None,batch_size=32):  
    train_data = RobustScaler().fit_transform(train_data)
    clf = AutoEncoder(epoch_num=epochs,lr=lr,batch_size=batch_size)
    train_data  = train_data.astype(np.float32)
    print(train_data.shape)
    print(train_data.dtype)
    print("Training")
    clf.fit(train_data)
    print("Fitted")
    return clf.decision_function(train_data)


def deepsvdd_wrapper(train_data: np.ndarray,epochs=100,batch_size=128):
    train_data = RobustScaler().fit_transform(train_data)
    train_data  = train_data.astype(np.float32)
    clf = DeepSVDD(train_data.shape[1],epochs=epochs,batch_size=batch_size)
    clf.fit(train_data)
    labels = set(np.where(clf.predict(train_data) == 1)[0])  # 1: anomaly, 0: normal
    score = clf.decision_function(train_data)  # higher means more anomalous
    return score


def copod_wrapper(train_data: np.ndarray):
    train_data = RobustScaler().fit_transform(train_data)
    clf = COPOD()
    clf.fit(train_data)
    # labels = set(np.where(clf.predict(train_data) == 1)[0])  # 1: anomaly, 0: normal
    score = clf.decision_function(train_data)  # higher means more anomalous
    return score


def iforest_wrapper(train_data: np.ndarray,n_estimators = 300,max_features=1):
    train_data = RobustScaler().fit_transform(train_data)
    clf = IForest(n_estimators=n_estimators,max_features=max_features)
    clf.fit(train_data)
    labels = set(np.where(clf.predict(train_data) == 1)[0])  # 1: anomaly, 0: normal
    score = clf.decision_function(train_data)  # higher means more anomalous
    return score


def lof_wrapper(train_data: np.ndarray,n_neighbors=10,metric="minkowski"):
    train_data = RobustScaler().fit_transform(train_data)
    clf = LOF(n_neighbors=n_neighbors,metric=metric)
    clf.fit(train_data)
    labels = set(np.where(clf.predict(train_data) == 1)[0])  # 1: anomaly, 0: normal
    score = clf.decision_function(train_data)  # higher means more anomalous
    return score


class NF_WRAPPER:
    def __init__(self,epochs=10,lr=1e-3,batch_size=32):

        self.epochs = epochs 
        self.lr = lr 
        self.batch_size = batch_size 
    
    def fit(self,data):
       
        X = data.astype(np.float32)
       
        columns = list(range(X.shape[1]))
        data = pd.DataFrame(X)

        # Create flow
        self.flow = Flow(columns)

        # Train the flow
        self.flow.train(data, epochs=self.epochs, batch_size=self.batch_size, verbose=False)

    def decision_function(self,data):
        
        X = data.astype(np.float32)
       
        columns = list(range(X.shape[1]))
        data = pd.DataFrame(X)
        log_probs = self.flow.log_prob(data)
        sample_scores = -log_probs

        return sample_scores 



def nf_wrapper(X: np.ndarray, epochs=10, lr=1e-3, batch_size=32):
    X = RobustScaler().fit_transform(X)
    X = X.astype(np.float32)
    columns = list(range(X.shape[1]))
    data = pd.DataFrame(X)

    # Create flow
    flow = Flow(columns)

    # Train the flow
    flow.train(data, epochs=epochs, batch_size=batch_size, verbose=False)

    # Sample-level anomaly
    log_probs = flow.log_prob(data)
    sample_scores = -log_probs

    # # Feature-level anomaly: marginal approximation
    # feature_scores = []
    # for col in columns:
    #     # Drop this column to approximate its contribution
    #     X_marg = data.drop(columns=[col])
    #     logp_marg = flow.log_prob(X_marg)
    #     # Contribution = difference between full and marginal NLL
    #     contribution = sample_scores - (-logp_marg)
    #     feature_scores.append(contribution)

    # feature_scores = np.stack(feature_scores, axis=1)
    return sample_scores


