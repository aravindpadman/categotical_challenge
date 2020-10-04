"""custom scoring function compatible with rapids"""

import os, sys

sys.path = ["/opt/conda/envs/rapids/lib/python3.7/site-packages"] + sys.path
sys.path = ["/opt/conda/envs/rapids/lib/python3.7"] + sys.path
sys.path = ["/opt/conda/envs/rapids/lib"] + sys.path 

from cuml.metrics import roc_auc_score

def roc_auc_gpu(estimator, X, y):
    y_proba = estimator.predict_proba(x)[:,1]
    roc_auc = roc_auc_score(y, y_proba)
    return roc_auc