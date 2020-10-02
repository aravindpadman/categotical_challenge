"""cuml logistic regresssion model with custom transformers for onehot and target encoding"""

import os, sys

sys.path = ["/opt/conda/envs/rapids/lib/python3.7/site-packages"] + sys.path
sys.path = ["/opt/conda/envs/rapids/lib/python3.7"] + sys.path
sys.path = ["/opt/conda/envs/rapids/lib"] + sys.path 

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from config import Config

import cudf
import cuml


np.random.seed(Config.RANDOM_SEED)



#  categories='auto', drop=None, sparse=True, dtype='float32', handle_unknown='error'
class OHEColumnTransform(BaseEstimator, TransformerMixin):
    def __init__(self, columns, **kwargs):
        self.columns= columns
        self.kwargs = kwargs
        self.ohe = None
    def fit(self, X, y=None):
        self.ohe = OneHotEncoder(**self.kwargs)
        self.ohe.fit(X[self.columns])
        return self
    
    def transform(self, X, y=None):
        X_transformed = X.copy()
        cp_ohe = self.ohe.transform(X_transformed[self.columns])
        temp = cudf.DataFrame(cp_ohe, index= X_transformed.index, columns = ['ohe_'+str(i) for i in range(cp_ohe.shape[1])])
        X_transformed = X_transformed.drop(self.columns, axis=1)
        X_transformed.join(temp)
        return X_transformed



if __name__ == "__main__":
    train = cudf.read_csv("/kaggle/input/cat-in-the-dat/train.csv", index_col='id')
    test =  cudf.read_csv("/kaggle/input/cat-in-the-dat/test.csv", index_col='id')
    submission = cudf.read_csv("/kaggle/input/cat-in-the-dat/sample_submission.csv", index_col='id')
