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
        if self.ohe:
            X_transformed = X.copy()
            cp_ohe = self.ohe.transform(X_transformed[self.columns])
            temp = cudf.DataFrame(cp_ohe, index= X_transformed.index, columns = ['ohe_'+str(i) for i in range(cp_ohe.shape[1])])
            X_transformed = X_transformed.drop(self.columns, axis=1)
            X_transformed = X_transformed.join(temp)
            return X_transformed
        else:
            raise("onehot encoding must fit before")


class TargetEncodeTransform(BaseEstimator, TransformerMixin):
    def __init__(self, columns, n_folds=4, smooth=0, seed=42, split_method='interleaved', output_type='auto'):
        self.columns= columns
        self.kwargs = kwargs
        self.te = None
        self.n_folds = n_folds
        self.smooth = smooth
        self.seed = seed
        self.split_method = split_method
        self.output_type = output_type

    def fit(self, X, y):
        self.ohe = TargetEncoder(self.n_folds, self.sm)
        self.ohe.fit(X[self.columns])
        return self
    
    def transform(self, X, y=None):
        if self.ohe:
            X_transformed = X.copy()
            cp_ohe = self.ohe.transform(X_transformed[self.columns])
            temp = cudf.DataFrame(cp_ohe, index= X_transformed.index, columns = ['ohe_'+str(i) for i in range(cp_ohe.shape[1])])
            X_transformed = X_transformed.drop(self.columns, axis=1)
            X_transformed = X_transformed.join(temp)
            return X_transformed
        else:
            raise("onehot encoding must fit before")

if __name__ == "__main__":
    train = cudf.read_csv("/kaggle/input/cat-in-the-dat/train.csv", index_col='id')
    test =  cudf.read_csv("/kaggle/input/cat-in-the-dat/test.csv", index_col='id')
    submission = cudf.read_csv("/kaggle/input/cat-in-the-dat/sample_submission.csv", index_col='id')
    
    columns = ['bin_0', 'nom_4']

    kwargs={'handle_unknown': 'ignore', 'sparse': False}
    ohe = OHEColumnTransform(columns, handle_unknown='ignore', sparse=False)
    ohe.fit(train)
    temp = ohe.transform(train)
    print("hello")