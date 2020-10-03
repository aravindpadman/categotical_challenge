"""cuml logistic regresssion model with custom transformers for onehot and target encoding"""

import os, sys

sys.path = ["/opt/conda/envs/rapids/lib/python3.7/site-packages"] + sys.path
sys.path = ["/opt/conda/envs/rapids/lib/python3.7"] + sys.path
sys.path = ["/opt/conda/envs/rapids/lib"] + sys.path 

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin

from config import Config

import cudf
import cuml
from cuml.preprocessing import OneHotEncoder, TargetEncoder
from cuml.linear_model import LogisticRegression


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
            return X_transformed.reset_index(drop=True)
        else:
            raise("onehot encoding must fit before transform")


class TargetEncodeTransform(BaseEstimator, TransformerMixin):
    def __init__(self, columns, n_folds=4, smooth=0, seed=42, split_method='interleaved', output_type='auto'):
        self.columns= columns
        self.te = None
        self.n_folds = n_folds
        self.smooth = smooth
        self.seed = seed
        self.split_method = split_method
        self.output_type = output_type

    def fit(self, X, y):
        self.te = TargetEncoder(self.n_folds, self.smooth, self.seed, self.split_method, self.output_type)
        self.te.fit(X[self.columns], y)
        return self
    
    def transform(self, X, y=None):
        if self.te:
            X_transformed = X.copy() # .reset_index(drop=True)
            cp_ohe = self.te.transform(X_transformed[self.columns])
            temp = cudf.DataFrame(cp_ohe, index= X_transformed.index, columns = ['targetencode_'+ "_".join(self.columns)])
            X_transformed = X_transformed.drop(self.columns, axis=1)
            X_transformed = X_transformed.join(temp)
            return X_transformed.reset_index(drop=True)
        else:
            raise("Target encoding must fit before transform")
    # def fit_transform(self, X, y, **fit_params):
    #     return self.fit(X, y).transform(X)

if __name__ == "__main__":
    train = cudf.read_csv("/kaggle/input/cat-in-the-dat/train.csv", index_col=None)
    test =  cudf.read_csv("/kaggle/input/cat-in-the-dat/test.csv", index_col=None)
    submission = cudf.read_csv("/kaggle/input/cat-in-the-dat/sample_submission.csv", index_col=None)

    X = train.drop('target', axis=1)
    y = train['target']
    
    one_hot_columns = ['bin_0', 'bin_1', 'bin_2', 'bin_3', 'bin_4', 'nom_0', 'nom_1', 'nom_2', 'nom_3', 'nom_4', 'day', 'month']
    pipe = make_pipeline(
        OHEColumnTransform(one_hot_columns, handle_unknown='ignore', sparse=False),
        TargetEncodeTransform(columns=['nom_5',]),
        TargetEncodeTransform(columns=['nom_6',]),
        TargetEncodeTransform(columns=['nom_7',]),
        TargetEncodeTransform(columns=['nom_8',]),
        TargetEncodeTransform(columns=['nom_9',]),
        TargetEncodeTransform(columns=['ord_0',]),
        TargetEncodeTransform(columns=['ord_1',]),
        TargetEncodeTransform(columns=['ord_2',]),
        TargetEncodeTransform(columns=['ord_3',]),
        TargetEncodeTransform(columns=['ord_4',]),
        TargetEncodeTransform(columns=['ord_5',]),
        None,
    )
    pipe = pipe.fit(train, y)
    print(pipe)
    X_transformed = pipe.transform(train)
    print(X_transformed.head())
    print("hello")
    if not os.path.isdir("tmp"):
        os.mkdir("tmp")

    with open('./tmp/train_transformed.csv', 'w') as file:
        X_transformed.to_csv(file, index=False, chunksize=1000)