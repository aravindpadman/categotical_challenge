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
    def fit(self, X, y=None):
        pass



if __name__ == "__main__":
    train = cudf.read_csv("/kaggle/input/cat-in-the-dat/train.csv", index_col='id')
    test =  cudf.read_csv("/kaggle/input/cat-in-the-dat/test.csv", index_col='id')
    submission = cudf.read_csv("/kaggle/input/cat-in-the-dat/sample_submission.csv", index_col='id')
