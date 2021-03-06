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
from sklearn.metrics import make_scorer
from custom_scoring_module import roc_auc_gpu

np.random.seed(Config.RANDOM_SEED)

kf = StratifiedKFold(n_splits=5, shuffle=False, random_state=Config.RANDOM_STATE)

class ToCudfTransform(BaseEstimator, TransformerMixin):
    def fit(self, X, y):
        return self
    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            X = cudf.from_pandas(X)
        return X

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
        if isinstance(y, pd.DataFrame) or isinstance(y, pd.Series):
            y = cudf.from_pandas(y)
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
    train = pd.read_csv("/kaggle/input/cat-in-the-dat/train.csv", index_col=None)
    test =  cudf.read_csv("/kaggle/input/cat-in-the-dat/test.csv", index_col=None)
    submission = cudf.read_csv("/kaggle/input/cat-in-the-dat/sample_submission.csv", index_col=None)

    X = train.drop('target', axis=1)
    y = train['target']
    
    one_hot_columns = ['bin_0', 'bin_1', 'bin_2', 'bin_3',
        'bin_4', 'nom_0', 'nom_1', 'nom_2', 'nom_3', 'nom_4', 'day', 'month',
        'ord_0', 'ord_1', 'ord_2', 'ord_3', 'ord_4', 'ord_5',
     ]
    
    pipe = make_pipeline(
        ToCudfTransform(),
        OHEColumnTransform(one_hot_columns, handle_unknown='ignore', sparse=False),
        TargetEncodeTransform(columns=['nom_5',]),
        TargetEncodeTransform(columns=['nom_6',]),
        TargetEncodeTransform(columns=['nom_7',]),
        TargetEncodeTransform(columns=['nom_8',]),
        TargetEncodeTransform(columns=['nom_9',]),
        # TargetEncodeTransform(columns=['ord_0',]),
        # TargetEncodeTransform(columns=['ord_1',]),
        # TargetEncodeTransform(columns=['ord_2',]),
        # TargetEncodeTransform(columns=['ord_3',]),
        # TargetEncodeTransform(columns=['ord_4',]),
        # TargetEncodeTransform(columns=['ord_5',]),
        LogisticRegression(),
    )
    pipe.fit(X, y)
    y_pred = pipe.predict(X)
    param_grid = {"logisticregression__C": [1,2,3,4,5,6]}
    rscv = RandomizedSearchCV(pipe, param_distributions=param_grid,
                                 scoring=make_scorer(roc_auc_gpu,  greater_is_better=True),
                                  cv=kf, verbose=6,  random_state=Config.RANDOM_STATE, n_iter=1)
    model = rscv.fit(X, y)













    print(pipe)
    # X_transformed = pipe.transform(train)
    # print(X_transformed.head())
    # print("hello")
    # if not os.path.isdir("tmp"):
    #     os.mkdir("tmp")

    # with open('./tmp/train_transformed.csv', 'w') as file:
    #     X_transformed.to_csv(file, index=False, chunksize=1000)