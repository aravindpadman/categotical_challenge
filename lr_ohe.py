import subprocess
subprocess.run("cp /kaggle/input/rapids/rapids.0.15.0 /opt/conda/envs/rapids.tar.gz".)

import os, sys
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from config import Config


np.random.seed(Config.RANDOM_SEED)



class LROHEmodel:

    def __init__(self, train_df, model=None, scoring='roc_auc'):
        if model:
            self.model = model
        else:
            print("Using SKlearn LogisticRegression model")
            self.model = LogisticRegression(n_jobs=-1)
        self.train_df = train_df
        self.X = train_df.drop("target", axis=1)
        self.y = train_df['target']
        self.scoring = scoring
        self.cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=Config.RANDOM_STATE)
        self.pipe = None
        self.best_estimator_ = None
        self.best_score_ = -1.
        self.best_params_ = None

    def create_pipeline(self, X, y=None):
        """create data pipeline which should for training and inference"""
        preprocess_1 = ColumnTransformer(
            [
            ("ohe", OneHotEncoder(categories = 'auto', sparse = True, dtype = 'uint8', handle_unknown="ignore"), 
            [col for col in self.X.columns if col not in ['target', 'kfold']])
        ])
        estimator = [("preprocess_1", preprocess_1)]
        pipe = Pipeline(steps=estimator)
        return pipe
    
    def RandomizedSearchCV(self, **kwargs):
        """perform RandomisedCV and modify the model and paramns if the new results has better score"""
        if self.pipe:
            X = self.pipe.transform(self.X)
        else:
            self.pipe = self.create_pipeline(self.X)
            X = self.pipe.fit_transform(self.X)
        
        print("starting RSCV") 
        rscv = RandomizedSearchCV(self.model, **kwargs, scoring=self.scoring)
        score = rscv.fit(X, self.y)

        if self.best_score_ < score.best_score_:
            print(f"new cv score ={round(score.best_score_, 4)} if better than the previous score = {round(self.best_score_, 4)}")
            self.best_score_ = score.best_score_
            self.best_estimator_= score.best_estimator_
            self.best_params_ = score.best_params_
            print("Updated the instance with new best model.")
            print(f"new model params={self.best_params_}")
        else:
            print(f"The new CV={round(score.best_score_, 4)} score is lesser than the previous model score={round(self.best_score_,4)}")
            print("Instance not updated with new CV results")
        print(pd.DataFrame(score.cv_results_))

if __name__ == "__main__":
    train_df = pd.read_csv("/kaggle/input/cat-in-the-dat/train.csv")
    lr = LROHEmodel(train_df)
    kwargs = {
        "param_distributions": {
            "C": np.random.uniform(0,.2, 100),
            "max_iter": [1000],
            "class_weight": [None],
        },
        "n_iter": 3,
        "n_jobs": 4, 
        "verbose": 10, 
    }
    lr.RandomizedSearchCV(**kwargs)


    
