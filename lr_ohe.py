import os, sys
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedKFold

from config import Config



class LROHEmodel:
    def __init__(self, train_df):
        self.train_df = train_df
        self.X = train_df.drop("target", axis=1)
        self.y = train_df['target']
        self.cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=Config.RANDOM_STATE)
