import os
import pandas as pd
from sklearn.model_selection import StratifiedKFold

def create_folds(df, kf):
    """create crossvalidation folds"""
    df = df.sample(frac=1, random_state=0).reset_index(drop=True)
    X = df.drop("target", axis=1)
    y = df['target']
    df['kfold'] = -1
    for fold_, (_, val_idx) in enumerate(kf.split(X, y)):
        df.loc[val_idx, 'kfold'] = fold_
    return df


if __name__ == "__main__":
    train_df = pd.read_csv("/kaggle/input/cat-in-the-dat/train.csv", index_col='id')
    kf = StratifiedKFold(n_splits=5)
    train_folds = create_folds(train_df, kf)
    if not os.path.isdir("./data"):
        os.mkdir("./data")
    train_folds.to_csv('./data/train_folds.csv', index=False)