import os
import pandas as pd
import cupy as cp
from pathlib import Path

class HeartDisease:
    def __init__(self):
        self.path = lambda : Path(f"dataset/HeartDisease/HeartDisease.csv")

    def load_data(self):
        return pd.read_csv(self.path())

class AirBnb:
    def __init__(self):
        self.test_cat_path = lambda _id: Path(f"dataset/AirBnb/inputs_categorical_test_{_id}.csv")
        self.test_num_path = lambda : Path(f"dataset/AirBnb/inputs_numerical_test.csv")
        self.train_cat_path = lambda _id: Path(f"dataset/AirBnb/inputs_categorical_train_{_id}.csv")
        self.train_num_path = lambda : Path(f"dataset/AirBnb/inputs_numerical_train.csv")
        self.train_target_path = lambda : Path(f"dataset/AirBnb/targets_train.csv")
        self.test_target_path = lambda : Path(f"dataset/AirBnb/targets_test.csv")

    def load_train_data(self):
        c1 = pd.read_csv(self.train_cat_path(_id=1))
        c2 = pd.read_csv(self.train_cat_path(_id=2))
        c3 = pd.read_csv(self.train_cat_path(_id=3))
        n1 = pd.read_csv(self.train_num_path())
        n1 = n1.merge(c1, on='id')
        n1 = n1.merge(c2, on='id')
        n1 = n1.merge(c3, on='id')
        X, y = n1, pd.read_csv(self.train_target_path())
        idx = cp.argsort(X['id'])
        return X.iloc[idx], y.iloc[idx]

    def load_test_data(self):
        c1 = pd.read_csv(self.test_cat_path(_id=1))
        c2 = pd.read_csv(self.test_cat_path(_id=2))
        c3 = pd.read_csv(self.test_cat_path(_id=3))
        n1 = pd.read_csv(self.test_num_path())
        n1 = n1.merge(c1, on='id')
        n1 = n1.merge(c2, on='id')
        n1 = n1.merge(c3, on='id')
        X, y = n1, pd.read_csv(self.test_target_path())
        idx = cp.argsort(X['id'])
        return X.iloc[idx], y.iloc[idx]