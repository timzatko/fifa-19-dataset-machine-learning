import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin


def transform_wage(val):
    if val is np.nan:
        return val

    val = val.replace('€', '')
    if 'M' in val:
        val = val.replace('M', '')
        return float(val) * 1000000
    if 'K' in val:
        val = val.replace('K', '')
        return float(val) * 1000


class MoneyTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, col):
        self.col = col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X[self.col] = X[self.col].astype('str').apply(transform_wage).astype('float64')
        return X


def transform_height(value):
    if value is np.nan:
        return value

    arr = value.split("'")
    return int(arr[0]) * 30.48 + int(arr[1]) * 3.48


class LengthTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, col):
        self.col = col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X[self.col] = X[self.col].apply(transform_height).astype('float64')
        return X


def transform_weight(value):
    if value is np.nan:
        return value

    return value.replace('lbs', '')


class WeightTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, col):
        self.col = col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X[self.col] = X[self.col].apply(transform_weight).astype('float64')
        return X


class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        assert isinstance(x, pd.DataFrame)

        return x[self.columns]