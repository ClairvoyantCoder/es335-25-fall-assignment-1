import pandas as pd
import numpy as np
from typing import Union

def accuracy(y_hat: pd.Series, y: pd.Series) -> float:
    assert y_hat.size == y.size
    return (y_hat == y).sum() / len(y)

def precision(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    tp = ((y_hat == cls) & (y == cls)).sum()
    fp = ((y_hat == cls) & (y != cls)).sum()
    return tp / (tp + fp + 1e-9)

def recall(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    tp = ((y_hat == cls) & (y == cls)).sum()
    fn = ((y_hat != cls) & (y == cls)).sum()
    return tp / (tp + fn + 1e-9)

def rmse(y_hat: pd.Series, y: pd.Series) -> float:
    assert y_hat.size == y.size
    return np.sqrt(((y_hat - y) ** 2).mean())

def mae(y_hat: pd.Series, y: pd.Series) -> float:
    assert y_hat.size == y.size
    return np.abs(y_hat - y).mean()
