import pandas as pd
import numpy as np

def one_hot_encoding(X: pd.DataFrame) -> pd.DataFrame:
    """
    Perform one-hot encoding on categorical columns.
    """
    return pd.get_dummies(X, drop_first=False)

def check_ifreal(y: pd.Series) -> bool:
    """
    Return True if y contains real (continuous) values, else False.
    """
    return pd.api.types.is_numeric_dtype(y) and (y.nunique() > 10)

def entropy(Y: pd.Series) -> float:
    """
    Entropy of a discrete distribution.
    """
    values, counts = np.unique(Y, return_counts=True)
    probs = counts / counts.sum()
    return -np.sum(probs * np.log2(probs + 1e-9))

def gini_index(Y: pd.Series) -> float:
    """
    Gini index for classification.
    """
    values, counts = np.unique(Y, return_counts=True)
    probs = counts / counts.sum()
    return 1 - np.sum(probs ** 2)

def mse(Y: pd.Series) -> float:
    """
    Mean squared error of regression target (variance-like).
    """
    return np.var(Y)

def information_gain(Y: pd.Series, attr: pd.Series, criterion: str) -> float:
    """
    Compute information gain or reduction in impurity.
    """
    if check_ifreal(Y):  # regression
        base = mse(Y)
    else:  # classification
        base = entropy(Y) if criterion == "information_gain" else gini_index(Y)

    values, counts = np.unique(attr, return_counts=True)
    weighted = 0
    for v, c in zip(values, counts):
        subset = Y[attr == v]
        if check_ifreal(Y):
            score = mse(subset)
        else:
            score = entropy(subset) if criterion == "information_gain" else gini_index(subset)
        weighted += (c / len(Y)) * score

    return base - weighted

def opt_split_attribute(X: pd.DataFrame, y: pd.Series, criterion, features: pd.Series):
    """
    Find best attribute (and threshold if continuous).
    """
    best_attr, best_val, best_score = None, None, -1e9

    for feature in features:
        if pd.api.types.is_numeric_dtype(X[feature]):
            # Try splitting at median
            threshold = X[feature].median()
            left = y[X[feature] <= threshold]
            right = y[X[feature] > threshold]
            if len(left) == 0 or len(right) == 0:
                continue

            if check_ifreal(y):
                base = mse(y)
                gain = base - (len(left)/len(y))*mse(left) - (len(right)/len(y))*mse(right)
            else:
                base = entropy(y) if criterion == "information_gain" else gini_index(y)
                score_left = entropy(left) if criterion == "information_gain" else gini_index(left)
                score_right = entropy(right) if criterion == "information_gain" else gini_index(right)
                gain = base - (len(left)/len(y))*score_left - (len(right)/len(y))*score_right
            if gain > best_score:
                best_score = gain
                best_attr, best_val = feature, threshold
        else:
            gain = information_gain(y, X[feature], criterion)
            if gain > best_score:
                best_score = gain
                best_attr, best_val = feature, None

    return best_attr, best_val

def split_data(X: pd.DataFrame, y: pd.Series, attribute, value):
    """
    Split data based on discrete attribute or real threshold.
    """
    if value is None:  # discrete
        subsets = {}
        for v in X[attribute].unique():
            mask = (X[attribute] == v)
            subsets[v] = (X[mask].drop(columns=[attribute]), y[mask])
        return subsets
    else:  # real
        mask_left = X[attribute] <= value
        mask_right = X[attribute] > value
        left = (X[mask_left], y[mask_left])
        right = (X[mask_right], y[mask_right])
        return {"<=": left, ">": right}
