"""
You can add your own functions here according to your decision tree implementation.
There is no restriction on following the below template, these fucntions are here to simply help you.
"""

import pandas as pd
import numpy as np

def one_hot_encoding(X: pd.DataFrame) -> pd.DataFrame:
    """
    Function to perform one hot encoding on the input data
    """
    return pd.get_dummies(X)

def check_ifreal(y: pd.Series) -> bool:
    """
    Function to check if the given series has real or discrete values
    """
    return pd.api.types.is_float_dtype(y)


def entropy(Y: pd.Series) -> float:
    """
    Function to calculate the entropy
    """
    probs = Y.value_counts(normalize=True)
    return -np.sum(probs*np.log2(probs+1e-9))


def gini_index(Y: pd.Series) -> float:
    """
    Function to calculate the gini index
    """
    probs = Y.value_counts(normalize=True)
    return 1-np.sum(probs**2)


def information_gain(Y: pd.Series, attr: pd.Series, criterion: str) -> float:
    """
    Function to calculate the information gain using criterion (entropy, gini index or MSE)
    """
    if criterion == "entropy":
        base = entropy(Y)
    elif criterion == "gini":
        base = gini_index(Y)
    elif criterion == "mse":
        base = np.var(Y) #variance = mse before the split
    else:
        raise ValueError
    
    #weighted impurity after split
    values = attr.unique()
    weighted_impurity = 0
    for v in values:
        mask = (attr == v)
        Y_v = Y[mask]
        if criterion == "entropy":
            weighted_impurity += (len(Y_v) / len(Y)) * entropy(Y_v)
        elif criterion == "gini":
            weighted_impurity += (len(Y_v) / len(Y)) * gini_index(Y_v)
        elif criterion == "mse":
            weighted_impurity += (len(Y_v) / len(Y)) * np.var(Y_v)
    return base-weighted_impurity



def opt_split_attribute(X: pd.DataFrame, y: pd.Series, criterion, features: pd.Series):
    """
    Function to find the optimal attribute to split about.
    If needed you can split this function into 2, one for discrete and one for real valued features.
    You can also change the parameters of this function according to your implementation.

    features: pd.Series is a list of all the attributes we have to split upon

    return: attribute to split upon
    """

    # According to wheather the features are real or discrete valued and the criterion, find the attribute from the features series with the maximum information gain (entropy or varinace based on the type of output) or minimum gini index (discrete output).
    best_gain = -1
    best_attr = None
    best_value = None

    for attr in features:
        if pd.api.types.is_numeric_dtype(X[attr]):  # real-valued feature
            unique_vals = X[attr].unique()
            unique_vals.sort()
            # try midpoints between sorted values
            thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2
            for t in thresholds:
                mask = X[attr] <= t
                gain = information_gain(y, mask, criterion)  # mask as "attribute"
                if gain > best_gain:
                    best_gain, best_attr, best_value = gain, attr, t
        else:  # discrete/categorical feature
            gain = information_gain(y, X[attr], criterion)
            if gain > best_gain:
                best_gain, best_attr, best_value = gain, attr, None

    return best_attr, best_value, best_gain


def split_data(X: pd.DataFrame, y: pd.Series, attribute, value):
    """
    Funtion to split the data according to an attribute.
    If needed you can split this function into 2, one for discrete and one for real valued features.
    You can also change the parameters of this function according to your implementation.

    attribute: attribute/feature to split upon
    value: value of that attribute to split upon

    return: splitted data(Input and output)
    """

    # Split the data based on a particular value of a particular attribute. You may use masking as a tool to split the data.

    if value is None:  # categorical split
        splits = {}
        for v in X[attribute].unique():
            mask = (X[attribute] == v)
            splits[v] = (X[mask], y[mask])
        return splits
    else:  # real-valued split
        left_mask = (X[attribute] <= value)
        right_mask = (X[attribute] > value)
        return (X[left_mask], y[left_mask]), (X[right_mask], y[right_mask])
