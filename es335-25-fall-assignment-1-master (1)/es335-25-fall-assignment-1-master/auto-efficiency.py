import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *
from sklearn.tree import DecisionTreeRegressor

np.random.seed(42)

# ---- load & clean UCI Auto MPG ----
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
cols = ["mpg","cylinders","displacement","horsepower","weight","acceleration","model_year","origin","car_name"]
data = pd.read_csv(url, delim_whitespace=True, header=None, names=cols, na_values="?")

# drop rows with missing targets or features we’ll use
data = data.dropna(subset=["mpg","cylinders","displacement","horsepower","weight","acceleration","model_year","origin"])
# cast dtypes
data["cylinders"] = data["cylinders"].astype(int)
data["model_year"] = data["model_year"].astype(int)
data["origin"] = data["origin"].astype(int)

# features & target (drop car_name)
X = data.drop(columns=["mpg","car_name"]).reset_index(drop=True)
y = data["mpg"].reset_index(drop=True)

# 70/30 split
N = len(X)
idx = np.random.permutation(N)
cut = int(0.7 * N)
tr_idx, te_idx = idx[:cut], idx[cut:]
Xtr, Xte = X.iloc[tr_idx].reset_index(drop=True), X.iloc[te_idx].reset_index(drop=True)
ytr, yte = y.iloc[tr_idx].reset_index(drop=True), y.iloc[te_idx].reset_index(drop=True)

# ---- our DecisionTree (regression uses MSE internally; criterion arg ignored) ----
dt = DecisionTree(criterion="information_gain", max_depth=6)
dt.fit(Xtr, ytr)
yhat = dt.predict(Xte)

print("== Our DecisionTree (Regression) ==")
print("RMSE:", round(rmse(yhat, yte), 4))
print("MAE :", round(mae(yhat, yte), 4))
print("\nTree:")
dt.plot()

# ---- scikit-learn DecisionTreeRegressor comparison ----
sk = DecisionTreeRegressor(max_depth=6, random_state=42)
sk.fit(Xtr, ytr)
yhat_sk = pd.Series(sk.predict(Xte))

print("\n== sklearn.tree.DecisionTreeRegressor ==")
print("RMSE:", round(rmse(yhat_sk, yte), 4))
print("MAE :", round(mae(yhat_sk, yte), 4))

# plot predictions vs ground truth
plt.figure()
plt.scatter(yte, yhat, label="Our DT", alpha=0.7)
plt.scatter(yte, yhat_sk, label="sklearn DT", alpha=0.7, marker="x")
miny, maxy = float(yte.min()), float(yte.max())
plt.plot([miny, maxy], [miny, maxy])
plt.xlabel("True MPG"); plt.ylabel("Predicted MPG")
plt.legend(); plt.title("Auto MPG: Predictions vs True")
plt.show()
