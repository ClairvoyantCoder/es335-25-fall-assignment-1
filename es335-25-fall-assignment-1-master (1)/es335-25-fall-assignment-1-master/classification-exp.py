import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *
from sklearn.datasets import make_classification

np.random.seed(42)

# ==== data ====
X_np, y_np = make_classification(
    n_features=2, n_redundant=0, n_informative=2, random_state=1,
    n_clusters_per_class=2, class_sep=0.5
)
X = pd.DataFrame(X_np, columns=["x1", "x2"])
y = pd.Series(y_np, dtype="category")

# quick scatter to visualize data
plt.figure()
plt.scatter(X["x1"], X["x2"], c=y.astype(int))
plt.title("Toy classification dataset")
plt.xlabel("x1"); plt.ylabel("x2")
plt.show()

# ==== 70/30 split ====
N = len(X)
idx = np.random.permutation(N)
cut = int(0.7 * N)
tr_idx, te_idx = idx[:cut], idx[cut:]
Xtr, Xte = X.iloc[tr_idx].reset_index(drop=True), X.iloc[te_idx].reset_index(drop=True)
ytr, yte = y.iloc[tr_idx].reset_index(drop=True), y.iloc[te_idx].reset_index(drop=True)

# train + evaluate with both criteria
for crit in ["information_gain", "gini_index"]:
    clf = DecisionTree(criterion=crit, max_depth=5)
    clf.fit(Xtr, ytr)
    yhat = clf.predict(Xte)

    print(f"\n== 70/30 split | criterion={crit} ==")
    print("Accuracy:", round(accuracy(yhat, yte), 4))
    for cls in y.cat.categories:
        print(f"Class {cls} -> Precision: {precision(yhat, yte, cls):.4f}, Recall: {recall(yhat, yte, cls):.4f}")

    print("\nTree structure:")
    clf.plot()

# ==== nested cross-validation to select best depth ====
def kfold_indices(n, k=5, shuffle=True, seed=42):
    idx = np.arange(n)
    if shuffle: np.random.default_rng(seed).shuffle(idx)
    folds = np.array_split(idx, k)
    return folds

def evaluate_depth(X, y, depth, inner_k=3, seed=123):
    folds = kfold_indices(len(X), k=inner_k, shuffle=True, seed=seed)
    scores = []
    for i in range(inner_k):
        val_idx = folds[i]
        tr_idx = np.hstack([folds[j] for j in range(inner_k) if j != i])
        Xtr, ytr = X.iloc[tr_idx].reset_index(drop=True), y.iloc[tr_idx].reset_index(drop=True)
        Xva, yva = X.iloc[val_idx].reset_index(drop=True), y.iloc[val_idx].reset_index(drop=True)
        clf = DecisionTree(criterion="information_gain", max_depth=depth)
        clf.fit(Xtr, ytr)
        yhat = clf.predict(Xva)
        scores.append(accuracy(yhat, yva))
    return np.mean(scores)

outer_folds = kfold_indices(N, k=5, shuffle=True, seed=2025)
depth_grid = list(range(1, 11))
outer_scores = []
chosen_depths = []

for i in range(5):
    te_idx = outer_folds[i]
    tr_idx = np.hstack([outer_folds[j] for j in range(5) if j != i])
    Xtr, ytr = X.iloc[tr_idx].reset_index(drop=True), y.iloc[tr_idx].reset_index(drop=True)
    Xte, yte = X.iloc[te_idx].reset_index(drop=True), y.iloc[te_idx].reset_index(drop=True)

    # inner CV to pick depth
    inner_scores = [evaluate_depth(Xtr, ytr, d, inner_k=3, seed=100 + i) for d in depth_grid]
    best_depth = depth_grid[int(np.argmax(inner_scores))]
    chosen_depths.append(best_depth)

    # retrain on outer-train with best depth, evaluate on outer-test
    clf = DecisionTree(criterion="information_gain", max_depth=best_depth)
    clf.fit(Xtr, ytr)
    yhat = clf.predict(Xte)
    score = accuracy(yhat, yte)
    outer_scores.append(score)

print("\n== Nested CV (outer 5-fold, inner 3-fold) ==")
print("Chosen depths per outer fold:", chosen_depths)
print("Outer-fold accuracies:", [round(s, 4) for s in outer_scores])
print("Mean outer accuracy:", round(float(np.mean(outer_scores)), 4))
print("Best depth overall (mode):", int(pd.Series(chosen_depths).mode()[0]))
