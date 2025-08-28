import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from tree.base import DecisionTree

np.random.seed(42)
num_average_time = 50  # adjust if runs are slow

def make_data(N, M, case):
    """
    case in {"disc_disc","real_disc","real_real","disc_real"}
    X: DataFrame, y: Series (category for discrete y, float for real y)
    Discrete X are ints in {0,1}; Real X ~ N(0,1).
    """
    if "disc" in case.split("_")[0]:
        X = pd.DataFrame(np.random.randint(0, 2, size=(N, M)), columns=[f"x{i}" for i in range(M)]).astype("category")
    else:
        X = pd.DataFrame(np.random.randn(N, M), columns=[f"x{i}" for i in range(M)])
    if "disc" in case.split("_")[1]:
        y = pd.Series(np.random.randint(0, 2, size=N), dtype="category")
    else:
        y = pd.Series(np.random.randn(N))
    return X, y

def time_fit_predict(X, y, criterion, max_depth=6, reps=5):
    t_fit, t_pred = [], []
    for _ in range(reps):
        model = DecisionTree(criterion=criterion, max_depth=max_depth)
        t0 = time.perf_counter()
        model.fit(X, y)
        t1 = time.perf_counter()
        _ = model.predict(X)
        t2 = time.perf_counter()
        t_fit.append(t1 - t0)
        t_pred.append(t2 - t1)
    return np.mean(t_fit), np.std(t_fit), np.mean(t_pred), np.std(t_pred)

def run_sweep(N_list, M_list, case):
    fit_times, pred_times = [], []
    for N in N_list:
        for M in M_list:
            X, y = make_data(N, M, case)
            crit = "information_gain"  # for classification; ignored for regression
            tf_mean, tf_std, tp_mean, tp_std = time_fit_predict(X, y, crit, reps=3)
            fit_times.append({"N": N, "M": M, "fit_mean": tf_mean, "fit_std": tf_std})
            pred_times.append({"N": N, "M": M, "pred_mean": tp_mean, "pred_std": tp_std})
    return pd.DataFrame(fit_times), pd.DataFrame(pred_times)

def plot_grid(df, value_col, title):
    # plot N on x, different curves for M
    plt.figure()
    Ms = sorted(df["M"].unique())
    Ns = sorted(df["N"].unique())
    for M in Ms:
        series = []
        for N in Ns:
            v = df[(df["N"] == N) & (df["M"] == M)][value_col].values
            series.append(v[0] if len(v) else np.nan)
        plt.plot(Ns, series, label=f"M={M}")
    plt.xlabel("N (samples)")
    plt.ylabel(value_col)
    plt.title(title)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    N_list = [200, 400, 800, 1600]
    M_list = [8, 16, 32]

    cases = {
        "disc_disc": "Discrete X, Discrete y",
        "real_disc": "Real X, Discrete y",
        "real_real": "Real X, Real y",
        "disc_real": "Discrete X, Real y",
    }

    for case_key, case_name in cases.items():
        print(f"\n=== {case_name} ===")
        df_fit, df_pred = run_sweep(N_list, M_list, case_key)
        plot_grid(df_fit, "fit_mean", f"Fit time vs N ({case_name})")
        plot_grid(df_pred, "pred_mean", f"Predict time vs N ({case_name})")

    print("\nInterpretation (theory vs. empirical):")
    print("- Building a DT is approximately O(N * M * log N) with simple median/branching,")
    print("  since each split scans features (M) and partitions data; depth scales with log N.")
    print("- Prediction is roughly O(depth) ≈ O(log N) per sample; on a batch of size N, ~ O(N log N).")
    print("- Curves should show near-linear growth with N for fit (with a log factor) and linear for predict across a fixed depth.")
