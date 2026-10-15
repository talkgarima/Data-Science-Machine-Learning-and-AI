# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 08:53:56 2026

@author: talkg
"""

"""
Capstone Bayesian Optimisation pipeline — simplified.

One loop handles all 8 functions (any input dimension, 2-8), instead of
8 near-duplicate blocks. Each week it:
  1. Rebuilds each function's full history (initial data + every week so far)
  2. Fits a GP surrogate
  3. Plots the GP fit ("Gaussian plot")
  4. Plots the UCB acquisition scores ("acquisition plot")
  5. Prints the recommended next point to submit

USAGE
-----
Change BASE to your actual path, then at the bottom set:
    next_points = run_week(up_to_week=2)
to whichever week you just completed (i.e. how many Week{n}/Data folders
exist), and run the script. It regenerates both plots for every function.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

BASE = "C:/Data Science and AI/Capstone Project"   # <-- edit if needed
#FUNCTIONS = "function_2"
FUNCTIONS = [f"function_{i}" for i in range(1, 2)]
KAPPA = 2.0  # UCB exploration weight 


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_initial_data(fname):
    folder = f"{BASE}/Initial_data_points_starter/initial_data/{fname}"
    X = np.load(f"{folder}/initial_inputs.npy")
    Y = np.load(f"{folder}/initial_outputs.npy").ravel()
    return X, Y


def load_week_data(week):
    """Each file holds a list of 8 arrays/values, one per function, in
    Function_1..Function_8 order — same format you were already using."""
    with open(f"{BASE}/Week{week}/Data/inputs.txt", "r") as f:
        X_list = eval(f.read(), {"array": np.array})
    with open(f"{BASE}/Week{week}/Data/outputs.txt", "r") as f:
        Y_list = eval(f.read(), {"array": np.array})
    return X_list, Y_list


def build_history(up_to_week):
    """Initial data + every week's data so far, per function."""
    history = {}
    for fname in FUNCTIONS:
        X, Y = load_initial_data(fname)
        history[fname] = {"X": X, "Y": Y}

    for week in range(1, up_to_week + 1):
        X_list, Y_list = load_week_data(week)
        for i, fname in enumerate(FUNCTIONS):
            history[fname]["X"] = np.vstack([history[fname]["X"], X_list[i]])
            history[fname]["Y"] = np.append(history[fname]["Y"], Y_list[i])

    return history


# ---------------------------------------------------------------------------
# Surrogate model
# ---------------------------------------------------------------------------

def fit_gp(X, Y):
    kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel()
    gp = GaussianProcessRegressor(
        kernel=kernel, normalize_y=True, random_state=42, n_restarts_optimizer=5
    )
    gp.fit(X, Y)
    return gp


# ---------------------------------------------------------------------------
# Plot 1: the "Gaussian plot" — how well the GP currently fits the data
# ---------------------------------------------------------------------------

def plot_gp_fit(fname, gp, X, Y):
    y_pred, sigma = gp.predict(X, return_std=True)

    plt.figure(figsize=(5, 5))
    plt.errorbar(Y, y_pred, yerr=sigma, fmt="o", alpha=0.6, ecolor="lightgray")
    lims = [min(Y.min(), y_pred.min()), max(Y.max(), y_pred.max())]
    plt.plot(lims, lims, "r--")
    plt.xlabel("Actual Output")
    plt.ylabel("Predicted Output (\u00b1 std)")
    plt.title(f"{fname}: GP fit (actual vs predicted)")
    plt.tight_layout()
    plt.show()

    # Bonus: full mean surface, only possible to draw directly for 2 inputs
    if X.shape[1] == 2:
        g = np.linspace(0, 1, 60)
        gx, gy = np.meshgrid(g, g)
        grid = np.column_stack([gx.ravel(), gy.ravel()])
        mean, _ = gp.predict(grid, return_std=True)

        plt.figure(figsize=(6, 5))
        plt.contourf(gx, gy, mean.reshape(gx.shape), levels=30, cmap="viridis")
        plt.colorbar(label="GP mean")
        plt.scatter(X[:, 0], X[:, 1], c="red", edgecolor="white", s=40)
        plt.xlabel("Input 1")
        plt.ylabel("Input 2")
        plt.title(f"{fname}: GP mean surface")
        plt.tight_layout()
        plt.show()


# ---------------------------------------------------------------------------
# Plot 2: the acquisition plot — where to query next (UCB, as in your code)
# ---------------------------------------------------------------------------

def acquisition_ucb(gp, candidates, kappa=KAPPA):
    mean, std = gp.predict(candidates, return_std=True)
    return mean + kappa * std


def plot_acquisition(fname, gp, X, kappa=KAPPA, n_candidates=10000, seed=0):
    dim = X.shape[1]
    rng = np.random.default_rng(seed)
    candidates = rng.random((n_candidates, dim))
    ucb = acquisition_ucb(gp, candidates, kappa)
    best_idx = np.argmax(ucb)
    best_point = candidates[best_idx]

    if dim == 2:
        g = np.linspace(0, 1, 60)
        gx, gy = np.meshgrid(g, g)
        grid = np.column_stack([gx.ravel(), gy.ravel()])
        ucb_grid = acquisition_ucb(gp, grid, kappa)

        plt.figure(figsize=(6, 5))
        plt.contourf(gx, gy, ucb_grid.reshape(gx.shape), levels=30, cmap="plasma")
        plt.colorbar(label="UCB score")
        plt.scatter(X[:, 0], X[:, 1], c="white", edgecolor="black", s=40, label="observed")
        plt.scatter(best_point[0], best_point[1], c="red", marker="*", s=200, label="next point")
        plt.legend()
        plt.xlabel("Input 1")
        plt.ylabel("Input 2")
        plt.title(f"{fname}: Acquisition (UCB)")
        plt.tight_layout()
        plt.show()
    else:
        top = np.argsort(ucb)[::-1][:30]
        plt.figure(figsize=(7, 4))
        plt.bar(range(len(top)), ucb[top])
        plt.xlabel("Top 30 candidates (ranked)")
        plt.ylabel("UCB score")
        plt.title(f"{fname}: Top acquisition scores")
        plt.tight_layout()
        plt.show()

    print(f"{fname}: next point to submit -> {np.round(best_point, 4)}  (UCB={ucb[best_idx]:.4f})")
    return best_point


# ---------------------------------------------------------------------------
# Run one week's analysis for all 8 functions
# ---------------------------------------------------------------------------

def run_week(up_to_week):
    history = build_history(up_to_week)
    next_points = {}

    for fname in FUNCTIONS:
        X, Y = history[fname]["X"], history[fname]["Y"]
        gp = fit_gp(X, Y)
        plot_gp_fit(fname, gp, X, Y)
        next_points[fname] = plot_acquisition(fname, gp, X)

    return next_points


if __name__ == "__main__":
    # Set this to the number of weeks of data you have so far (e.g. 2 after
    # Week 2's outputs are in). Re-run after each week.
    next_points = run_week(up_to_week=1)