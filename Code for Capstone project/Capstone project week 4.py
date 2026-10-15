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
FUNCTIONS = [f"function_{i}" for i in range(1, 9)]
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
    print(f"\nLoading Week {week} data...")

    inputs_file = f"{BASE}/Week{week}/Data/inputs.txt"
    outputs_file = f"{BASE}/Week{week}/Data/outputs.txt"

    print("Inputs:", inputs_file)
    print("Outputs:", outputs_file)

    with open(inputs_file, "r") as f:
        input_text = f.read()
        print("First part of inputs.txt:")
        print(input_text[:300])
        X_list = eval(input_text, {"array": np.array})

    with open(outputs_file, "r") as f:
        output_text = f.read()
        print("First part of outputs.txt:")
        print(output_text[:300])
        Y_list = eval(output_text, {"array": np.array})

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
    print(gp.kernel_)
    return gp


# ---------------------------------------------------------------------------
# Plot 1: the "Gaussian plot" — how well the GP currently fits the data
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Plot 1: Improved GP fit — actual vs predicted
# ---------------------------------------------------------------------------

def plot_gp_fit(fname, gp, X, Y):
    """
    Plot actual vs predicted outputs.

    Blue dot  = actual value
    Red dot   = GP predicted value
    Grey line = GP uncertainty (±1 standard deviation)
    Red dashed line = perfect prediction
    """

    y_pred, sigma = gp.predict(X, return_std=True)

    plt.figure(figsize=(7, 6))

    # ---------------------------------------------------------------
    # Perfect prediction line
    # ---------------------------------------------------------------
    all_values = np.concatenate([Y, y_pred])

    min_val = all_values.min()
    max_val = all_values.max()

    padding = 0.05 * (max_val - min_val)

    if padding == 0:
        padding = 0.001

    lims = [
        min_val - padding,
        max_val + padding
    ]

    plt.plot(
        lims,
        lims,
        "r--",
        linewidth=2,
        label="Perfect prediction"
    )

    # ---------------------------------------------------------------
    # Plot actual and predicted values
    # ---------------------------------------------------------------

    # Actual values
    plt.scatter(
        Y,
        Y,
        marker="o",
        s=70,
        label="Actual"
    )

    # Predicted values
    plt.scatter(
        Y,
        y_pred,
        marker="x",
        s=80,
        label="GP prediction"
    )

    # ---------------------------------------------------------------
    # Prediction uncertainty
    # ---------------------------------------------------------------

    for actual, prediction, uncertainty in zip(
        Y, y_pred, sigma
    ):
        plt.plot(
            [actual, actual],
            [
                prediction - uncertainty,
                prediction + uncertainty
            ],
            linewidth=1,
            alpha=0.5
        )

    # ---------------------------------------------------------------
    # Labels
    # ---------------------------------------------------------------

    plt.xlabel("Actual Output")
    plt.ylabel("GP Predicted Output")
    plt.title(f"{fname}: GP prediction vs actual")

    plt.xlim(lims)
    plt.ylim(lims)

    plt.grid(alpha=0.2)
    plt.legend()

    plt.tight_layout()
    plt.show()

    # ---------------------------------------------------------------
    # Print prediction accuracy
    # ---------------------------------------------------------------

    absolute_errors = np.abs(Y - y_pred)

    print(f"\n{fname} GP fit statistics")
    print("-" * 50)

    print(
        f"Mean absolute error: "
        f"{np.mean(absolute_errors):.6f}"
    )

    print(
        f"Maximum absolute error: "
        f"{np.max(absolute_errors):.6f}"
    )

    print(
        f"Average GP uncertainty: "
        f"{np.mean(sigma):.6f}"
)


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
# Plot: How well did GP predict previous submissions?
# ---------------------------------------------------------------------------

def plot_submission_predictions(results):
    """
    Plot actual vs predicted values for previous BO submissions.

    These are OUT-OF-SAMPLE predictions because the GP prediction
    was made before the actual result was added to the training data.
    """

    if len(results) == 0:
        print("No previous submissions to plot.")
        return

    actual = np.array([r["actual"] for r in results])
    predicted = np.array([r["predicted"] for r in results])
    std = np.array([r["std"] for r in results])

    plt.figure(figsize=(7, 6))

    # ---------------------------------------------------------------
    # Perfect prediction line
    # ---------------------------------------------------------------

    min_val = min(actual.min(), predicted.min())
    max_val = max(actual.max(), predicted.max())

    padding = 0.05 * (max_val - min_val)

    if padding == 0:
        padding = 0.001

    lims = [
        min_val - padding,
        max_val + padding
    ]

    plt.plot(
        lims,
        lims,
        "r--",
        linewidth=2,
        label="Perfect prediction"
    )

    # ---------------------------------------------------------------
    # Predictions with uncertainty
    # ---------------------------------------------------------------

    plt.errorbar(
        actual,
        predicted,
        yerr=std,
        fmt="o",
        markersize=8,
        capsize=5,
        alpha=0.8,
        label="Submission prediction ± 1 std"
    )

    # ---------------------------------------------------------------
    # Label each submission
    # ---------------------------------------------------------------

    for r, x, y in zip(results, actual, predicted):

        label = f"Week {r['week']}"

        plt.annotate(
            label,
            (x, y),
            xytext=(7, 7),
            textcoords="offset points"
        )

    plt.xlabel("Actual Output")
    plt.ylabel("GP Predicted Output")

    plt.title(
        f"{results[0]['function']}: "
        "Out-of-sample submission predictions"
    )

    plt.xlim(lims)
    plt.ylim(lims)

    plt.grid(alpha=0.2)
    plt.legend()

    plt.tight_layout()
    plt.show()

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

# ---------------------------------------------------------------------------
# Evaluate previous Bayesian Optimisation submissions
# ---------------------------------------------------------------------------

def evaluate_submissions(up_to_week):
    """
    Evaluate how appropriate each previous BO submission was.

    For Week n:
      - GP is trained using initial data + Weeks 1...(n-1)
      - Week n submission is predicted by that GP
      - Prediction is compared with the actual Week n output

    This gives a genuine out-of-sample evaluation.
    """

    print("\n" + "=" * 70)
    print("EVALUATION OF PREVIOUS SUBMISSIONS")
    print("=" * 70)

    results_by_function = {
        fname: []
        for fname in FUNCTIONS
    }

    # ---------------------------------------------------------------
    # Start with initial data only
    # ---------------------------------------------------------------

    history = {}

    for fname in FUNCTIONS:
        X, Y = load_initial_data(fname)
        history[fname] = {
            "X": X,
            "Y": Y
        }

    # ---------------------------------------------------------------
    # Evaluate each week
    # ---------------------------------------------------------------

    for week in range(1, up_to_week + 1):

        X_list, Y_list = load_week_data(week)

        print(f"\n{'=' * 70}")
        print(f"WEEK {week}")
        print(f"{'=' * 70}")

        for i, fname in enumerate(FUNCTIONS):

            # -----------------------------------------------------------
            # 1. Fit GP WITHOUT seeing this week's result
            # -----------------------------------------------------------

            X_train = history[fname]["X"]
            Y_train = history[fname]["Y"]

            gp = fit_gp(X_train, Y_train)

            # -----------------------------------------------------------
            # 2. Get actual submitted point and result
            # -----------------------------------------------------------

            submitted_X = np.asarray(
                X_list[i]
            ).reshape(1, -1)

            actual_Y = np.asarray(
                Y_list[i]
            ).ravel()[0]

            # -----------------------------------------------------------
            # 3. Predict BEFORE seeing actual result
            # -----------------------------------------------------------

            predicted_Y, predicted_std = gp.predict(
                submitted_X,
                return_std=True
            )

            predicted_Y = predicted_Y[0]
            predicted_std = predicted_std[0]

            # -----------------------------------------------------------
            # 4. Calculate UCB
            # -----------------------------------------------------------

            ucb = (
                predicted_Y
                + KAPPA * predicted_std
            )

            # -----------------------------------------------------------
            # 5. Prediction error
            # -----------------------------------------------------------

            absolute_error = abs(
                actual_Y - predicted_Y
            )

            if abs(actual_Y) > 1e-10:
                percentage_error = (
                    absolute_error
                    / abs(actual_Y)
                ) * 100
            else:
                percentage_error = np.nan

            # -----------------------------------------------------------
            # 6. Compare with previous best
            # -----------------------------------------------------------

            previous_best = np.max(Y_train)

            improvement = (
                actual_Y - previous_best
            )

            TOL = 1e-8

            if improvement > TOL:
                result = "IMPROVED BEST"

            elif abs(improvement) <= TOL:
                result = "MATCHED BEST"

            else:
                result = "DID NOT IMPROVE BEST"

            # -----------------------------------------------------------
            # 7. Exploration contribution
            # -----------------------------------------------------------

            exploration = (
                KAPPA * predicted_std
            )

            # -----------------------------------------------------------
            # 8. Store results
            # -----------------------------------------------------------

            results_by_function[fname].append({

                "week": week,

                "function": fname,

                "actual": actual_Y,

                "predicted": predicted_Y,

                "std": predicted_std,

                "ucb": ucb,

                "error": absolute_error,

                "previous_best": previous_best,

                "improvement": improvement,

                "exploration": exploration
            })

            # -----------------------------------------------------------
            # 9. Print results
            # -----------------------------------------------------------

            print(f"\n{fname}")
            print("-" * 40)

            print(
                f"Submitted point : "
                f"{np.round(submitted_X[0], 4)}"
            )

            print(
                f"Predicted output: "
                f"{predicted_Y:.6f}"
            )

            print(
                f"Predicted std   : "
                f"{predicted_std:.6f}"
            )

            print(
                f"UCB             : "
                f"{ucb:.6f}"
            )

            print(
                f"Actual output   : "
                f"{actual_Y:.6f}"
            )

            print(
                f"Absolute error  : "
                f"{absolute_error:.6f}"
            )

            if not np.isnan(percentage_error):

                print(
                    f"Percentage error: "
                    f"{percentage_error:.2f}%"
                )

            print(
                f"Previous best   : "
                f"{previous_best:.6f}"
            )

            print(
                f"Improvement     : "
                f"{improvement:.6f}"
            )

            print(
                f"Result          : "
                f"{result}"
            )

            print(
                f"Exploration contribution: "
                f"{exploration:.6f}"
            )

            # -----------------------------------------------------------
            # 10. Add this week's observation to history
            # -----------------------------------------------------------

            history[fname]["X"] = np.vstack(
                [
                    history[fname]["X"],
                    submitted_X
                ]
            )

            history[fname]["Y"] = np.append(
                history[fname]["Y"],
                actual_Y
            )

            # -----------------------------------------------------------
            # 11. Statistics AFTER adding this week's observation
            # -----------------------------------------------------------

            Y_current = history[fname]["Y"]

            print(
                f"Data statistics after Week {week}:"
            )

            print(
                f"Min : {np.min(Y_current):.6f}"
            )

            print(
                f"Max : {np.max(Y_current):.6f}"
            )

            print(
                f"Mean: {np.mean(Y_current):.6f}"
            )

            print(
                f"Std : {np.std(Y_current):.6f}"
            )

    # -------------------------------------------------------------------
    # Plot all submission predictions AFTER all weeks have been evaluated
    # -------------------------------------------------------------------

    for fname in FUNCTIONS:

        plot_submission_predictions(
            results_by_function[fname]
        )

    # -------------------------------------------------------------------
    # Return results
    # -------------------------------------------------------------------

    return results_by_function
            


if __name__ == "__main__":
    # Set this to the number of weeks of data you have so far (e.g. 2 after
    # Week 2's outputs are in). Re-run after each week.
    next_points = run_week(up_to_week=3)
    # Evaluate how good the previous submissions were
    results = evaluate_submissions(up_to_week=3)