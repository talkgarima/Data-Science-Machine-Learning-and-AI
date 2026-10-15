```python
# ============================================================
# FUNCTION 5 - NEURAL NETWORK SURROGATE ANALYSIS
# ============================================================
#
# Purpose:
# 1. Train a small neural network on Function 5
# 2. Compare NN predictions with actual outputs
# 3. Calculate gradients for each of the 4 inputs
# 4. Identify which inputs have the greatest influence
#
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score


# ============================================================
# 1. LOAD FUNCTION 5 DATA
# ============================================================

# CHANGE THESE PATHS TO YOUR ACTUAL FILE LOCATIONS

INPUT_FILE = r"C:/Data Science and AI/Capstone Project/Initial_data_points_starter/initial_data/function_5/initial_inputs.npy"
OUTPUT_FILE = r"C:/Data Science and AI/Capstone Project/Initial_data_points_starter/initial_data/function_5/initial_outputs.npy"

X = np.load(INPUT_FILE)
y = np.load(OUTPUT_FILE)

# Make sure y is one-dimensional
y = y.ravel()

print("Input shape:", X.shape)
print("Output shape:", y.shape)

print("\nFirst 5 inputs:")
print(X[:5])

print("\nFirst 5 outputs:")
print(y[:5])


# ============================================================
# 2. ADD YOUR WEEKLY DATA IF AVAILABLE
# ============================================================
#
# If you want the NN to use ALL your observations up to Week 4,
# add the weekly points here.
#
# Your Function 5 points from the earlier weeks were:
#
# Week 1:
# [0.25, 0.3, 0.9, 0.7] -> 136.920025
#
# Week 2:
# [0.4054, 0.9465, 0.9911, 0.8934] -> 2693.675237
#
# Week 3:
# [0.9322, 0.9281, 0.9819, 0.9831] -> 5840.000772
#
# Add Week 4 below if you have it.

weekly_X = np.array([
    [0.25,   0.3,    0.9,    0.7],
    [0.4054, 0.9465, 0.9911, 0.8934],
    [0.9322, 0.9281, 0.9819, 0.9831],
    
    # ADD WEEK 4 HERE IF YOU HAVE IT
    # [x1, x2, x3, x4],
])

weekly_y = np.array([
    136.920025,
    2693.675237,
    5840.000772,
    
    # ADD WEEK 4 OUTPUT HERE
    # output,
])


# Combine initial + weekly data
X = np.vstack([X, weekly_X])
y = np.concatenate([y, weekly_y])

print("\nTotal observations:", len(y))
print("Number of inputs:", X.shape[1])


# ============================================================
# 3. STANDARDISE THE INPUTS AND OUTPUT
# ============================================================

X_scaler = StandardScaler()
y_scaler = StandardScaler()

X_scaled = X_scaler.fit_transform(X)
y_scaled = y_scaler.fit_transform(y.reshape(-1, 1)).ravel()


# ============================================================
# 4. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y_scaled,
    test_size=0.2,
    random_state=42
)


# ============================================================
# 5. BUILD A SMALL NEURAL NETWORK
# ============================================================

model = keras.Sequential([
    
    layers.Input(shape=(4,)),
    
    layers.Dense(16, activation="relu"),
    
    layers.Dense(16, activation="relu"),
    
    layers.Dense(1)
])


model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.01),
    loss="mse"
)


# ============================================================
# 6. TRAIN THE NETWORK
# ============================================================

history = model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    epochs=300,
    batch_size=8,
    verbose=0
)


print("\nNeural network training complete.")


# ============================================================
# 7. EVALUATE THE MODEL
# ============================================================

pred_scaled = model.predict(X_test, verbose=0).ravel()

# Convert predictions back to original output scale
pred = y_scaler.inverse_transform(
    pred_scaled.reshape(-1, 1)
).ravel()

actual = y_scaler.inverse_transform(
    y_test.reshape(-1, 1)
).ravel()


mae = mean_absolute_error(actual, pred)
r2 = r2_score(actual, pred)


print("\n==============================")
print("NEURAL NETWORK PERFORMANCE")
print("==============================")

print("Mean Absolute Error:", mae)
print("R²:", r2)


# ============================================================
# 8. ACTUAL VS PREDICTED
# ============================================================

plt.figure(figsize=(7, 6))

plt.scatter(actual, pred)

minimum = min(actual.min(), pred.min())
maximum = max(actual.max(), pred.max())

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    "--"
)

plt.xlabel("Actual output")
plt.ylabel("NN predicted output")

plt.title("Function 5 - Neural Network Predictions")

plt.grid(alpha=0.2)

plt.tight_layout()

plt.show()


# ============================================================
# 9. TRAINING CURVE
# ============================================================

plt.figure(figsize=(7, 5))

plt.plot(history.history["loss"], label="Training loss")
plt.plot(history.history["val_loss"], label="Validation loss")

plt.xlabel("Epoch")
plt.ylabel("MSE")

plt.title("Function 5 - Neural Network Training")

plt.legend()

plt.grid(alpha=0.2)

plt.tight_layout()

plt.show()


# ============================================================
# 10. CALCULATE GRADIENTS
# ============================================================
#
# The gradient tells us how sensitive the NN output is
# to each input.
#
# Large absolute gradient =
# greater influence on predicted output.
#
# ============================================================

X_tensor = tf.convert_to_tensor(
    X_scaled,
    dtype=tf.float32
)

with tf.GradientTape() as tape:

    tape.watch(X_tensor)

    predictions_scaled = model(X_tensor)


gradients_scaled = tape.gradient(
    predictions_scaled,
    X_tensor
).numpy()


# ============================================================
# 11. CONVERT GRADIENTS BACK TO ORIGINAL SCALE
# ============================================================

#
# Because both X and y were standardised, the gradient from
# the neural network is initially in standardised units.
#
# Convert it approximately back to:
#
# change in output / change in original input
#

output_std = y_scaler.scale_[0]
input_std = X_scaler.scale_

gradients_original = (
    gradients_scaled * output_std / input_std
)


# ============================================================
# 12. AVERAGE ABSOLUTE GRADIENT
# ============================================================

mean_abs_gradient = np.mean(
    np.abs(gradients_original),
    axis=0
)


print("\n==============================")
print("INPUT GRADIENT ANALYSIS")
print("==============================")


for i, gradient in enumerate(mean_abs_gradient):

    print(
        f"Input {i+1}: "
        f"Mean absolute gradient = {gradient:.4f}"
    )


# ============================================================
# 13. RANK INPUT VARIABLES
# ============================================================

ranking = np.argsort(
    mean_abs_gradient
)[::-1]


print("\n==============================")
print("INPUT IMPORTANCE RANKING")
print("==============================")


for rank, index in enumerate(ranking, start=1):

    print(
        f"{rank}. Input {index+1} "
        f"(gradient = {mean_abs_gradient[index]:.4f})"
    )


# ============================================================
# 14. PLOT INPUT IMPORTANCE
# ============================================================

input_names = [
    "Input 1",
    "Input 2",
    "Input 3",
    "Input 4"
]

plt.figure(figsize=(8, 5))

plt.bar(
    input_names,
    mean_abs_gradient
)

plt.xlabel("Input variable")

plt.ylabel("Mean absolute gradient")

plt.title(
    "Function 5 - Neural Network Input Influence"
)

plt.grid(
    axis="y",
    alpha=0.2
)

plt.tight_layout()

plt.show()


# ============================================================
# 15. SHOW GRADIENT DIRECTION
# ============================================================
#
# Positive gradient:
# increasing the input tends to increase predicted output.
#
# Negative gradient:
# increasing the input tends to decrease predicted output.
#
# ============================================================

mean_gradient = np.mean(
    gradients_original,
    axis=0
)


print("\n==============================")
print("GRADIENT DIRECTION")
print("==============================")


for i, gradient in enumerate(mean_gradient):

    if gradient > 0:
        direction = "Increasing input tends to increase output"
    else:
        direction = "Increasing input tends to decrease output"

    print(
        f"Input {i+1}: "
        f"{gradient:.4f} -> {direction}"
    )


# ============================================================
# 16. FIND THE GRADIENT AT YOUR BEST OBSERVATION
# ============================================================

best_index = np.argmax(y)

best_X = X_scaled[best_index:best_index+1]

best_tensor = tf.convert_to_tensor(
    best_X,
    dtype=tf.float32
)

with tf.GradientTape() as tape:

    tape.watch(best_tensor)

    best_prediction = model(best_tensor)


best_gradient_scaled = tape.gradient(
    best_prediction,
    best_tensor
).numpy()[0]


best_gradient_original = (
    best_gradient_scaled
    * output_std
    / input_std
)


print("\n==============================")
print("GRADIENT AT BEST OBSERVATION")
print("==============================")


print("Best input:")
print(X[best_index])

print("\nBest actual output:")
print(y[best_index])

print("\nGradients:")

for i, gradient in enumerate(best_gradient_original):

    print(
        f"Input {i+1}: {gradient:.4f}"
    )


# ============================================================
# 17. SIMPLE INTERPRETATION
# ============================================================

strongest_input = np.argmax(
    np.abs(best_gradient_original)
)

print("\n==============================")
print("INTERPRETATION")
print("==============================")

print(
    f"At the best observed point, "
    f"Input {strongest_input + 1} "
    f"has the largest local gradient."
)

print(
    "This suggests that this input is the most "
    "sensitive variable locally according to the NN."
)

print(
    "For maximisation, a positive gradient suggests "
    "that increasing the variable may increase the output, "
    "while a negative gradient suggests decreasing it."
)
