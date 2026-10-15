import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

# Week 2 Input


with open('C:/Data Science and AI/Capstone Project/Week2/Data/inputs.txt', 'r') as f:
    text = f.read()

Week2_Input = eval(text, {"array": np.array})

print(Week2_Input)


function_names = [
    "Function_1_X_week2",
    "Function_2_X_week2",
    "Function_3_X_week2",
    "Function_4_X_week2",
    "Function_5_X_week2",
    "Function_6_X_week2",
    "Function_7_X_week2",
    "Function_8_X_week2"
]

functions_X = dict(zip(function_names, Week2_Input))

print(functions_X["Function_8_X_week2"])

#Week 2 Outputs

with open('C:/Data Science and AI/Capstone Project/Week2/Data/outputs.txt', 'r') as f:
    text = f.read()

Week2_Outputs = eval(text, {"array": np.array})

print(Week2_Outputs)


function_names = [
    "Function_1_Y_week2",
    "Function_2_Y_week2",
    "Function_3_Y_week2",
    "Function_4_Y_week2",
    "Function_5_Y_week2",
    "Function_6_Y_week2",
    "Function_7_Y_week2",
    "Function_8_Y_week2"
]

functions_Y = dict(zip(function_names, Week2_Outputs))

print(functions_Y["Function_8_Y_week2"])


# Function 1
Function_1_input = np.load('C:/Data Science and AI/Capstone Project/Initial_data_points_starter/initial_data/function_1/initial_inputs.npy')
Function_1_input_week2 = np.vstack([Function_1_input, functions["Function_1_X_week2"]])




Function_1_output = np.load('C:/Data Science and AI/Capstone Project/Initial_data_points_starter/initial_data/function_1/initial_outputs.npy')
Function_1_output_week2 = np.append(Function_1_output, functions["Function_1_Y_week2"])






X=Function_1_input_week2
Y=Function_1_output_week2

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(X[:,0], X[:,1], Y)

for i in range(len(Y)):
    ax.text(
        X[i,0], X[i,1], Y[i],
        f"({X[i,0]:.2f}, {X[i,1]:.2f})",
        fontsize=8
    )
    
ax.set_xlabel("Input 1")
ax.set_ylabel("Input 2")
ax.set_zlabel("Output")

plt.show()

#Week 2Comments
# Week 1 submission is very close to 0. I am switching to exploitation as I think I am close to 0
0.15,0.95

# 3d plot shows when y axis is above .70 we get higher output.
#Since there are 4 input dat apoints exhibiting this behaviour, I want to explore more towards lower x but higher y

# 0.2-0.9

#Function 2

Function_2_input = np.load('C:/Data Science and AI/Capstone Project/Initial_data_points_starter/initial_data/function_1/initial_inputs.npy')
Function_2_input_week2 = np.vstack([Function_2_input, functions["Function_2_X_week2"]])

Function_2_output = np.load('C:/Data Science and AI/Capstone Project/Initial_data_points_starter/initial_data/function_1/initial_outputs.npy')
Function_2_output_week2 = np.append(Function_2_output, functions["Function_2_Y_week2"])




X=Function_2_input_week2
Y=Function_2_output_week2

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(X[:,0], X[:,1], Y)

for i in range(len(Y)):
    ax.text(
        X[i,0], X[i,1], Y[i],
        f"({X[i,0]:.2f}, {X[i,1]:.2f})",
        fontsize=8
    )
    
ax.set_xlabel("Input 1")
ax.set_ylabel("Input 2")
ax.set_zlabel("Output")

plt.show()

# Same logic as above 0.3,0.99


#Function 3

Function_3_input = np.load('function_3/initial_inputs.npy')
Function_3_output = np.load('function_3/initial_outputs.npy')



X=Function_3_input
Y=Function_3_output

#Plot each input against the output

fig, ax = plt.subplots(1, 3, figsize=(15,4))

for i in range(3):
    ax[i].scatter(X[:, i], Y)
    ax[i].set_xlabel(f'Input {i+1}')
    ax[i].set_ylabel('Output')

plt.tight_layout()
plt.show()

#Input 1  Seasonality with multiple peaks in the data. check 0.9
# Input 2 two peaks, when the input issmall and big. check middle value0.55
# Input 3 Peak at input 0 and 0.4. check 0.7

#0.9-0.55-0.7

# Colour a 2d scatter plot

# Input 1 vs Input 2- No trend
plt.figure(figsize=(6,5))

plt.scatter(X[:,0], X[:,1], c=Y, cmap='viridis', s=80)

plt.xlabel("Input 1")
plt.ylabel("Input 2")
plt.colorbar(label="Output")

plt.show()

# Input 1 vs Input 3- Higher data point when input 3 is smaller
plt.figure(figsize=(6,5))

plt.scatter(X[:,0], X[:,2], c=Y, cmap='viridis', s=80)

plt.xlabel("Input 1")
plt.ylabel("Input 3")
plt.colorbar(label="Output")

plt.show()

# Input 2 vs Input 3- Input 3 prefers smaller values and input 2 seems to be having two peaks
plt.figure(figsize=(6,5))

plt.scatter(X[:,1], X[:,2], c=Y, cmap='viridis', s=80)

plt.xlabel("Input 2")
plt.ylabel("Input 3")
plt.colorbar(label="Output")

plt.show()

#######################
#Make a 3D scatter plot- difficult to interpret
#######################

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(
    X[:,0],
    X[:,1],
    X[:,2],
    c=Y,
    cmap='viridis',
    s=80
)

ax.set_xlabel("Input 1")
ax.set_ylabel("Input 2")
ax.set_zlabel("Input 3")

plt.colorbar(scatter, label="Output")

plt.show()

#######################
#Calculate correlations
#######################

import pandas as pd

df = pd.DataFrame(X, columns=["Input1", "Input2", "Input3"])
df["Output"] = Y

print(df.corr())

#Week 1 submisison
#0.9-0.55-0.7



#FUNCTION 4

Function_4_input = np.load('function_4/initial_inputs.npy')
Function_4_output = np.load('function_4/initial_outputs.npy')



X=Function_4_input
Y=Function_4_output

#Plot each input against the output

fig, ax = plt.subplots(1, 4, figsize=(15,4))

for i in range(4):
    ax[i].scatter(X[:, i], Y)
    ax[i].set_xlabel(f'Input {i+1}')
    ax[i].set_ylabel('Output')

plt.tight_layout()
plt.show()

#Input 1 seems negatively correlated with output 0.75
# Input 2 negative correlated with lots of noise  0.3
# Input 3 Peak at centre 0.1
# Input 4 seems negatively correlated with output 0.35

#0.75-0.3-0.1-0.35


#######################
#Calculate correlations
#######################

import pandas as pd

df = pd.DataFrame(X, columns=["Input1", "Input2", "Input3","Input4"])
df["Output"] = Y

print(df.corr())


#FUNCTION 5

Function_5_input = np.load('function_5/initial_inputs.npy')
Function_5_output = np.load('function_5/initial_outputs.npy')



X=Function_5_input
Y=Function_5_output

#Plot each input against the output

fig, ax = plt.subplots(1, 4, figsize=(20,4))

for i in range(4):
    ax[i].scatter(X[:, i], Y)
    ax[i].set_xlabel(f'Input {i+1}')
    ax[i].set_ylabel('Output')

plt.tight_layout()
plt.show()

#Input 1 seems not affecting the output. But there could be two small peaks hidden in there 0.25
# Input 2 same as Input 1  0.3
# Input 3 Peak at centre 0.9
# Input 4 increasing with input variable. 0.7

#0.25-0.3-0.9-0.7


#FUNCTION 6- No clear trend with indiviudal variables. That is more likely because of interaction between two variables
#0.33392848, 0.26935535, 0.49023239, 0.73847362, 0.04950843
Function_6_input = np.load('function_6/initial_inputs.npy')
Function_6_output = np.load('function_6/initial_outputs.npy')



X=Function_6_input
Y=Function_6_output

#Plot each input against the output

fig, ax = plt.subplots(1, 5, figsize=(20,4))

for i in range(5):
    ax[i].scatter(X[:, i], Y)
    ax[i].set_xlabel(f'Input {i+1}')
    ax[i].set_ylabel('Output')

plt.tight_layout()
plt.show()

#Input 1 Not a clear upward or downward trend
# Input 2 May be more tendency of having higher ouput at low input 2
# Input 3 some higher ouputs at both low and higher input 
# Input 4 slight increasing trend
# Input 5 negative trend

# with increase in colinearity it is getting mor edifficult to interpret the interaction
# Hence Gaussian model in fitted

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel()

gp = GaussianProcessRegressor(
    kernel=kernel,
    normalize_y=True,
    random_state=42
)

gp.fit(X, Y.ravel())

y_pred, sigma = gp.predict(X, return_std=True)

# Comapres prediction ato actual values

import matplotlib.pyplot as plt

plt.figure(figsize=(6,6))

plt.scatter(Y, y_pred)
plt.plot([Y.min(), Y.max()],
         [Y.min(), Y.max()],
         'r--')

plt.xlabel("Actual Output")
plt.ylabel("Predicted Output")

plt.show()

#The GP fits well, the points lie close to the red dashed line.

# Inpext the learned Kernal
print(gp.kernel_)

candidate = np.random.rand(10000, X.shape[1])

mean, std = gp.predict(candidate, return_std=True)

best_point = candidate[np.argmax(mean)]

kappa = 2.0
ucb = mean + kappa * std

best_point = candidate[np.argmax(ucb)]


#FUNCTION 7- No clear trend with indiviudal variables. That is more likely because of interaction between two variables
#0.33392848, 0.26935535, 0.49023239, 0.73847362, 0.04950843
Function_7_input = np.load('function_7/initial_inputs.npy')
Function_7_output = np.load('function_7/initial_outputs.npy')



X=Function_7_input
Y=Function_7_output

#Plot each input against the output

fig, ax = plt.subplots(1, 6, figsize=(20,4))

for i in range(6):
    ax[i].scatter(X[:, i], Y)
    ax[i].set_xlabel(f'Input {i+1}')
    ax[i].set_ylabel('Output')

plt.tight_layout()
plt.show()

#Input 1 Not a clear upward or downward trend
# Input 2 May be more tendency of having higher ouput at low input 2
# Input 3 some higher ouputs at both low and higher input 
# Input 4 slight increasing trend
# Input 5 negative trend

# with increase in colinearity it is getting mor edifficult to interpret the interaction
# Hence Gaussian model in fitted

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel()

gp = GaussianProcessRegressor(
    kernel=kernel,
    normalize_y=True,
    random_state=42
)

gp.fit(X, Y.ravel())

y_pred, sigma = gp.predict(X, return_std=True)

# Comapres prediction ato actual values

import matplotlib.pyplot as plt

plt.figure(figsize=(6,6))

plt.scatter(Y, y_pred)
plt.plot([Y.min(), Y.max()],
         [Y.min(), Y.max()],
         'r--')

plt.xlabel("Actual Output")
plt.ylabel("Predicted Output")

plt.show()

#The GP fits well, the points lie close to the red dashed line.

# Inpext the learned Kernal
print(gp.kernel_)

candidate = np.random.rand(10000, X.shape[1])

mean, std = gp.predict(candidate, return_std=True)

best_point = candidate[np.argmax(mean)]

kappa = 2.0
ucb = mean + kappa * std

best_point = candidate[np.argmax(ucb)]




#FUNCTION 8- No clear trend with indiviudal variables. That is more likely because of interaction between two variables
#0.33392848, 0.26935535, 0.49023239, 0.73847362, 0.04950843



Function_8_input = np.load('C:/Data Science and AI/Capstone Project/Initial_data_points_starter/initial_data/function_8/initial_inputs.npy')
Function_8_output = np.load('C:/Data Science and AI/Capstone Project/Initial_data_points_starter/initial_data/function_8/initial_outputs.npy')

Function_8_input_week2 = np.vstack([Function_8_input, functions_X["Function_8_X_week2"]])
Function_8_output_week2 = np.append(Function_8_output, functions_Y["Function_8_Y_week2"])


X=Function_8_input_week2
Y=Function_8_output_week2

#Plot each input against the output

fig, ax = plt.subplots(1, 8, figsize=(20,4))

for i in range(8):
    ax[i].scatter(X[:, i], Y)
    ax[i].set_xlabel(f'Input {i+1}')
    ax[i].set_ylabel('Output')

plt.tight_layout()
plt.show()

#Input 1 Not a clear upward or downward trend
# Input 2 May be more tendency of having higher ouput at low input 2
# Input 3 some higher ouputs at both low and higher input 
# Input 4 slight increasing trend
# Input 5 negative trend

# with increase in colinearity it is getting mor edifficult to interpret the interaction
# Hence Gaussian model in fitted

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel()

gp = GaussianProcessRegressor(
    kernel=kernel,
    normalize_y=True,
    random_state=42
)

gp.fit(X, Y.ravel())

y_pred, sigma = gp.predict(X, return_std=True)

# Comapres prediction ato actual values

import matplotlib.pyplot as plt

plt.figure(figsize=(6,6))

plt.scatter(Y, y_pred)
plt.plot([Y.min(), Y.max()],
         [Y.min(), Y.max()],
         'r--')

plt.xlabel("Actual Output")
plt.ylabel("Predicted Output")

plt.show()

#The GP fits well, the points lie close to the red dashed line.

# Inpext the learned Kernal
print(gp.kernel_)

candidate = np.random.rand(10000, X.shape[1])

mean, std = gp.predict(candidate, return_std=True)

best_point = candidate[np.argmax(mean)]

kappa = 2.0
ucb = mean + kappa * std

best_point = candidate[np.argmax(ucb)]


