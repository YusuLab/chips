import numpy as np
import math
import random
import csv
import pickle

# Methods
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
import sklearn.gaussian_process as gp

# Metrics
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error

# Mean Absolute Relative Error
def mean_absolute_relative_error(y_truth, y_pred):
    return np.mean(np.abs(y_truth - y_pred) / np.abs(y_truth))

# Fix random seed for reproducibility
random.seed(123456789)

# Load data
X = np.load('X_full.npy')
y = np.load('Y.npy')

# Statistics
num_samples = y.shape[0]
print('X shape:', X.shape)
print('y shape:', y.shape)
print('y statistics: min =', np.min(y), ', max =', np.max(y), ', mean =', np.mean(y), ', std =', np.std(y)) 

# Targets normalization
y_mean = np.mean(y)
y_std = np.std(y)
y = (y - y_mean) / y_std

# Methods we want to try
method_names = [
    'LinearRegression', # Linear Regression
    'Ridge', # Ridge Regression
    'Linear-SVM', # Linear Support Vector Machine (i.e. linear kernel)
    'RBF-SVM', # Kernelized SVM with Radial Basis Function kernel
    'Gaussian-Processes' # Gaussian Processes
]
num_methods = len(method_names)

# Train/Test split
train_indices = np.load('old_train_idx.npy')
test_indices = np.load('old_test_idx.npy')

X_train = X[train_indices, :]
y_train = y[train_indices]
X_test = X[test_indices, :]
y_test = y[test_indices]

# Train and test for each method
for idx in range(num_methods):
    method_name = method_names[idx]
        
    # Create the model
    if method_name == 'LinearRegression':
        model = LinearRegression()
    elif method_name == 'Ridge':
        # You will need to search for the optimal hyper-parameter
        model = Ridge(alpha = 10.0)
    elif method_name == 'Linear-SVM':
        # You will need to search for the optimal hyper-parameters
        model = SVR(kernel = 'linear', C = 10.0, epsilon = 2)
    elif method_name == 'RBF-SVM':
        # You will need to search for the optimal hyper-parameters
        model = SVR(kernel = 'rbf', C = 10.0)
    elif method_name == 'Gaussian-Processes':
        # You will need to search for the optimal hyper-parameter
        kernel = gp.kernels.ConstantKernel(1.0, (1e-1, 1e3)) * gp.kernels.RBF(1.0, (1e-3, 1e3))
        model = gp.GaussianProcessRegressor(kernel = kernel, alpha = 0.01, normalize_y = True)
    else:
        print('Unsupported method!')
        assert False

    # Fit the model
    model.fit(X_train, y_train)

    # Make prediction
    y_hat = model.predict(X_test)

    # Evaluate
    mae = mean_absolute_error(y_test, y_hat)
    mse = mean_squared_error(y_test, y_hat)
    rmse = math.sqrt(mse)
    mape = mean_absolute_percentage_error(y_test * y_std + y_mean, y_hat * y_std + y_mean)
    mare = mean_absolute_relative_error(y_test * y_std + y_mean, y_hat * y_std + y_mean)

    print('\n', method_name, ':')
    print('* MAE (normalized targets) =', mae)
    print('* MAE =', mae * y_std)
    print('* RMSE (normalized targets) =', rmse)
    print('* RMSE =', rmse * y_std)
    print('* MAPE =', mape)
    print('* MARE =', mare)

print('Done')
