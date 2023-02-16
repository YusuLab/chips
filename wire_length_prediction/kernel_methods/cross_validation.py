import numpy as np
import math
import random
import csv

# Methods
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
import sklearn.gaussian_process as gp

# Metrics
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

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

# Standard way of creating indices for cross-validation
'''
num_folds = 10
fold_indices = [[] for fold in range(num_folds)]
for idx in range(num_samples):
    fold = random.randint(0, num_folds - 1)
    fold_indices[fold].append(idx)
'''

# Read the netlist information
netlist_info = []
with open('df_clean.csv', newline = '') as csv_file:
    reader = csv.reader(csv_file, delimiter = ',', quotechar = '|')
    for row in reader:
        netlist_info.append(row[2])
netlist_info = netlist_info[1:]
assert len(netlist_info) == num_samples

netlist_names = []
for idx in range(num_samples):
    parts = netlist_info[idx].split('_')
    name = '_'.join(parts[0:5])
    netlist_info[idx] = name
    netlist_names.append(name)

netlist_names = list(set(netlist_names))
netlist_names.sort()
num_netlists = len(netlist_names)

print('Netlists:', netlist_names)
print('Number of netlists:', num_netlists)

# Create indices for cross-validation
num_folds = 4
fold_netlists = [[] for fold in range(num_folds)]
for idx in range(num_netlists):
    fold = random.randint(0, num_folds - 1)
    fold_netlists[fold].append(netlist_names[idx])

fold_indices = [[] for fold in range(num_folds)]
for fold in range(num_folds):
    for name in fold_netlists[fold]:
        fold_indices[fold] += [idx for idx in range(num_samples) if netlist_info[idx] == name]
    fold_indices[fold].sort()

count_netlists = 0
count_samples = 0
for fold in range(num_folds):
    print('Fold', fold, ':')
    print('- Number of netlists:', len(fold_netlists[fold]))
    print('- Number of samples:', len(fold_indices[fold]))
    count_netlists += len(fold_netlists[fold])
    count_samples += len(fold_indices[fold])

assert count_netlists == num_netlists
assert count_samples == num_samples

# Methods we want to try
method_names = [
    'LinearRegression', # Linear Regression
    'Ridge', # Ridge Regression
    'Linear-SVM', # Linear Support Vector Machine (i.e. linear kernel)
    'RBF-SVM', # Kernelized SVM with Radial Basis Function kernel
    'Gaussian-Processes' # Gaussian Processes
]
num_methods = len(method_names)

# Results for Mean Average Error (MAE)
mae_results = [[] for idx in range(num_methods)]

# Results for Root Mean Square Error (RMSE)
rmse_results = [[] for idx in range(num_methods)]

# Cross-validation
for fold in range(num_folds):
    print('Fold', fold, '----------------------------------')

    # For each fold, create the corresponding test set and train set
    test_indices = fold_indices[fold]
    train_indices = [idx for idx in range(num_samples) if idx not in test_indices]
    assert len(train_indices) + len(test_indices) == num_samples

    X_train = X[train_indices, :]
    y_train = y[train_indices]
    X_test = X[test_indices, :]
    y_test = y[test_indices]

    # Train and test for each method with this fold
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
            kernel = gp.kernels.ConstantKernel(1.0, (1e-1, 1e3)) * gp.kernels.RBF(10.0, (1e-3, 1e3))
            model = gp.GaussianProcessRegressor(kernel = kernel, alpha = 0.1, normalize_y = True)
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

        # Save the result
        mae_results[idx].append(mae)
        rmse_results[idx].append(rmse)

        print('Done', method_name)

# Summary
print('Summary ----------------------------------')
for idx in range(num_methods):
    method_name = method_names[idx]
    print('\n', method_name, ':')

    array = np.array(mae_results[idx]) # Remember to scale back to the original scale
    mae_mean = np.mean(array)
    mae_std = np.std(array)
    print('* MAE (normalized targets) =', mae_mean, '+/-', mae_std)

    array = np.array(mae_results[idx]) * y_std # Original scale
    mae_mean = np.mean(array)
    mae_std = np.std(array)
    print('* MAE =', mae_mean, '+/-', mae_std)

    array = np.array(rmse_results[idx]) # Remember to scale back to the original scale
    rmse_mean = np.mean(array)
    rmse_std = np.std(array)
    print('* RMSE (normalized targets) =', rmse_mean, '+/-', rmse_std)

    array = np.array(rmse_results[idx]) * y_std # Original scale
    rmse_mean = np.mean(array)
    rmse_std = np.std(array)
    print('* RMSE =', rmse_mean, '+/-', rmse_std)

print('Done')
