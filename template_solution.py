# This serves as a template which will guide you through the implementation of this task. It is advised
# to first read the whole template and get a sense of the overall structure of the code before trying to fill in any of the TODO gaps.
# First, we import necessary libraries:
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold

# Add any additional imports here (however, the task is solvable without using 
# any additional imports)
# import ...

def fit(X, y, lam): #|samples| x |features|, labels, hyperparameter

    num_features = X.shape[1] # 'd' in the script
    I = np.eye(num_features)
    weights = np.linalg.inv(X.T @ X + lam * I) @ X.T @ y # w^hat = (X^T X + lambda * I_d)^(-1) X^T y
    assert weights.shape == (13,)
    return weights 


def calculate_RMSE(w, X, y):

    y_pred = X @ w # Predicted labels based based on the coefficients in w.
    rmse = np.sqrt(np.mean((y - y_pred)**2)) 
    assert np.isscalar(rmse)
    return rmse


def average_LR_RMSE(X, y, lambdas, n_folds):

    RMSE_mat = np.zeros((n_folds, len(lambdas)))
    kf = KFold(n_splits=n_folds, shuffle=True, random_state = 42)

    for i, (train_idx, test_idx) in enumerate (kf.split(X)): # Loop over the 15 folds, here: example when first fold is the test set.

        X_train, y_train = X[train_idx], y[train_idx] # Take the corresponding rows
        X_test, y_test = X[test_idx], y[test_idx] # Take the corresponding rows

        for j, lam in enumerate(lambdas): # For each lambda train ridge regression on the folds 1-9, then evaluate w^hat on fold 1. 
            w = fit(X_train, y_train, lam) #w^hat
            RMSE_mat[i, j] = calculate_RMSE(w, X_test, y_test) 


    avg_RMSE = np.mean(RMSE_mat, axis=0)
    assert avg_RMSE.shape == (5,)
    return avg_RMSE


# Main function. You don't have to change this
if __name__ == "__main__":
    # Data loading
    data = pd.read_csv("train.csv")
    y = data["y"].to_numpy()
    data = data.drop(columns="y")
    # print a few data samples
    print(data.head())

    X = data.to_numpy()
    # The function calculating the average RMSE
    lambdas = [0.1, 1, 10, 100, 200]
    n_folds = 10
    avg_RMSE = average_LR_RMSE(X, y, lambdas, n_folds)
    # Save results in the required format
    np.savetxt("./results.csv", avg_RMSE, fmt="%.12f")

"""
def average_LR_RMSE(X, y, lambdas, n_folds):
    
    num_samples = X.shape[0]
    fold_size = num_samples // n_folds 
    RMSE_mat = np.zeros((n_folds, len(lambdas)))
    kf = KFold(n_splits=n_folds, shuffle=True, random_state = 42)

    for i in range(n_folds): # Loop over the 15 folds, here: example when first fold is the test set.
        start = i * fold_size
        end = (i + 1) * fold_size

        test_idx = np.arange(start, end)  # [0, 1, ..., 14], test idx rows
        train_idx = np.setdiff1d(np.arange(num_samples), test_idx) #[15, ..., 149], train idx rows

        X_train, y_train = X[train_idx], y[train_idx] # Take the corresponding rows
        X_test, y_test = X[test_idx], y[test_idx] # Take the corresponding rows

        for j in range(len(lambdas)): # For each lambda train ridge regression on the folds 1-9, then evaluate w^hat on fold 1. 
            lam = lambdas[j]
            w = fit(X_train, y_train, lam) #w^hat
            RMSE_mat[i, j] = calculate_RMSE(w, X_test, y_test) 


    avg_RMSE = np.mean(RMSE_mat, axis=0)
    assert avg_RMSE.shape == (5,)
    return avg_RMSE
"""
