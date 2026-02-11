import numpy as np

class LinearRegression:

    def __init__(self):
        self.b = None   # coefficients
        self.d = None   # number of features
        self.n = None   # sample size

    def fit(self, X, Y):

        # Add column of 1s for intercept (beta0)
        ones = np.ones((X.shape[0], 1))
        X_design = np.hstack((ones, X))

        # Store n and d
        self.n = X_design.shape[0]
        self.d = X_design.shape[1] - 1

        # Least squares solution
        XtX = X_design.T @ X_design
        XtX_inv = np.linalg.inv(XtX)
        XtY = X_design.T @ Y
        self.b = XtX_inv @ XtY

        return self.b

    def predict(self, X):
  
        # Predict using learned coefficients
        ones = np.ones((X.shape[0], 1))
        X_design = np.hstack((ones, X))
        return X_design @ self.b

    def sample_variance(self, X, Y):

        # Unbiased estimator of variance
        Y_pred = self.predict(X)
        residuals = Y - Y_pred
        SSE = np.sum(residuals**2)

        variance = SSE / (self.n - self.d - 1)
        return variance

    def standard_deviation(self, X, Y):

        variance = self.sample_variance(X, Y)
        return np.sqrt(variance)

    def rmse(self, X, Y):

        Y_pred = self.predict(X)
        mse = np.mean((Y - Y_pred)**2)
        return np.sqrt(mse)