import numpy as np
import scipy.stats as stats

class LinearRegression:

    def __init__(self, X, y):
        self.X = np.array(X) # Features
        self.y = np.array(y) # Target
        self.d = self.X.shape[1]# Number of features
        self.n = self.X.shape[0] # Number of samples


    def least_squares_mean(self):
        return np.mean(self.y)
    
    def standard_deviation(self):
        return np.std(self.y, ddof=1)
    
if __name__ == "__main__":
    pass