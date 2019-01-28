import numpy as np
from scipy.linalg import null_space
import matplotlib.pyplot as plt

class TestSigmaPoints(object):

    def __init__(self, x, Pxx):
         # Sample mean
         self.x = x
         # Sample covariance
         self.Pxx = Pxx
         # State dimension
         self.n = len(x)
         # Number of sigma points
         self.N = 2*self.n + 1

         
    def get_points(self, x_new, div = 3.):
        n = self.n
        N = self.N
        x = self.x
        Pxx = self.Pxx
        Pxx_sqrt = np.linalg.cholesky(Pxx)

        
        ### Build sigma points in N(0,I) space
        ##################################################

        # The mean in N(0,I) space
        z = np.linalg.solve(Pxx_sqrt, x_new - x)
        # Length of z
        z_norm = np.linalg.norm(z)
        # Columns form an orthonormal basis for the null space of z
        z_null = null_space(np.tile(z, n).reshape(n, n)).T
        l = z_norm / 4.
        
        # Sigma points
        X = np.zeros((N, n))
        X[0,:]         = z
        X[n, :]        = -(z / z_norm) * l
        X[1:n,:]       = z_null * l
        X[(n+1):-1, :] = -z_null * l

        # Covariance weights
        w_c = np.zeros(N)
        w_c[:] = 1. / (2.*l**2)
        w_c[0] = 1. / (2.*z_norm**2)
        w_c[-1] = 1.

        # Mean weights
        w_m = np.ones(N)
        w_m[n] = z_norm / l
        w_m /= 2.*w_m[0:-1].sum()
        #w_m /= w_m.sum()
        w_m[-1] = 0.5

        
        ### Transform the sigma points into N(x,P) space
        ##################################################

        X_new = (Pxx_sqrt @ X.T).T + x

        
        """
        # Check sigma point statistics

        xf = np.zeros(n)
        Cf = np.zeros((n,n))

        # Mean 
        for i in range(N):
            xf += w_m[i]*X_new[i,:]

        # Covariance
        for i in range(N):
            Cf += w_c[i]*np.outer(X_new[i,:] - xf, X_new[i,:] - xf)"""

        return X_new, w_m, w_c
