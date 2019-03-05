import numpy as np
from scipy.linalg import null_space
import matplotlib.pyplot as plt

class SigmaPoints(object):

    def __init__(self, x, Pxx):
         # Sample mean
         self.x = x
         # Sample covariance
         self.Pxx = Pxx
         # State dimension
         self.n = len(x)
         # Matrix square root of Pxx
         self.Pxx_sqrt = np.linalg.cholesky(Pxx)


    # Return a minimal set of n+1 sigma points
    def get_minimal_set(self, w0 = 0.5):
        n = self.n
        x = self.x
        Pxx_sqrt = self.Pxx_sqrt

        alpha = np.sqrt((1. - w0) / n)
        C = np.linalg.cholesky(np.diag(np.ones(n), 0) - (alpha**2)*np.ones((n, n)))
        W = np.diag(np.diag(np.linalg.multi_dot([w0*(alpha**2)*np.linalg.inv(C), np.ones((n,n)), np.linalg.inv(C.T)])), 0)
        W_sqrt = np.linalg.cholesky(W)
        X = np.zeros((n, n+1))
        X[:,0] = np.dot(-Pxx_sqrt, (alpha / np.sqrt(w0))*np.ones(n))
        X[:,1:] = np.linalg.multi_dot([Pxx_sqrt, C, np.linalg.inv(W_sqrt)])
        X = X.T
        X += x

        # Array of weights
        w = np.zeros(n+1)
        w[0] = w0
        w[1:] = np.diag(W, 0)

        return X, w, w

        
    # Return a set of 2*n+1 sigma points where one point is located at the predicted
    # posterior mean 
    def get_posterior_set(self, x_new, div = 4.):
        n = self.n
        x = self.x
        Pxx_sqrt = self.Pxx_sqrt
        N = 2*n + 1
        
        ### Build sigma points in N(0,I) space
        ##################################################

        # The mean in N(0,I) space
        z = np.linalg.solve(Pxx_sqrt, x_new - x)
        # Length of z
        z_norm = np.linalg.norm(z)
        # Columns form an orthonormal basis for the null space of z
        z_null = null_space(np.tile(z, n).reshape(n, n)).T
        l = z_norm / div
        
        # Sigma points
        X = np.zeros((N, n))
        X[0,:]         = z
        X[n, :]        = -(z / z_norm) * l
        X[1:n,:]       = z_null * l
        X[(n+1):-1, :] = -z_null * l

        # Covariance weights
        wc = np.zeros(N)
        wc[:] = 1. / (2.*l**2)
        wc[0] = 1. / (2.*z_norm**2)
        wc[-1] = 1.

        # Mean weights
        wm = np.ones(N)
        wm[n] = z_norm / l
        wm /= 2.*w_m[0:-1].sum()
        wm[-1] = 0.5

        
        ### Transform the sigma points into N(x,P) space
        ##################################################

        X_new = (Pxx_sqrt @ X.T).T + x

        return X_new, wm, wc


    # Get sigma points and weights for a fifth order cubature filter
    def get_fifth_order_set(self, r = np.sqrt(3.)):

        # Dimension
        N = self.n
        x = self.x

        ### Generate Weights
        ########################################################

        # Coordinate for the first symmetric set
        r1 = (r*np.sqrt(N-4.))/np.sqrt(N - r**2 - 1.)
        # First symmetric set weight
        w2 = (4. - N) / (2. * r1**4)
        # Second symmetric set weight
        w3 = 1. / (4. * r**4)
        # Center point weight
        w1 = 1. - 2.*N*w2 - 2.*N*(N-1)*w3
        # Vector of weights
        w = np.block([w1, np.repeat(w2, 2*N), np.repeat(w3, 2*N*(N-1))])


        ### Generate Points
        ########################################################
        
        # First fully symmetric set
        X0 = r1*np.eye(N)
        X0_s = np.block([X0, -X0])
        
        # Second fully symmetric set
        X1 = r*np.eye(N)
        indexes_i = []
        indexes_j = []
        for i in range(1,N):
            indexes_i.append(np.repeat([i],i))
            indexes_j.append(np.arange(0,i))
        indexes_i = np.concatenate(indexes_i).ravel()
        indexes_j = np.concatenate(indexes_j).ravel()
        P1 = X1[indexes_i, :].T + X1[indexes_j, :].T
        P2 = X1[indexes_i, :].T - X1[indexes_j, :].T
        X1_s = np.block([P1, P2, -P1, -P2])

        # Full set of points (columns are points)
        X = np.block([np.zeros(N)[:,None], X0_s, X1_s])

        # Change variables
        X = x[:,None].repeat(2*N**2 + 1, axis = 1) + self.Pxx_sqrt@X

        return X.T, w, w



    # Just take random draws from the distribution (for testing)
    def __get_random_set__(self, num_draws):
        samples = np.random.multivariate_normal(self.x, self.Pxx, num_draws)
        w = np.ones(num_draws)
        return samples, w, w
        
        
