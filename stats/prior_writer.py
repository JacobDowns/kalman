import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

class PriorWriter(object):

    def __init__(self, inputs):

        # Output directory
        out_dir = inputs['out_dir']
        # Times 
        sigma_ts = inputs['sigma_ts'] 
        # Prior mean
        x = inputs['x']
        # Length of state vector
        N = len(x)
        # Covariance 
        Px = inputs['Px']

        
        ### Plot samples from prior
        ##########################################################################

        samples = np.random.multivariate_normal(x, Px, 100)

        for i in range(samples.shape[0]):
            plt.plot(samples[i])
        plt.show()

        
        ### Compute a minimal set of sigma points
        ##############################################

        # Compute X matrix where each row is a sigma point
        w_0 = 0.5
        if 'w_0' in inputs:
            w_0 = inputs['w_0']
            
        alpha = np.sqrt((1. - w_0) / N)
        C = np.linalg.cholesky(np.diag(np.ones(N), 0) - (alpha**2)*np.ones((N, N)))
        W = np.diag(np.diag(np.linalg.multi_dot([w_0*(alpha**2)*np.linalg.inv(C), np.ones((N,N)), np.linalg.inv(C.T)])), 0)
        W_sqrt = np.linalg.cholesky(W)
        P_sqrt = np.linalg.cholesky(Px)
        X = np.zeros((N, N+1))
        X[:,0] = np.dot(-P_sqrt, (alpha / np.sqrt(w_0))*np.ones(N))
        X[:,1:] = np.linalg.multi_dot([P_sqrt, C, np.linalg.inv(W_sqrt)])
        X = X.T
        X += x

        # Array of weights
        weights = np.zeros(N+1)
        weights[0] = w_0
        weights[1:] = np.diag(W, 0)

        x_n = np.zeros(N)
        P_n = np.zeros((N,N))
        # Plot the sigma points and check the mean and covariance
        for i in range(N+1):
            w_i = weights[i]
            x_i = X[i]
            x_n += w_i*x_i
            P_n += w_i*np.outer(x_i - x, x_i - x)
            plt.plot(x_i)

        plt.show()


        ### Write the prior info to a file
        ####################################################
        np.savetxt(out_dir + 'sigma_ts.txt', sigma_ts)
        np.savetxt(out_dir + 'prior_m.txt', x)
        np.savetxt(out_dir + 'prior_P.txt', Px)
        np.savetxt(out_dir + 'm_weights.txt', weights)
        np.savetxt(out_dir + 'c_weights.txt', weights)
        np.savetxt(out_dir + 'X.txt', X)

        
