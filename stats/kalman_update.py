import matplotlib.pyplot as plt
import numpy as np
from filterpy.kalman import MerweScaledSigmaPoints
from filterpy.kalman import JulierSigmaPoints
from scipy.interpolate import interp1d

class KalmanUpdate(object):

    def __init__(self, prior_m, prior_P, X, Y, m_weights, c_weights, obs_indexes):

        # Prior mean
        self.m = prior_m
        # Prior covariance
        self.P = prior_P
        # Array of sigma points
        self.X = X
        # Sigma points passed through forward model
        self.Y = Y
        # Mean weights
        self.m_weights = m_weights
        # Covariance weights
        self.c_weights = c_weights

        # Dimensions of the sigma point
        self.N1 = self.X.shape[1]
        # Dimensions of the output vector
        self.N2 = self.Y.shape[1]
        # Dimension of joint distribution 
        self.N = self.N1 + self.N2
        # Mean of full joint distribution
        self.m_full = np.zeros(self.N)


    # Do a Kalman update step incorporating an measurement y, with
    # covariance R
    def update(self, y, R):

        ### Unscented transform computations
        ######################################################################
        
        # Compute predicted mean
        mu = np.dot(self.m_weights, self.Y)


        # Compute predicted measurement covariance
        S = np.zeros((self.Y.shape[1], self.Y.shape[1]))
        for i in range(len(self.c_weights)):
            print(i)
            S += self.c_weights[i]*np.outer(self.Y[i] - mu, self.Y[i] - mu)
        S += R

 
        # Compute predicted measurement covariance
        C = np.zeros((self.X.shape[1], self.Y.shape[1]))
        for i in range(len(self.c_weights)):
            print(i)
            C += self.c_weights[i]*np.outer(self.X[i] - self.m, self.Y[i] - mu)


        ### Build the mean and covariance of the full joint distribution
        ######################################################################

        # Mean 
        self.m_full[0:self.N1] = self.m
        self.m_full[self.N1:] = mu

        # Covariance
        self.P_full = np.block([[self.P, C], [C.T, S]])


        ### Redefine the submatrices
        ######################################################################

        # Number of variables to condition on
        N_cond = len(y)
        N_opt = self.N - N_cond

        print(self.P_full.shape)
        print(N_cond)
        print(N_opt)

        P = self.P_full[0:N_opt, 0:N_opt]
        S = self.P_full[N_opt:, N_opt:]
        C = self.P_full[N_opt:,0:N_opt]

        print(P)
        print()
        print(S)
        print()
        print(C)
        print()
        print(P.shape, S.shape, C.shape)
        quit()
        

        ### Compute Kalman gain, revised mean, and covariance
        ######################################################################

        K = np.dot(C, np.linalg.inv(S))
        m_p = self.m + np.dot(K, y - mu)
        P_p = self.P - np.dot(np.dot(K, S), K.T)

        return m_p, P_p, mu, K

