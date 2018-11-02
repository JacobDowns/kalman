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
        m_full = np.zeros(self.N)
        m_full[0:self.N1] = self.m
        m_full[self.N1:] = mu

        # Covariance
        P_full = np.block([[self.P, C], [C.T, S]])


        ### Repartition the mean and covariance in order to compute
        ### the correct conditional probability 
        ######################################################################

        # Number of conditional variables
        N_cond = len(y)
        # Number of random variables
        N_rand = self.N - N_cond

        # Repartition
        m = m_full[0:N_rand]
        mu = m_full[N_rand:]
        P = P_full[0:N_rand, 0:N_rand]
        S = P_full[N_rand:, N_rand:]
        C = P_full[0:N_rand,N_rand:]
        

        ### Compute Kalman gain, revised mean, and covariance
        ######################################################################

        K = np.dot(C, np.linalg.inv(S))
        m_p = m + np.dot(K, y - mu)
        P_p = P - np.dot(np.dot(K, S), K.T)
        v = P_p[range(len(P_p)), range(len(P_p))]

        print(m_p[-5:])
        plt.plot(m_p[0:-5])
        plt.plot(m_p[0:-5] + 2.0*np.sqrt(v)[0:-5])
        plt.plot(m_p[0:-5] - 2.0*np.sqrt(v)[0:-5])
        plt.show()
        quit()
        return m_p, P_p, mu, K

