import numpy as np
import matplotlib.pyplot as plt
from sigma_points_scalar import *

"""
Simple scalar unscented Kalman filter
"""

class UKF(object):

    def __init__(self, F, H):

        # Process model
        self.F = F
        # Measurement model
        self.H = H
        # SMB parameter mean
        self.m_k = 1.
        # SMB parameter variance
        self.P_k = 1.
        # Time step
        self.dt = 1.
        # Object for calculating sigma points
        self.mwer_sigma = SigmaPointsScalar(alpha = 0.1, beta = 2., kappa = 2.)
        # Number of steps
        self.i = 0


    # Predict the state
    def predict(self, Q):
        # Run sigma points through process model
        x = self.mwer_sigma.sigma_points(self.m_k, self.P_k)
        # Y = F(x)
        X_hat_k = self.F(x)
        # Prior mean
        m_minus_k = np.dot(self.mwer_sigma.mean_weights, X_hat_k)
        # Prior covariance
        P_minus_k = np.dot(self.mwer_sigma.variance_weights, (X_hat_k - m_minus_k)**2) + Q

        return m_minus_k, P_minus_k


    # Update the state given the measurement
    def step(self, y_k, R, Q):

        ### Compute state mean and covariance
        ########################################################################

        # Run process model to get the prior
        m_minus_k, P_minus_k = self.predict(Q)
        # Form sigma points
        x = self.mwer_sigma.sigma_points(m_minus_k, P_minus_k)
        # Run the mesurement model
        Y_hat_k = self.H(x)


        ### Compute predicted mean and covariance of measurement
        ########################################################################

        # Predicted observation mean
        mu_k = np.dot(self.mwer_sigma.mean_weights, Y_hat_k)
        # Predicted observation variance (math works with R added here, but why?)
        S_k = np.dot(self.mwer_sigma.variance_weights, (Y_hat_k - mu_k)**2) + R
        # Cross covariance
        C_k = np.dot(self.mwer_sigma.variance_weights, (x - m_minus_k)*(Y_hat_k - mu_k))


        ### Compute filter gain, filtered state mean, and covariance
        ########################################################################

        # Kalman gain
        K_k = C_k * (1.0 / S_k)
        # State mean
        m_k = m_minus_k + K_k * (y_k - mu_k)
        # State variance
        P_k = P_minus_k - K_k*S_k*K_k

        self.m_k = m_k
        self.P_k = P_k

        return m_k, P_k


ukf = UKF(lambda x : x, lambda y : y)

print ukf.step(2.0, 1.0, 0.1)
