from model.inputs import *
import numpy as np
from model.adot_inputs_elevation_dependent import *
from model.forward_model.forward_ice_model import *
import matplotlib.pyplot as plt
from sigma_points_scalar import *

"""
Simple unscented Kalman filter
"""

class UKF(object):

    def __init__(self):

        # SMB parameter mean
        self.x_mu = 1.
        # SMB parameter variance
        self.x_sigma2 = 1.
        # Initial time
        self.t = 0.0
        # Time step
        self.dt = 1.
        # Object for calculating sigma points
        self.mwer_sigma = SigmaPointsScalar(alpha = 0.1, beta = 2., kappa = 2.)
        # Process model variance
        self.Q = 0.1

        self.Y = None

    # Process model
    def F(self, x):
        return x

    # Measurement Model
    def H(self, x):
        return x


    ### Add a little noise to the smb
    def predict(self):
        """
        Run the process model to get the prior. (Just adds noise.)
        """

        # Run sigma points through process model
        sigma_points = self.mwer_sigma.sigma_points(self.x_mu, self.x_sigma2)
        # Y = F(x)
        Y = self.F(sigma_points)
        # Prior mean
        x_bar = np.dot(self.mwer_sigma.mean_weights, Y)
        # Prior covariance
        P_bar = np.dot(self.mwer_sigma.variance_weights, (Y - x_bar)**2) + self.Q

        return x_bar, P_bar, Y


    ### Update
    def step(self):
        self.t += 1.

        ### Compute state mean and covariance
        ########################################################################
        # Run process model to get the prior
        x_bar, P_bar, Y = self.predict()


        print "Prior"
        print x_bar, P_bar

        # Run the mesurement model
        L = self.H(Y)
        # Observation mean
        mu_z = np.dot(self.mwer_sigma.mean_weights, L)

        print "mu_z", mu_z
        quit()
        # Get the observation mean and variance
        z = 2.0
        R = 2.0
        # Residual
        y = z - mu_z
        # Measurement variance
        P_z = np.dot(self.mwer_sigma.variance_weights, (L - mu_z)**2) + R

        print "Measurement Model"
        print mu_z, P_z

        print "Measurement Data"
        print z, R

        # Kalman gain
        K = np.dot(self.mwer_sigma.variance_weights, (Y - x_bar)*(L - mu_z)) * (1. / P_z)
        # State mean
        x = x_bar + K*y
        # State variance
        P = P_bar - K*P_z*K

        print "Posterior"
        print x, P



        ### Take a real step in the model using the "optimal" SMB param
        ########################################################################
        """
        self.adot_mu = x
        self.adot_sigma2 = P
        L = self.model.try_step(self.dt, self.adot_mu, accept = True)

        print x, P, z-L
        return x, L, z-L"""


    # Return a random variable representing the observation at the given time
    def get_obs(self, t):
        ### Compute the mean and variance of the observation at the current time
        mu = self.L_init + self.retreat_rate*t
        sigma = (1500.0*np.sin( 2.*np.pi*t / 2000.) + 250.0) / 2.
        return (mu, sigma**2)


    def sigma_points(self, mean, variance):
        points = np.array([mean - np.sqrt((1. + self.lam)*variance), mean, mean + np.sqrt((1. + self.lam)*variance)])
        return points

ukf = UKF()

ukf.step()
