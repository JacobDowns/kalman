from model.inputs.paleo_inputs import *
from model.forward_model.forward_ice_model import *
import matplotlib.pyplot as plt
import numpy as np
from sigma_points_scalar import *

"""
Particle filter.
"""

class ParticleFilter(object):

    def __init__(self):

        ### Setup the model
        inputs = PaleoInputs('paleo_inputs/inverse_start.hdf5', dt = 0.33)
        model = ForwardIceModel(inputs, "out", "paleo")

        # Store the model
        self.model = model
        # Store model inputs
        self.inputs = inputs
        # SMB parameter mean
        self.adot0_mu = -5.158476
        # SMB parameter variance
        self.adot0_sigma2 = 2.5**2
        # Initial time
        self.t = 0.0
        # Time step
        self.dt = 0.33
        # Initial margin position
        self.L_init = inputs.L_init
        # Expected approximate rate of retreat
        self.retreat_rate = -25.
        # Number of particles
        self.num_particles = 300
        # Particle values
        self.particle_vals = np.linspace(self.adot0_mu - 3., self.adot0_mu + 3., self.num_particles)
        # Particle weights
        self.particle_weights = np.ones(self.num_particles)
        # Process model variance
        self.process_sigma2 = 0.001


    # Process model
    def F(self, x):
        return x


    # Measurement Model
    def H(self, x):
        L = np.zeros(len(x))
        for i in range(len(x)):
            L[i] = self.model.step(x[i], accept = False)

        return L


    def predict(self):
        """
        The prediction step just adds noise to the particles.
        """

        self.particle_vals += np.sqrt(self.process_sigma2)*np.random.randn(self.num_particles)


    # Update
    def step(self):

        ### Precict to get the prior
        ########################################################################

        # Run the process model (blurs adot distribution)
        self.predict()
        # Get the prior distribution of Ls
        Ls = self.H(self.particle_vals)


        ### Now update to get the posterior
        ########################################################################

        L_obs_mu, L_obs_sigma2 = self.get_obs(self.t)
        # Distance between particles and the observed margin position
        distances = np.sqrt((Ls - L_obs_mu)**2)

        weights *= scipy.stats.norm(distances, L_obs_sigma2).pdf(z[i])

        weights += 1.e-300      # avoid round-off to zero
        weights /= sum(weights) # normalize




    # Return a random variable representing the observation at the given time
    def get_obs(self, t):
        ### Compute the mean and variance of the observation at the current time
        mu = self.L_init + self.retreat_rate*t
        sigma2 = 1.0**2

        return (mu, sigma**2)



    def sigma_points(self, mean, variance):
        points = np.array([mean - np.sqrt((1. + self.lam)*variance), mean, mean + np.sqrt((1. + self.lam)*variance)])
        return points

pf = ParticleFilter()
pf.step()
