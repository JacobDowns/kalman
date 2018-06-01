from model.inputs.paleo_inputs import *
from model.forward_model.forward_ice_model import *
import matplotlib.pyplot as plt
import numpy as np
from sigma_points_scalar import *

model_inputs = PaleoInputs('paleo_inputs/inverse_start.hdf5', dt = 0.33)
model = ForwardIceModel(model_inputs, "out", "paleo")


mwer_sigma = SigmaPointsScalar(alpha = 0.1, beta = 2., kappa = 2.)

# UKF v. particle filter mean?

mu = -5.158476
sigma2 = 2.5**2

delta_temps = mwer_sigma.sigma_points(mu, sigma2)

Ls = []

for delta_temp in delta_temps:
    Ls.append(model.step(delta_temp, accept = False) - model_inputs.L_init)



print Ls

mu = np.dot(mwer_sigma.mean_weights, Ls)
print mu
#plt.hist(Ls, bins = 30)
#plt.plot(delta_temps,Ls, 'ko-')
plt.show()
