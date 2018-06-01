from model.inputs.paleo_inputs import *
from model.forward_model.forward_ice_model import *
import matplotlib.pyplot as plt
import numpy as np
from scalar_ukf import ScalarUKF

model_inputs = PaleoInputs('paleo_inputs/inverse_start.hdf5', dt =  1./3.)
model = ForwardIceModel(model_inputs, "out", "paleo")

# Initial mean and variance for state variable delta_temp
delta_temp_mu = -4.
delta_temp_sigma2 = 1.**2
# Retreat rate in m/a
retreat_rate = 25.

# Process model function
def F(xs):
    return xs

# Measurement model function
def H(xs):
    print "H", xs
    ys = np.zeros_like(xs)

    for i in range(len(xs)):
        ys[i] = model.step(xs[i], accept = False)

    return ys

ukf = ScalarUKF(delta_temp_mu, delta_temp_sigma2, F, H)


Ls = []
delta_temps = []

for i in range(255*4):
    # Observation mean and covariance at the current time
    L_mu = model_inputs.L_init - 25.*model.t
    L_sigma2 = 50.
    # Process noise
    Q = 0.05

    delta_temp, delta_temp_sigma2 = ukf.step(L_mu, L_sigma2, Q)
    delta_temps.append(delta_temp)

    print "opt delta temp", delta_temp
    L = model.step(delta_temp, accept = True)
    Ls.append(L)
    print "dif", L - L_mu





plt.plot(delta_temps)
plt.show()
