from model.inputs.paleo_inputs import *
from model.forward_model.forward_ice_model import *
import matplotlib.pyplot as plt
import numpy as np
from scalar_ukf import ScalarUKF

### Setup model
################################################################################

dt = 3.
model_inputs = PaleoInputs('paleo_inputs/is_paleo_11_6.h5', dt = dt)
model = ForwardIceModel(model_inputs, "out", "paleo")

# Initial mean and variance for state variable delta_temp
delta_temp_mu = -8.
delta_temp_sigma2 = 4.

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


### Do the filtering with some made up observations
################################################################################

# Retreat rate in m/a
Ls = []
delta_temps = []

for i in range(10000):
    # Observation mean and covariance at the current time
    L_mu = model_inputs.L_init
    L_sigma2 = 100.**2
    # Process noise
    Q = 1.

    delta_temp, delta_temp_sigma2 = ukf.step(L_mu, L_sigma2, Q)
    delta_temps.append(delta_temp)

    print "L_sigma2", L_sigma2
    print "opt delta temp", delta_temp, delta_temp_sigma2
    L = model.step(delta_temp, accept = True)
    Ls.append(L)
    print "dif", L - L_mu

    #dolfin.plot(model.S)
    #dolfin.plot(model.B)
    #aplt.show()

model.write_steady_file('is_paleo_steady_11_6.hdf5')
"""
if i % 50*3 == 0:
    dolfin.plot(model.adot_prime_func)
    plt.show()"""

#plt.plot(delta_temps)
#plt.show()
