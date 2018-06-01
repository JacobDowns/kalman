from model.inputs.paleo_inputs import *
from model.forward_model.forward_ice_model import *
import matplotlib.pyplot as plt
import numpy as np
from scalar_ukf import ScalarUKF

### Setup model
################################################################################

dt = 1./3.
# Start from 11.3 thousand years ago
#model_inputs = PaleoInputs('paleo_inputs/start_11_3.hdf5', dt = dt)
model_inputs = PaleoInputs('paleo_inputs/next_problem_point.hdf5', dt = dt)
model = ForwardIceModel(model_inputs, "out", "paleo")

# Initial mean and variance for state variable delta_temp
delta_temp_mu = -18.9779511229
delta_temp_sigma2 = 4.02843921951
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


### Do the filtering with some made up observations
################################################################################

# Retreat rate in m/a
Ls = []
delta_temps = []
ts = []


for i in range(4*3 + 2):
    # Observation mean and covariance at the current time
    L_mu = model_inputs.L_init - 25.*(model.t + dt)
    L_sigma2 = 750e3 #min(750e3, model.t*25000.)
    # Process noise (assumed noise in delta T)
    Q = 0.001

    delta_temp, delta_temp_sigma2 = ukf.step(L_mu, L_sigma2, Q)
    delta_temps.append(delta_temp)

    print "L_sigma2", L_sigma2
    print "opt delta temp", delta_temp, delta_temp_sigma2
    L = model.step(delta_temp, accept = True)
    Ls.append(L)
    ts.append(-11147.33 + model.t)
    print "dif", L - L_mu

#dolfin.plot(model.S)
#dolfin.plot(model.B)
#plt.show()

#print model.H0_c.vector().array()

#model.write_steady_file('paleo_inputs/next_problem_point')
