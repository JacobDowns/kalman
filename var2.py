from model.inputs.paleo_inputs import *
from model.forward_model.forward_ice_model import *
import matplotlib.pyplot as plt
import numpy as np
from scalar_ukf import ScalarUKF


### Setup model
################################################################################

dt = 1./4.
model_inputs = PaleoInputs('paleo_inputs/start_11_3.hdf5', dt = dt)
model = ForwardIceModel(model_inputs, "out", "paleo")

# Initial mean and variance for state variable delta_temp
delta_temp_mu = -5.0092650344
delta_temp_sigma2 = 0.05
# Retreat rate in m/a
retreat_rate = 25.
#

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

for i in range(8000):
    print model.t
    # Observation mean and covariance at the current time
    L_mu = model_inputs.L_init - 25.*(model.t + dt)
    L_sigma2 = min(900e3, model.t*25000.)
    # Process noise
    Q = 0.001

    delta_temp, delta_temp_sigma2 = ukf.step(L_mu, L_sigma2, Q)
    delta_temps.append(delta_temp)

    print "L_sigma2", L_sigma2
    print "opt delta temp", delta_temp, delta_temp_sigma2
    L = model.step(delta_temp, accept = True)
    Ls.append(L)
    ts.append(-11147.33 + model.t)
    print "dif", L - L_mu


np.savetxt('filter/var3_t.txt', np.array(ts))
np.savetxt('filter/var3_T.txt', np.array(delta_temps))
np.savetxt('filter/var3_L.txt', np.array(Ls))
