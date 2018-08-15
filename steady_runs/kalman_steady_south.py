from model.inputs.paleo_inputs1 import *
from model.forward_model.forward_ice_model import *
import matplotlib.pyplot as plt
import numpy as np
from stats.scalar_ukf import ScalarUKF

### Setup model
################################################################################

dt = 1.
model_inputs = PaleoInputs('paleo_inputs/south_ideal.h5', dt = dt)
model = ForwardIceModel(model_inputs, "out", "bc_test")
#model.sea_level.assign(Constant(-47.))


#print model.H0_c.vector().min()
#quit()

sea_level = Function(model_inputs.V_cg)
sea_level.interpolate(model.sea_level)
dolfin.plot(model.H0_c + model.B)
dolfin.plot(model.B)
dolfin.plot(sea_level)
plt.ylim([-250., 3000.])
plt.show()

# Initial mean and variance for state variable delta_temp
delta_temp_mu = -8.
delta_temp_sigma2 = 1.

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

for i in range(9000):
    # Observation mean and covariance at the current time
    L_mu = 424777
    L_sigma2 = 100.**2
    # Process noise
    Q = 0.1**2

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

model.write_steady_file('south_paleo_steady_11_6_new')
"""
if i % 50*3 == 0:
    dolfin.plot(model.adot_prime_func)
    plt.show()"""

#plt.plot(delta_temps)
#plt.show()
