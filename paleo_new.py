from model.inputs.paleo_inputs import *
from model.forward_model.forward_ice_model import *
import matplotlib.pyplot as plt
import numpy as np
from scalar_ukf import ScalarUKF

dt = 1./3.
model_inputs = PaleoInputs('paleo_inputs/inverse_start.hdf5', dt =  dt)
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



### Setup plot
################################################################################

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})

plt.ion()
fig,ax = plt.subplots(nrows=2,sharex=True,figsize=(7,7))
xs = model_inputs.mesh.coordinates() * model_inputs.L_init
L_init = model_inputs.L_init

bed = project(model.B)
surface = project(model.S0_c)
adot = project(model.adot_prime_func)
surface_ref = project(model_inputs.input_functions['S_ref'])

ph_bed, = ax[0].plot(xs, bed.compute_vertex_values(), 'b', linewidth = 2.5)
ph_surface, = ax[0].plot(xs, surface.compute_vertex_values(), 'k', linewidth = 2.5)
ph_term, = ax[0].plot([model_inputs.L_init, model_inputs.L_init], [-500., 4000.], 'r--', linewidth = 2.5)
ph_term1, = ax[0].plot([model_inputs.L_init, model_inputs.L_init], [-500., 4000.], 'r--', linewidth = 2.5)
ph_term2, = ax[0].plot([model_inputs.L_init, model_inputs.L_init], [-500., 4000.], 'r--', linewidth = 2.5)
#ax[0].plot(xs, surface_ref.compute_vertex_values(), 'g', linewidth = 2.5)
ax[0].set_ylim(-400, 3500.)
L_start = 375e3
ax[0].set_xlim(L_start, L_init)
ax[0].set_ylabel('S (m)')

ph_adot, = ax[1].plot(xs, adot.compute_vertex_values(), 'k', linewidth = 2.5)
ax[1].plot(xs, 0.*xs, 'b', linewidth = 2.5)
ax[1].set_ylim(-2.5, 0.75)
ax[1].set_xlim(L_start, L_init)
ax[1].set_xlabel('x (m)')
ax[1].set_ylabel('smb (m/a)')

N = 4000
delta_temps = []
Ls = []


for i in range(N):
    # Observation mean and covariance at the current time
    L_mu = model_inputs.L_init - 25.*(model.t + dt)
    L_sigma2 = min(750000., 1.0 + model.t*5000.)
    # Process noise
    Q = 0.01

    delta_temp, delta_temp_sigma2 = ukf.step(L_mu, L_sigma2, Q)
    delta_temps.append(delta_temp)

    print "L_sigma2", L_sigma2
    print "opt delta temp", delta_temp, delta_temp_sigma2
    L = model.step(delta_temp, accept = True)
    Ls.append(L)
    print "dif", L - L_mu


    if model.i % 20 == 0:
        xs = model_inputs.mesh.coordinates() * float(model.L0)

        # Plot thickness
        ax[0].set_xlim(L_start, L_init)
        surface = project(model.S0_c)
        ph_surface.set_xdata(xs)
        ph_surface.set_ydata(surface.compute_vertex_values())

        # Plot smb
        ax[1].set_xlim(L_start, L_init)
        adot = project(model.adot_prime_func)
        ph_adot.set_xdata(xs)
        ph_adot.set_ydata(adot.compute_vertex_values())

        # Plot terminus line

        ph_term.set_xdata([L_mu, L_mu])
        ph_term.set_ydata([-550., 4000.])

        ph_term1.set_xdata([L_mu - 2.0*np.sqrt(L_sigma2), L_mu - 2.0*np.sqrt(L_sigma2)])
        ph_term1.set_ydata([-550., 4000.])

        ph_term2.set_xdata([L_mu + 2.0*np.sqrt(L_sigma2), L_mu + 2.0*np.sqrt(L_sigma2)])
        ph_term2.set_ydata([-550., 4000.])

        plt.pause(0.000000000000000001)


np.savetxt('filter/delta_temps1.txt', np.array(delta_temps))
np.savetxt('filter/delta_temps1.txt', np.array(Ls))

plt.plot(delta_temps)
plt.show()
