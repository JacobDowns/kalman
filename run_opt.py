
from model.inputs.paleo_inputs1 import *
from model.forward_model.forward_ice_model import *
import matplotlib.pyplot as plt
import numpy as np
import sys
import os.path

in_dir = 'filter/prior5/'
adots_opt = np.loadtxt(in_dir + 'opt_m3.txt')

dt = 1./3.
model_inputs = PaleoInputs('paleo_inputs/is_paleo_11_6_steady.hdf5', dt = dt)
model = ForwardIceModel(model_inputs, "out", "paleo")

sigma_ts = np.loadtxt(in_dir + 'sigma_ts.txt')
model_ts = -11.6e3 + np.linspace(0., 4300., 4300*3)
delta_temp_interp = interp1d(sigma_ts, adots_opt, kind = 'linear')
delta_temps = delta_temp_interp(model_ts)

"""
xs = model_inputs.mesh.coordinates() * model_inputs.L_init
L_init = model_inputs.L_init
plt.plot(xs, project(model.B).compute_vertex_values(), lw = 2)
plt.plot(xs, project(model.B + model.H0_c).compute_vertex_values(), lw = 2)
plt.ylim([-250., 2000.])
plt.xlim([0., L_init])
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.show()"""

dolfin.plot(model.adot_prime)
plt.show()

quit()

N = 4300*3
Ls = []
ages = []

for j in range(N):
    age = -11.6e3 + model.t + dt
    ages.append(age)

    print age

    L = model.step(delta_temps[j], accept = True)
    Ls.append(L)

plt.plot(ages, Ls)
plt.show()

np.savetxt('opt_ages3.txt', np.array(ages))
np.savetxt('opt_L3.txt', np.array(Ls))
