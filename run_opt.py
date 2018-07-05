from model.inputs.paleo_inputs import *
from model.forward_model.forward_ice_model import *
import matplotlib.pyplot as plt
import numpy as np
import sys
import os.path

in_dir = 'filter/prior1/'
adots_opt = np.loadtxt(in_dir + 'opt_m1.txt')
delta_temps = np.repeat(adots_opt, 15)

dt = 1./3.
model_inputs = PaleoInputs('paleo_inputs/is_paleo_11_6_steady.hdf5', dt = dt)
model = ForwardIceModel(model_inputs, "out", "paleo")

N = 4295*3
Ls = []
ages = []

for j in range(N):
    print model.t + dt
    age = -11.6e3 + model.t + dt
    ages.append(age)

    L = model.step(delta_temps[j], accept = True)
    Ls.append(L)

plt.plot(ages, Ls)
plt.show()

np.savetxt('opt_ages.txt', np.array(ages))
np.savetxt('opt_L.txt', np.array(Ls))
