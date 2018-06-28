from model.inputs.paleo_inputs import *
from model.forward_model.forward_ice_model import *
import matplotlib.pyplot as plt
import numpy as np
from scalar_ukf import ScalarUKF
from scipy.interpolate import interp1d

dt = 1./3.
model_inputs = PaleoInputs('paleo_inputs/is_paleo_11_6_steady.hdf5', dt = dt)
model = ForwardIceModel(model_inputs, "out", "paleo")

smoothed_delta_temps = np.loadtxt('filter/3/delta_temps_smoothed.txt')

N = 4299*3
print N
ages = []
Ls = []

for i in range(N):
    # Observation mean and covariance at the current time
    print model.t + dt
    age = -11.6e3 + model.t + dt
    ages.append(age)
    delta_temp = smoothed_delta_temps[i]
    L = model.step(delta_temp, accept = True)
    Ls.append(L)


np.savetxt('filter/3/Ls_smoothed.txt', Ls)
plt.plot(ages, Ls)
plt.show()
