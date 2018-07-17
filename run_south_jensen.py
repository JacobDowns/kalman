from model.inputs.paleo_inputs import *
from model.forward_model.forward_ice_model import *
import matplotlib.pyplot as plt
import numpy as np
import sys
import os.path
from scipy.interpolate import interp1d

### Initialize model
############################################################################

dt = 1./3.
model_inputs = PaleoInputs('paleo_inputs/south_paleo_steady_11_6.hdf5', dt = dt)
model = ForwardIceModel(model_inputs, "out", "paleo")
model.sea_level.assign(Constant(-1000.))


### Load Jensen delta T's
############################################################################

data = np.loadtxt('paleo_data/jensen_dye3.txt')
# Years before present (2000)
years = data[:,0] - 2000.0
# Temps. in K
temps = data[:,1]
# Interp. delta temp. 
delta_temp_interp = interp1d(years, temps - temps[-1], kind = 'linear')

N = 4300*3
Ls = []
ages = []

for j in range(N):
    age = -11.6e3 + model.t + dt
    ages.append(age)

    delta_temp = delta_temp_interp(age)

    print "delta temp.", delta_temp
    print "age", age
    print age

    L = model.step(delta_temp, accept = True)
    Ls.append(L)

plt.plot(ages, Ls)
plt.show()

np.savetxt('paleo_runs/opt_ages_land.txt', np.array(ages))
np.savetxt('paleo_runs/opt_L_land.txt', np.array(Ls))
