from model.inputs.paleo_inputs import *
from model.forward_model.forward_ice_model import *
import matplotlib.pyplot as plt
import numpy as np

model_inputs = PaleoInputs('paleo_inputs/inverse_start.hdf5', dt = 0.33)
model = ForwardIceModel(model_inputs, "out", "paleo")


# UKF v. particle filter mean?

mean = -5.158476
dev = 2.5

delta_temps = dev*np.random.randn(1000) + mean

Ls = []

for delta_temp in delta_temps:
    print delta_temp
    Ls.append(model.step(delta_temp, accept = False) - model_inputs.L_init)
    dolfin.plot(model.adot_prime_func)

#plt.hist(delta_temps)
plt.ylim([-2.5, 1.])

print np.array(Ls).mean()

#plt.hist(Ls, bins = 30)
#plt.plot(delta_temps,Ls, 'ko-')
plt.show()
