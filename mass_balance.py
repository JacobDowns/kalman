from model.inputs.paleo_inputs import *
from model.forward_model.forward_ice_model import *
import matplotlib.pyplot as plt
import numpy as np

### Setup model
################################################################################

dt = 1./3.
# Start from 11.3 thousand years ago
#model_inputs = PaleoInputs('paleo_inputs/start_11_3.hdf5', dt = dt)
model_inputs = PaleoInputs('paleo_inputs/problem_point.hdf5', dt = dt)
model = ForwardIceModel(model_inputs, "out", "paleo")


delta_temps = np.linspace(-15., 15.)
adot_ints = []

for delta_temp in delta_temps:
    model_inputs.update_inputs(model_inputs.L_init, adot0 = delta_temp)
    adot_ints.append(assemble(model_inputs.adot*dx))

plt.plot(delta_temps, adot_ints)
print adot_ints

plt.show()
