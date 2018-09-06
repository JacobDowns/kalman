import numpy as np
from model.inputs.paleo_inputs import *
from model.forward_model.forward_ice_model import *
import matplotlib.pyplot as plt

"""
Tune beta2 to get a reasonable sliding velocity.
"""

### Model inputs
#######################################################

# Model input dictionary
inputs = {}
inputs['dt'] = 1./4.
inputs['beta2'] = 2e-3
# Create model inputs
model_inputs = PaleoInputs('paleo_inputs/is_real.h5', inputs)
model = ForwardIceModel(model_inputs, 'out', 'out')

#dolfin.plot(model.H0_c)
for i in range(50):
    model.step(0., True)


us = project(model.momentum_form.u(0))

#print us
dolfin.plot(abs(us))
#dolfin.plot(model.u2n)
plt.show()

