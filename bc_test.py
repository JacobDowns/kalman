from model.inputs.paleo_inputs import *
from model.forward_model.forward_ice_model import *
import matplotlib.pyplot as plt
import numpy as np
from scalar_ukf import ScalarUKF
from scipy.interpolate import interp1d

dt = 1.
model_inputs = PaleoInputs('paleo_inputs/south_ideal.h5', dt = dt)
model = ForwardIceModel(model_inputs, "out", "bc_test")
model.sea_level.assign(-47.)


### Setup filter
#######################################################################

# Initial mean and variance for state variable delta_temp
delta_temp_mu = -8.11458677211
delta_temp_sigma2 = 1.31015393411


sea_level = Function(model_inputs.V_cg)
sea_level.interpolate(model.sea_level)

"""
B = model_inputs.original_cg_functions['B']
#dolfin.plot(model_inputs.original_cg_functions['B'])
#dolfin.plot(f)
#plt.show()
rho_w = 1000.
rho = 900.

def softplus(y1,y2,alpha=1):
    # The softplus function is a differentiable approximation
    # to the ramp function.  Its derivative is the logistic function.
    # Larger alpha makes a sharper transition.
    return dolfin.Max(y1,y2) + (1./alpha)*dolfin.ln(1.+dolfin.exp(alpha*(dolfin.Min(y1,y2)-dolfin.Max(y1,y2))))

# Grounding line thickness
H_g = softplus(Constant(rho_w / rho)*(sea_level - B), Constant(15.), alpha = 0.5)
dolfin.plot(H_g, 'ko-')
plt.show()"""

for i in range(1000):
    model.step(delta_temp_mu, accept = True)


dolfin.plot(model.H0_c + model.B)
dolfin.plot(model.B)
dolfin.plot(sea_level)
plt.ylim([-100., 2000.])
plt.show()
quit()

for i in range(N):
    # Observation mean and covariance at the current time
    print model.t + dt
    age = -11.6e3 + model.t + dt
    ages.append(age)
   
    L = model.step(delta_temp_mu, accept = True)
    Ls.append(L)


