from model.inputs.paleo_inputs import *
from model.forward_model.forward_ice_model import *
import matplotlib.pyplot as plt
import numpy as np
from scalar_ukf import ScalarUKF
from scipy.interpolate import interp1d

dt = 1./3.
model_inputs = PaleoInputs('paleo_inputs/is_paleo_11_6_steady.hdf5', dt = dt)
model = ForwardIceModel(model_inputs, "out", "paleo")

def softplus(y1,y2,alpha=0.5):
    # The softplus function is a differentiable approximation
    # to the ramp function.  Its derivative is the logistic function.
    # Larger alpha makes a sharper transition.
    return dolfin.Max(y1,y2) + (1./alpha)*dolfin.ln(1.+dolfin.exp(alpha*(dolfin.Min(y1,y2)-dolfin.Max(y1,y2))))

min_thickness = Function(model_inputs.V_cg)
min_thickness.interpolate(Constant(15.))

#dolfin.plot(project(softplus(model_inputs.original_cg_functions['B'], min_thickness), model_inputs.V_cg))
#plt.show()


quit()

### Setup filter
#######################################################################

# Initial mean and variance for state variable delta_temp
delta_temp_mu = -8.11458677211
delta_temp_sigma2 = 1.31015393411

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


### Observation function

ages = np.array([-11.6, -10.2, -9.2, -8.2, -7.3])*1e3
Ls = [406878.12855486432, 396313.20004890749, 321224.04532276397, 292845.40895793668, 288562.44342502725]
L_interp = interp1d(ages, Ls, kind = 'linear')


N = 4299*3
delta_temps = []
Ls = []
ages = []
sigma2s = []
adot_ints = []


for i in range(N):
    # Observation mean and covariance at the current time
    print model.t + dt
    age = -11.6e3 + model.t + dt
    ages.append(age)
    L_mu = L_interp(age)
    L_sigma2 = 6000.0**2
    # Process noise
    Q = 0.005
    print "L : ", (L_mu, L_sigma2)

    delta_temp, delta_temp_sigma2 = ukf.step(L_mu, L_sigma2, Q)
    delta_temps.append(delta_temp)
    sigma2s.append(delta_temp_sigma2)

    print "opt delta temp", delta_temp, delta_temp_sigma2
    L = model.step(delta_temp, accept = True)
    Ls.append(L)

    adot_int = assemble(model.adot_prime_func*dx)
    adot_ints.append(adot_int)
    print "adot_int", adot_int
    print "dif", L - L_mu


np.savetxt('filter/delta_temps1.txt', np.array(delta_temps))
np.savetxt('filter/Ls1.txt', np.array(Ls))
np.savetxt('filter/adot_ints1.txt', np.array(adot_ints))
plt.plot(delta_temps)
plt.show()
