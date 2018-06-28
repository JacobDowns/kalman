from model.inputs.paleo_inputs import *
from model.forward_model.forward_ice_model import *
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from filterpy.kalman import JulierSigmaPoints
from filterpy.kalman import UnscentedKalmanFilter

dt = 1./3.
model_inputs = PaleoInputs('paleo_inputs/is_paleo_11_6_steady.hdf5', dt = dt)
model = ForwardIceModel(model_inputs, "out", "paleo")

### Setup filter
#######################################################################

# Process model function
def fx(x, dt):
    return x

# Measurement model function
def hx(x):
    return model.step(x, accept = False)

# Init. UKF
#points = MerweScaledSigmaPoints(n=1, alpha=.1, beta=2., kappa=1.)
points = JulierSigmaPoints(n=1, kappa = 2.)
ukf = UnscentedKalmanFilter(dim_x=1, dim_z=1, dt=dt, fx=fx, hx=hx, points=points)
# Initial mean
ukf.x = np.array([-8.11458677211])
# Initial variance
ukf.P *= 1.31015393411
# Process noise 
ukf.Q *= 0.005
# Observation noise
ukf.R *= 5000.0**2


### Observation function

ages = np.array([-11.6, -10.2, -9.2, -8.2, -7.3])*1e3
Ls = [406878.12855486432, 396313.20004890749, 321224.04532276397, 292845.40895793668, 288562.44342502725]
L_interp = interp1d(ages, Ls, kind = 'linear')

N = 4299*3
delta_temps = []
Ls = []
ages = []
sigma2s = []
# Process noise at each step 
qs = []
# Observation noise at each step
rs = []

for i in range(N):
    # Observation mean and covariance at the current time
    print model.t + dt
    age = -11.6e3 + model.t + dt
    ages.append(age)
    L_mu = L_interp(age)

    ukf.predict()
    ukf.update(L_mu)

    delta_temp = ukf.x.copy()[0]
    delta_temps.append(delta_temp)
    delta_temp_sigma2 = ukf.P.copy()[0][0]
    sigma2s.append(delta_temp_sigma2)

    L = model.step(delta_temp, accept = True)
    Ls.append(L)
    print delta_temp, delta_temp_sigma2
    print "dif", L - L_mu


np.savetxt('filter/4/ages.txt', np.array(ages))
np.savetxt('filter/4/delta_temps.txt', np.array(delta_temps))
np.savetxt('filter/4/delta_temps_sigma2.txt', np.array(sigma2s))
np.savetxt('filter/4/Ls.txt', np.array(Ls))

plt.plot(delta_temps)
plt.show()
