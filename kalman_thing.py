import matplotlib.pyplot as plt
import numpy as np
from filterpy.kalman import MerweScaledSigmaPoints
from filterpy.kalman import JulierSigmaPoints
from filterpy.kalman import UnscentedKalmanFilter

### Load sigma points
######################################################################

# delta T sigma points
sigma_points = np.loadtxt('jensen_sigma_points.txt')
N = sigma_points.shape[0]
# Sigma points run through the forward model 
sigma_Ls = np.loadtxt('filter/jensen_sigmas/all_Ls.txt')
# Load prior mean
m = np.loadtxt('filter/jensen_sigmas/prior_m.txt')
P = np.loadtxt('filter/jensen_sigmas/prior_P.txt')


# Setup the Kalman filter
######################################################################

# Observed lengths
Ls = [406878.12855486432, 396313.20004890749, 321224.04532276397, 292845.40895793668, 288562.44342502725]

# Observation model -- this is a hack to get load in some pre computed stuff
def hx(x):
    sigma_index = np.argmin(np.sum(np.abs(sigma_points - x), axis = 1))

    print "x", x
    print "here"
    print sigma_points - x

    quit()
    return sigma_Ls[sigma_index]

# Dummy fx (not used)
def fx(x, dt):
    return x


# Sigma points
points = JulierSigmaPoints(429, kappa=-300.)
# Unscented Kalman filter
ukf = UnscentedKalmanFilter(dim_x=429, dim_z=len(Ls), dt=1., fx=fx, hx=hx, points=points)



ukf.predict()
# Initial mean
ukf.x[:] = m
# Intial covariance
ukf.P[:,:] = P
# Observation noise
ukf.R *= 0.001#5000.**2
ukf.update(Ls)

plt.plot(m, 'r')
plt.plot(ukf.x, 'k')

print m - ukf.x
plt.show()


