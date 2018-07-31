import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from kalman_update import *

in_dir = 'filter/north_prior2/'

### Observations 
#############################################################

#obs_ages = np.array([-11.6, -10.2, -9.2, -8.2, -7.3])*1e3
#obs_Ls = [406878.12855486432, 396313.20004890749, 321224.04532276397, 292845.40895793668, 288562.44342502725]
# Observed ages 
obs_ages = np.array([-11.6, -10.2, -9.2, -8.2, -7.3])*1e3
# Observed lengths
obs_Ls = np.array([443746.66897917818, 397822.86008538032, 329757.49741948338, 292301.29712071194, 285478.05793305294])
# Interpolate the observations 
L_interp = interp1d(obs_ages, obs_Ls, kind = 'linear')
# Model time steps 
model_ages = np.loadtxt(in_dir + 'ages_0.txt')
# To reduce computation time, we only  use observations at periodic intervals for the kalman update
#skip = 4
obs_indexes = np.array([0, 1400*3, 2400*3, 3400*3, len(model_ages) - 2])

#print model_ages[obs_indexes]
#quit()
#obs_indexes = range(len(model_ages))[::skip]
# Observation
y = L_interp(model_ages[obs_indexes])
# Assumed observation covariance
#R = 500.**2 * np.identity(len(y))
R = np.zeros((len(y), len(y)))
error_ts = np.array([-11.6, -10.9, -10.2, -9.7, -9.2, -8.7, -8.2, -7.75, -7.3])*1e3
min_err = 5000.**2
max_err = 50000.**2 #1000.**2
error_vs = np.array([min_err,  max_err,  min_err,   max_err,  min_err,   max_err,  min_err,   max_err,  min_err])
error_interp = interp1d(error_ts, error_vs, kind = 'linear')
errors = error_interp(model_ages[obs_indexes])
R[range(R.shape[0]), range(R.shape[0])] = errors


### Load stuff we need for the unscented transform 
#############################################################

# Sigma points
X = np.loadtxt(in_dir + 'X.txt')
# Sigma points run through 
Y = np.loadtxt(in_dir + 'Y.txt')
Y = Y[:, obs_indexes]
# Prior mean
m = np.loadtxt(in_dir + 'prior_m.txt')
# Prior covariance 
P = np.loadtxt(in_dir + 'prior_P.txt')
# Load mean weights
m_weights = np.loadtxt(in_dir + 'm_weights.txt')
# Load covariance weights
c_weights = np.loadtxt(in_dir + 'c_weights.txt')


### Do Kalman update
#############################################################

# Object for doing the Kalman update computations
ku = KalmanUpdate(m, P, X, Y, m_weights, c_weights, obs_indexes)

m_p, P_p, mu, K = ku.update(y, R)

plt.plot(m, 'ko')
v = P_p[range(len(P)), range(len(P))]
plt.plot(m_p, 'r')
plt.plot(m_p - 2.0*np.sqrt(v), 'r-')
plt.plot(m_p + 2.0*np.sqrt(v), 'r-')
plt.show()

np.savetxt(in_dir + 'var.txt', v)
np.savetxt(in_dir + 'mu.txt', mu)
np.savetxt(in_dir + 'opt_m.txt', m_p)
np.savetxt(in_dir + 'opt_P.txt', P_p)



"""
np.savetxt('opt_m.txt', m_p)

plt.subplot(3,1,2)
plt.plot(model_ages, np.repeat(m_p, 30))
plt.plot(model_ages, np.repeat(m, 30))

plt.subplot(3,1,3)
v = P[range(len(P)), range(len(P))]
plt.plot(v)
plt.show()"""

