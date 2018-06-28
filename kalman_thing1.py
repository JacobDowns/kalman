import matplotlib.pyplot as plt
import numpy as np
from filterpy.kalman import MerweScaledSigmaPoints
from filterpy.kalman import JulierSigmaPoints
from filterpy.kalman import UnscentedKalmanFilter
from scipy.interpolate import interp1d


# delta T sigma points
X = np.loadtxt('jensen_sigma_points.txt')
N = X.shape[0]
# Sigma points run through the forward model F(X)
Y = np.loadtxt('filter/jensen_sigmas/all_Ls.txt')
# Prior mean and covariance
m = np.loadtxt('filter/jensen_sigmas/prior_m.txt')
P = np.loadtxt('filter/jensen_sigmas/prior_P.txt')
# Observed lengths
obs_ages = np.array([-11.6, -10.2, -9.2, -8.2, -7.3])*1e3
obs_Ls = [406878.12855486432, 396313.20004890749, 321224.04532276397, 292845.40895793668, 288562.44342502725]
L_interp = interp1d(obs_ages, obs_Ls, kind = 'linear')
model_ages = np.loadtxt('filter/jensen_sigmas/ages_0.txt')
L_obs = L_interp(model_ages[::5])
# Assumed observation covariance
R = 500.**2*np.identity(len(L_obs))
# Sigma points
points = JulierSigmaPoints(429, kappa=-300.)


### Compute unscented mean, measurement covariance, and cross covariance 
######################################################################

### Compute predicted mean
mu = np.dot(points.weights()[0], Y)

plt.subplot(3,1,1)
plt.plot(model_ages[::5], mu, 'ro-')
plt.plot(model_ages[::5], L_obs, 'ko-')


### Compute predicted measurement covariance
S = np.zeros((Y.shape[1], Y.shape[1]))
for i in range(len(points.weights()[1])):
    print i
    S += points.weights()[1][i]*np.outer(Y[i] - mu, Y[i] - mu)
S += R


### Compute predicted measurement covariance
C = np.zeros((X.shape[1], Y.shape[1]))
for i in range(len(points.weights()[1])):
    print i
    C += points.weights()[1][i]*np.outer(X[i] - m, Y[i] - mu)

    
### Compute Kalman gain, revised mean, and covariance
######################################################################

K = np.dot(C, np.linalg.inv(S))
m_p = m + np.dot(K, L_obs - mu)
P = P - np.dot(np.dot(K, S), K.T)

np.savetxt('opt_m.txt', m_p)

plt.subplot(3,1,2)
plt.plot(model_ages, np.repeat(m_p, 30))
plt.plot(model_ages, np.repeat(m, 30))

plt.subplot(3,1,3)
v = P[range(len(P)), range(len(P))]
plt.plot(v)
plt.show()
