import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from filterpy.kalman import JulierSigmaPoints

# Output directory
out_dir = 'filter/prior1/'
# Load Jensen dye3 temp.
data = np.loadtxt('jensen_dye3.txt')
# Years before present (2000)
years = data[:,0] - 2000.0
# Temps. in K
temps = data[:,1]
# Delta temps. 
delta_temp_interp = interp1d(years, temps - temps[-1], kind = 'linear')
# Model time steps
years = -11.6e3 + np.arange(0., 4295., 5.)


### Mean and covariance of prior
##########################################################################
N = len(years)
x = delta_temp_interp(years)
P = np.zeros((N, N))
P[range(N), range(N)] = 2.
P[range(1,N), range(N-1)] = -1.
P[range(N-1), range(1,N)] = -1.
P[N-1, N-1] = 1.
P = 300.*P
P = np.linalg.inv(P)

np.savetxt(out_dir + 'prior_m.txt', x)
np.savetxt(out_dir + 'prior_P.txt', P)


### Plot samples from prior
##########################################################################
samples = np.random.multivariate_normal(x, P, 500)
for i in range(samples.shape[0]):
    plt.plot(samples[i])
plt.show()


### Compute sigma points
##########################################################################
points = JulierSigmaPoints(N, kappa=-len(x)/2)
sigma_points = points.sigma_points(x, P)
for i in range(100, 1600):
    plt.plot(sigma_points[i])
plt.show()

print sigma_points.shape
quit()

np.savetxt(out_dir + 'prior_sigma_points.txt', sigma_points)

