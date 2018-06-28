import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from filterpy.kalman import JulierSigmaPoints



data = np.loadtxt('jensen_dye3.txt')
# Years before present (2000)
years = data[:,0] - 2000.0
# Temps. in K
temps = data[:,1]

delta_temp_interp = interp1d(years, temps - temps[-1], kind = 'linear')
years = -11.6e3 + np.arange(0., 4290., 10.)

#plt.plot(years, delta_temp_interp(years))
#plt.show()
#print len(years)

N = len(years)
x = delta_temp_interp(years)
P = np.zeros((N, N))
P[range(N), range(N)] = 2.
P[range(1,N), range(N-1)] = -1.
P[range(N-1), range(1,N)] = -1.
P[N-1, N-1] = 1.
P = 75.*P
P = np.linalg.inv(P)

np.savetxt('filter/jensen_sigmas/prior_m.txt', x)
np.savetxt('filter/jensen_sigmas/prior_P.txt', P)


samples = np.random.multivariate_normal(x, P, 25)
print samples.shape

for i in range(samples.shape[0]):
    plt.plot(samples[i])

plt.show()

points = JulierSigmaPoints(N, kappa=-300.)


sigma_points = points.sigma_points(x, P)

print sigma_points.shape

for i in range(100, 800):
    plt.plot(sigma_points[i])

plt.show()



np.savetxt('jensen_sigma_points.txt', sigma_points)
#np.savetxt(

#print P
