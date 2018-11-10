import matplotlib.pyplot as plt
import numpy as np

# Observation times
ts = -np.array([11.6, 10.3, 9.0, 8.1, 7.3, 0.])
# Observaion lengths
Ls = np.array([406878.128554864, 396313.200048907, 321224.045322763, 292845.408957936, 288562.443425027, 279753.709979666])
# Time standard deviation
sigmas = np.array([0.38,  0.19,  0.20, 0.29, 0.28, 0.]) / 2.
# Learned mean
x = np.loadtxt('x.txt')
# Learned standard deviation
P = np.loadtxt('P.txt')
# Times for mean
N = len(x)
path_ts = np.linspace(-11.6, 0., N)
indexes = []


### Generate random paths
#########################################################################################

# Samples that meet the criteria
good_samples = []
# Random samples
samples = np.random.multivariate_normal(x, P, 500)

for i in range(len(samples)):
    plt.plot(path_ts, samples[i], lw = 1.5)

v = np.sqrt(P[range(N), range(N)])
plt.plot(path_ts, x, 'k', lw = 5)
plt.plot(path_ts, x + v, 'k--', lw = 5)
plt.plot(path_ts, x - v, 'k--', lw = 5)
plt.plot(path_ts, x + 2.*v, 'k--', lw = 5)
plt.plot(path_ts, x - 2.*v, 'k--', lw = 5)
plt.plot(ts, Ls, 'ro-', lw = 5, ms = 5)
plt.show()


