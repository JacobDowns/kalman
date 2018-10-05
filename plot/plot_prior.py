import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from filterpy.kalman import JulierSigmaPoints
import seaborn as sns

plt.rcParams.update({'font.size': 22})


### Prior mean
####################################################
N = 45
x = np.zeros(N)

### Prior covariance
####################################################

delta = 250e3
# Covariance matrix
P = np.zeros((N, N))
P[range(N), range(N)] = 2.
P[range(1,N), range(N-1)] = -1.
P[range(N-1), range(1,N)] = -1.
P[N-1, N-1] = 2.
P = delta*P
P = np.linalg.inv(P)

samples = np.random.multivariate_normal(x, P, 50)
for i in range(samples.shape[0]):
    plt.plot(samples[i])
plt.show()
