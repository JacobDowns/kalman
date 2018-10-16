import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from filterpy.kalman import JulierSigmaPoints
import seaborn as sns

plt.rcParams.update({'font.size': 16})

fig = plt.figure(figsize=(9,6))
current_palette = np.array(sns.color_palette())
#current_palette = np.array(sns.color_palette("muted", 10)[1:])
#current_palette = current_palette[[0,2,3]]

### Prior mean
##########################################################################

N = 45
x = np.zeros(N)
ts = (-11.6e3 + np.linspace(0., 11590, N)) / 1000.
chi = np.linspace(0., 1., len(ts))
x = np.zeros(len(ts)) #0.45*np.ones(len(ts)) - 0.45*(chi)**4


### Prior covariance
##########################################################################

delta = 250e3
# Covariance matrix
P = np.zeros((N, N))
P[range(N), range(N)] = 2.
P[range(1,N), range(N-1)] = -1.
P[range(N-1), range(1,N)] = -1.
P = delta*P
P = np.linalg.inv(P)


### Prior plot
##########################################################################

#current_palette = sns.color_palette("deep", 12)
samples = np.random.multivariate_normal(x, P, 15)
ts /= 1000.

for i in range(samples.shape[0]):
    plt.plot(ts, samples[i], ms = 3, linewidth = 2., color = current_palette[i % 10], marker = 'o')

plt.grid(color='slategray', linestyle=':', linewidth=0.75)
plt.xlim([ts.min(), ts.max()])
plt.ylabel(r'$\Delta P$ (m.w.e. a$^{-1}$)')
plt.xlabel('Age (ka BP)')
plt.ylim([-0.02, 0.02])


plt.savefig('prior.png', dpi=700)
