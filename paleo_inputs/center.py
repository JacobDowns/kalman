import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal


Ls = np.array([406878.12855486432, 396313.20004890749, 321224.04532276397, 292845.40895793668, 288562.44342502725, 279753.70997966686])
#Ls = [424777.2650658561, 394942.08036138373, 332430.9181651594, 303738.499327732, 296659.0156905292, 284686.5963970118]
# Mean observation times
obs_ts = np.array([-11686.0, -10416.0, -9156.0, -8196.0, -7366.0, 0.])
# Observation variances
obs_sigmas = np.array([0.4, 0.2, 0.2, 0.3, 0.3, 1e-16])*1000. / 2.
Pt = np.diag(obs_sigmas**2)
# Number of random paths to generate
num_samples = 200000


### Build GMRF covariance matrices for generating random paths between observation times
#########################################################################################

# Path times
path_ts = np.arange(obs_ts.min(), 0., 25.)
# Number of points in time
N = len(path_ts)
# Smoothness variable
delta = 0.000001
Q = np.diag(2.*np.ones(N), 0) + np.diag(-np.ones(N - 1), -1) + np.diag(-np.ones(N - 1), 1)
Q = delta*Q
# GMRF covariance
Py = np.linalg.inv(Q)


### Generate random paths
#########################################################################################
rand_paths = np.random.multivariate_normal(np.zeros(len(Py)), Py, num_samples)
rand_ts = np.random.multivariate_normal(obs_ts, Pt, num_samples) - obs_ts[0]

#rand_ts[:,0] = abs(rand_ts[:,0])
#print(rand_ts)
#quit()

nearest_index = np.array(rand_ts / 25., dtype = int) + np.array(np.round((rand_ts / 25.) % 1), dtype = int)
nearest_time = nearest_index*25 - obs_ts.min()

good_paths = []
good_ts = []
for i in range(num_samples):
    rand_path = rand_paths[i]
    t_indexes = nearest_index[i]
    rand_t = path_ts[t_indexes]
    path = np.interp(path_ts, rand_t, Ls - rand_path[t_indexes]) + rand_path

    # Check if the moraine formation time is good
    keep = True

    #print(t_indexes)
    for j in [1, 2, 3, 4]:
        if np.where(path > Ls[j])[0].max() > t_indexes[j]:
            keep = False
            break
        
    if keep:
        good_paths.append(path)
        good_ts.append(rand_t)
        #plt.plot(path_ts, path)
        print("yes")

good_ts = np.array(good_ts)
good_paths = np.array(good_paths)
print(len(good_paths))
v = np.std(good_paths, axis = 0)

### Mean 
y = np.array(good_paths).mean(axis = 0)
plt.plot(path_ts, y, 'k', lw = 4)
plt.plot(path_ts, y + 2.0*v, 'k', lw = 4)
plt.plot(path_ts, y - 2.0*v, 'k', lw = 4)
plt.plot(obs_ts, Ls, 'ro-', lw = 4, ms = 4)
plt.show()

Py = np.diag(v**2, 0)

np.savetxt('y_c.txt', y)
np.savetxt('Py_c.txt', Py)
np.savetxt('y_ages.txt', path_ts)
