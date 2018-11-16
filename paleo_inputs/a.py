import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal


Ls = np.array([406878.12855486432, 396313.20004890749, 321224.04532276397, 292845.40895793668, 288562.44342502725, 279753.70997966686])
#Ls = [424777.2650658561, 394942.08036138373, 332430.9181651594, 303738.499327732, 296659.0156905292, 284686.5963970118]
# Mean observation times
obs_ts = -np.array([11.6, 10.3, 9.0, 8.1, 7.3, 0.])
# Observation variances
obs_sigmas = np.array([0.38,  0.2,  0.20, 0.3, 0.3, 0.001]) / 2.


### Build GMRF covariance matrices for generating random paths between observation times
#########################################################################################

# Number of points in time
N = 2*233 - 1
# Path times
path_ts = np.linspace(-11.6, 0., N)
# Indexes corresponding to mean observation times
indexes = np.array([0, 52, 104, 140, 172, N-1])
# Dimension of the precision matrices
Q_ns = np.array(indexes[1:] - indexes[:-1]) - 2
# Covariance matrices
Ps = []
# Smoothness variable
delta = 0.0000000005

for i in range(len(Q_ns)):
    Q_n = Q_ns[i]
    Q_i = np.diag(2.*np.ones(Q_n), 0) + np.diag(-np.ones(Q_n - 1), -1) + np.diag(-np.ones(Q_n - 1), 1)
    Q_i = delta*Q_i
    print(Q_i.shape)
    Ps.append(np.linalg.inv(Q_i))

    
### Generate random path segments
#########################################################################################

num_samples = 25000
path_segments = []

for P_i in Ps:
    path_segment = np.random.multivariate_normal(np.zeros(len(P_i)), P_i, num_samples)
    path_segments.append(path_segment)


### Generate random paths and discard bad ones
#########################################################################################

# To store the good paths
good_paths = []

for i in range(len(samples)):
    if i % 25 == 0:
        print(i)
        
    # Randomize the moraine formation times
    rand_ts = obs_ts + np.random.randn(len(obs_ts))*obs_sigmas

   
    path = samples[i]

    """
    L_indexes = np.where(samples[i] > Ls[1])
    if len(L_indexes[0]) > 0:
        formation_time = path_ts[L_indexes[0].max()]
        time_dif = formation_time - ts[1]
        difs.append(time_dif)"""
    
    times = []
    for j in range(1, 5):
        L_indexes = np.where(path > Ls[j])

        # Get the last index at which the glacier length exceeds the given
        # moraine position
        if len(L_indexes[0]) > 0:
            # Get the time at which the moraine formed
            formation_time = path_ts[L_indexes[0].max()]
            times.append(formation_time)
        else:
            break

    if len(times) == 4:
        times = np.array(times)
        u = np.random.uniform(0., 1.5*pdf.pdf(times)) + 1e-9
        #print(u)
        #print(pdf.pdf(times))
        
        if u <= pdf.pdf(times):
            print(u)
            plt.plot(path_ts, path)
            good_samples.append(path)
            print("goodly sample", times)
        
    

print(len(good_samples))
y = np.array(good_samples).mean(axis = 0)
plt.plot(path_ts, y, 'k', lw = 4)
plt.show()

# Covariance
P = np.zeros((N,N))
for i in range(len(good_samples)):
    P += np.outer(good_samples[i] - y, good_samples[i] - y)

P /= len(P) - 1
v = np.sqrt(P[range(N), range(N)])

print(v.max())

plt.plot(path_ts, y + 2.*v, 'ko--', lw = 3)
plt.plot(path_ts, y - 2.*v, 'ko--', lw = 3)
plt.show()

#np.savetxt('y_c.txt', y)
#np.savetxt('Py_c.txt', P)



