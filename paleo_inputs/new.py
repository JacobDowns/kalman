import matplotlib.pyplot as plt
import numpy as np


Ls = np.array([406878.12855486432, 396313.20004890749, 321224.04532276397, 292845.40895793668, 288562.44342502725, 279753.70997966686])

# Mean 
ts = -np.array([11.6, 10.3, 9.0, 8.1, 7.3, 0.])
sigmas = np.array([0.0,  0.19,  0.20, 0.29, 0.28, 0.]) / 2.
# Number of samples
num_samples = 100
# Observation time noise samples
t_samples = np.random.rand(len(ts), num_samples).T*sigmas


### Build GMRF covariance matrices for generating random paths between observation times
#########################################################################################
Pns = np.array(np.abs(ts[1:] - ts[:-1])*10 + 1, dtype = int)
Ps = []
delta = 0.0000000005
for i in range(len(Pns)):
    Pn = Pns[i]
    Q_i = np.diag(2.*np.ones(Pn - 1), 0) + np.diag(-np.ones(Pn - 2), -1) + np.diag(-np.ones(Pn - 2), 1)
    Q_i = delta*Q_i
    Ps.append(np.linalg.inv(Q_i))


### Generate random paths
#########################################################################################

path_segments = []

for i in range(len(Pns)):
    

for i in range(num_samples):
    # Add random noise to the time observations
    random_ts = ts + sigmas*np.random.randn(len(ts))

    print(random_ts)
    # Generate a random path that has the right length at the randomized ts
    

quit()


quit()
# Add random noise to observation points
sigma = 100.
obs_ts[1:] += 100.*np.random.randn(6)
# Number of subsample points
n_sub = 9
t = np.arange(0, n_sub * len(obs_ts), n_sub)
tt = np.arange((len(obs_ts) - 1) * n_sub + 1)
# Array of times including observations times and subsample times
ts = np.interp(tt, t, obs_ts)
# Offsets at observations times and subsample times
Ls = np.interp(ts, obs_ts, obs_offsets)


### Add noise between observation points
################################################################################

Q = np.diag(2.*np.ones(n_sub - 1), 0) + np.diag(-np.ones(n_sub - 2), -1) + np.diag(-np.ones(n_sub - 2), 1)
delta = 0.0000000005
h = ts[1]
Q = delta * Q

sample_indexes = np.ones(len(ts))
sample_indexes[::n_sub] = 0.

# Distances between subsample points
hs = (ts[1:] - ts[:-1])
hs = hs[np.mod(np.arange(hs.size),n_sub)!=0]

noise =  (1. / hs) * np.random.multivariate_normal(np.zeros(n_sub - 1), np.linalg.inv(Q), 6).flatten()
Ls[sample_indexes == 1.] += noise



