import matplotlib.pyplot as plt
import numpy as np


Ls = np.array([406878.12855486432, 396313.20004890749, 321224.04532276397, 292845.40895793668, 288562.44342502725, 279753.70997966686])

# Mean 
ts = -np.array([11.6, 10.3, 9.0, 8.1, 7.3, 0.])
sigmas = np.array([0.38,  0.19,  0.20, 0.29, 0.28, 0.]) / 2.
# Number of samples
num_samples = 1000
# Observation time noise samples
t_samples = np.random.rand(len(ts), num_samples).T*sigmas


### Build GMRF covariance matrices for generating random paths between observation times
#########################################################################################

N = 1000
x = np.zeros(N)
delta = 0.000001
Q = np.diag(2.*np.ones(N), 0) + np.diag(-np.ones(N - 1), -1) + np.diag(-np.ones(N - 1), 1)
Q = delta*Q
P = np.linalg.inv(Q)
path_ts = np.linspace(-11.6, 0., N)

### Generate random paths
#########################################################################################

samples = np.random.multivariate_normal(x, P, 1000000)
difs = []

for i in range(num_samples):

    #plt.plot(samples[i])
    if i % 25 == 0:
        print(i)
        
    path = samples[i]
    
    L_indexes = np.where(abs(path) > 30000.)
    print(L_indexes)

    if len(L_indexes[0]) > 0:
        #print("asdf")
        # Get the time at which the moraine formed
        formation_time = path_ts[L_indexes[0].max()]
        difs.append(formation_time)
        


plt.hist(difs, bins = 'auto')
plt.show()
quit()
print(len(samples))
# Mean 
x = np.array(samples).mean(axis = 0)
plt.plot(path_ts, x, 'k', lw = 4)

#np.savetxt('x.txt', x)

# Covariance
P = np.zeros((N,N))
for i in range(len(samples)):
    P += np.outer(samples[i] - x, samples[i] - x)

P /= len(P-1)
plt.plot(path_ts, x + 2.*np.sqrt(P[range(N), range(N)]), 'k--', lw = 4)
plt.plot(path_ts, x + np.sqrt(P[range(N), range(N)]), 'k--', lw = 4)
plt.plot(path_ts, x - 2.*np.sqrt(P[range(N), range(N)]), 'k--', lw = 4)
plt.plot(path_ts, x - np.sqrt(P[range(N), range(N)]), 'k--', lw = 4)
plt.plot(ts, Ls, 'ro-', lw = 4)
plt.show()
#np.savetxt('P.txt', P)



quit()
#plt.show()
#quit()
plt.imshow(np.linalg.inv(P))
plt.colorbar()
plt.show()
quit()
    


plt.plot(path_ts, np.array(samples).mean(axis = 0), 'k',  lw = 5)


plt.show()




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

noise =  (1. / hs) *  np.random.multivariate_normal(np.zeros(n_sub - 1), np.linalg.inv(Q), 6).flatten()
Ls[sample_indexes == 1.] += noise


