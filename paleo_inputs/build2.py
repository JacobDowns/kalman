import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal


Ls = np.array([406878.12855486432, 396313.20004890749, 321224.04532276397, 292845.40895793668, 288562.44342502725, 279753.70997966686])
# Mean 
ts = -np.array([11.6, 10.3, 9.0, 8.1, 7.3, 0.])
sigmas = np.array([0.1,  0.19,  0.20, 0.29, 0.28, 0.1]) / 2.
# Number of samples
num_samples = 10000

pdf_mean = np.array(ts[1:-1])
pdf_cov = np.diag(sigmas[1:-1]**2, 0)
pdf = multivariate_normal(pdf_mean, pdf_cov)


### Build GMRF covariance matrices for generating random paths between observation times
#########################################################################################

x = np.loadtxt('x3.txt')
P = np.loadtxt('P3.txt')
N = len(x)
path_ts = np.linspace(-11.6, 0., N)


### Generate random paths
#########################################################################################

# Samples that meet the criteria
samples = np.random.multivariate_normal(x, P, 2500)
good_samples = []
for i in range(len(samples)):
    #plt.plot(samples[i])

    if i % 25 == 0:
        print(i)
   
    path = samples[i]

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
        u = np.random.uniform(0., 10.*pdf.pdf(times)) + 1e-16
        #print(u)
        #print(pdf.pdf(times))
        
        if u <= pdf.pdf(times):
            print(u)
            plt.plot(path_ts, path)
            good_samples.append(path)
            print("goodly sample", times)
        
    
   


print(len(good_samples))
x = np.array(good_samples).mean(axis = 0)
plt.plot(path_ts, x, 'k', lw = 4)

print(x)

# Covariance
P = np.zeros((N,N))
for i in range(len(good_samples)):
    P += np.outer(good_samples[i] - x, good_samples[i] - x)

P /= len(P) - 1
plt.plot(path_ts, x + 2.*np.sqrt(P[range(N), range(N)]), 'k--', lw = 4)
plt.plot(path_ts, x + np.sqrt(P[range(N), range(N)]), 'k--', lw = 4)
plt.plot(path_ts, x - 2.*np.sqrt(P[range(N), range(N)]), 'k--', lw = 4)
plt.plot(path_ts, x - np.sqrt(P[range(N), range(N)]), 'k--', lw = 4)
plt.plot(ts, Ls, 'ro-', lw = 4)
plt.show()

np.savetxt('x4.txt', x)
np.savetxt('P4.txt', P)

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



