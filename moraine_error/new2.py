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
P = 4.*np.loadtxt('P.txt')
# Times for mean
N = len(x)
path_ts = np.linspace(-11.6, 0., N)
indexes = []



### Generate random paths
#########################################################################################

# Samples that meet the criteria
good_samples = []
# Random samples
samples = np.random.multivariate_normal(x, P, 2000000)

for i in range(len(samples)):
    path = samples[i]
    if i % 25 == 0:
        print(i)

    # Only keep good samples
    keep = True
    for j in range(5):
        L_indexes = np.where(path > Ls[j])

        # Get the last index at which the glacier length exceeds the given
        # moraine position
        if len(L_indexes[0]) > 0:
            # Get the time at which the moraine formed
            formation_time = path_ts[L_indexes[0].max()]
            # How far off is this formation time from the predicted time?
            time_dif = abs(formation_time - ts[j])
            # Draw a random number with the same observation variance to
            # determine if we keep this path
            keep = abs(1.0*sigmas[j]*np.random.randn(1)[0]) > time_dif
            
            if not keep:
                #plt.plot(path_ts, path)
                #plt.plot(ts, Ls)
                #plt.show()
                #print(time_dif)
                break 

    #print("Keep", keep)
    if keep:
        print("yes")
        good_samples.append(path)
        plt.plot(path_ts, path)

    
print(len(good_samples))
# Mean 
x = np.array(good_samples).mean(axis = 0)
plt.plot(path_ts, x, 'k', lw = 4)

np.savetxt('x1.txt', x)
# Covariance
P = np.zeros((N,N))
for i in range(len(good_samples)):
    print(i)
    P += np.outer(good_samples[i] - x, good_samples[i] - x)

P /= len(P-1)
plt.plot(path_ts, x + 2.*np.sqrt(P[range(N), range(N)]), 'k--', lw = 4)
plt.plot(path_ts, x + np.sqrt(P[range(N), range(N)]), 'k--', lw = 4)
plt.plot(path_ts, x - 2.*np.sqrt(P[range(N), range(N)]), 'k--', lw = 4)
plt.plot(path_ts, x - np.sqrt(P[range(N), range(N)]), 'k--', lw = 4)
plt.plot(ts, Ls, 'ro-', lw = 4)
plt.xlim([ts.min(), ts.max()])
plt.show()
np.savetxt('P1.txt', P)

plt.imshow(P)
plt.colorbar()
plt.show()

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



