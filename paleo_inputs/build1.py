import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal


Ls = np.array([406878.12855486432, 396313.20004890749, 321224.04532276397, 292845.40895793668, 288562.44342502725, 279753.70997966686])
#Ls = [424777.2650658561, 394942.08036138373, 332430.9181651594, 303738.499327732, 296659.0156905292, 284686.5963970118]

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

N = 233
delta = 0.0000001
Q = np.diag(2.*np.ones(N), 0) + np.diag(-np.ones(N - 1), -1) + np.diag(-np.ones(N - 1), 1)
Q = delta*Q
P = np.linalg.inv(Q)
path_ts = np.linspace(-11.6, 0., N)

### Generate random paths
#########################################################################################


# Samples that meet the criteria
mean = np.interp(path_ts, ts, Ls)
samples = np.random.multivariate_normal(mean, P, 100000)
good_samples = []
for i in range(len(samples)):
    #plt.plot(samples[i])

    if i % 25 == 0:
        print(i)
   
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



