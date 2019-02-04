import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
import seaborn as sns
import matplotlib
matplotlib.rcParams.update({'font.size': 16})

ts = np.loadtxt('../transform_start/center2_1/sigma_ts.txt')
m1 = np.loadtxt('../transform_start/center2_1/opt1/opt_m.txt')
P1 = np.loadtxt('../transform_start/center2_1/opt1/opt_P.txt')
v1 = np.sqrt(np.diag(P1))
m2 = np.loadtxt('../transform_start/center2_high1/opt1/opt_m.txt')
P2 = np.loadtxt('../transform_start/center2_high1/opt1/opt_P.txt')
v2 = np.sqrt(np.diag(P2))

xs = np.linspace(0., 1., 2)
ys = np.eye(2)
w_funcs = [] 
f1 = interp1d(xs, ys[0])
f2 = interp1d(xs, ys[1])
#f3 = interp1d(xs, ys[2])

num_samps = 500
weights = np.random.rand(num_samps)

w1 = f1(weights)
w2 = f2(weights)
#w3 = f3(weights)

s1 = np.random.multivariate_normal(m1, P1, num_samps)
s2 = np.random.multivariate_normal(m2, P2, num_samps)
#s3 = np.random.multivariate_normal(m3, P3, num_samps)

samps = (s1.T*w1).T + (s2.T*w2).T
#for i in range(num_samps):
#    plt.plot(samps[i])

P = np.cov(samps.T)
m = np.mean(samps, axis = 0)
v = np.sqrt(np.diag(P))

plt.plot(m, 'k', linewidth = 3)
plt.plot(m - 2.*v, 'k', linewidth = 3)
plt.plot(m + 2.*v, 'k', linewidth = 3)
plt.plot(m1, 'r')
plt.plot(m1 - 2.*v1, 'r')
plt.plot(m1 + 2.*v1, 'r')
plt.plot(m2, 'b')
plt.plot(m2 - 2.*v2, 'b')
plt.plot(m2 + 2.*v2, 'b')
plt.show()

np.savetxt('../transform_start/center2/opt/opt_m.txt', m)
np.savetxt('../transform_start/center2/opt/opt_P.txt', P)
quit()

    
xs  = np.array([0.,1./2.,0.])
ys1 = np.array([1.,0.,0.])
ys2 = np.array([0.,1.,0.])
ys3 = np.array([0.,0.,1.])
#x1 = interp1d([0., 
samples = np.random.multivariate_normal(x, Pxx, 500)


sd1 = np.sqrt(v1)
sd2 = np.sqrt(v2)
sd3 = np.sqrt(v3)

u1 = m1 + 2.0*sd1
u2 = m2 + 2.0*sd2
u3 = m3 + 2.0*sd3

l1 = m1 - 2.0*sd1
l2 = m2 - 2.0*sd2
l3 = m3 - 2.0*sd3

l = np.minimum(l3, np.minimum(l1, l2))
u = np.maximum(u3, np.maximum(u1, u2))
m = (1./3.)*(m1 + m2 + m3)

plt.plot(m, 'k')
plt.plot(l, 'k--')
plt.plot(u, 'k--')
plt.plot(x + 2.*vx, 'r')
plt.plot(x - 2.*vx, 'r')
plt.plot(l3, 'b')
plt.plot(u3, 'b')

plt.show()

quit()

