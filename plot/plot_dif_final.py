import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
import seaborn as sns
import matplotlib

Ls_center = np.array([406878.12855486432, 396313.20004890749, 321224.04532276397, 292845.40895793668, 288562.44342502725, 279753.70997966686])
Ls_south  = np.array([424777.2650658561, 394942.08036138373, 332430.9181651594, 303738.499327732, 296659.0156905292, 284686.5963970118])

age = np.loadtxt('transform/center2/opt1/opt_age.txt')
L = np.loadtxt('transform/center2/opt1/opt_L.txt')
v = np.loadtxt('transform/center2/opt1/y_v.txt')
#L1 = np.loadtxt('transform/center2/opt1/opt_L.txt')
meas_indexes = range(0, len(age), 25*3)


y_age = np.loadtxt('paleo_inputs/y_ages.txt')
y_c = np.loadtxt('paleo_inputs/y_c1.txt')
y_s = np.loadtxt('paleo_inputs/y_s1.txt')
Py_c = np.loadtxt('paleo_inputs/Py_c1.txt')
Py_s = np.loadtxt('paleo_inputs/Py_s1.txt')

plt.subplot(2,1,1)
plt.plot(age, L, 'k')
plt.plot(age[meas_indexes], L[meas_indexes]-v, 'k--')
plt.plot(age[meas_indexes], L[meas_indexes]+v, 'k--')
#plt.plot(y_age, y_c, 'k')
#plt.plot(y_age, y_c - v1, 'k')
#plt.plot(y_age, y_c + v1, 'k')

for i in [0,1,2,3,4]:
    index = np.where(L > Ls_center[i])[0].max()
    time = age[index]
    print(time)


obs_ts = np.array([-11686.0, -10416.0, -9156.0, -8196.0, -7366.0, 0.])
# Observation variances
obs_sigmas = np.array([0.4, 0.2, 0.2, 0.3, 0.3, 0.1])*1000. / 2.

for i in range(len(obs_ts)):
    plt.plot(obs_ts[i], Ls_center[i], 'ro')
    plt.plot([obs_ts[i] - obs_sigmas[i], obs_ts[i] + obs_sigmas[i]], [Ls_center[i], Ls_center[i]], 'ko-')
"""
plt.subplot(2,1,2)
plt.plot(age, L2, 'r')
plt.plot(y_age, y_s, 'k')
plt.plot(y_age, y_s - v2, 'k')
plt.plot(y_age, y_s + v2, 'k')
"""
plt.show()



