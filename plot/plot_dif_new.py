import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
import seaborn as sns
import matplotlib

Ls_center = np.array([406878.12855486432, 396313.20004890749, 321224.04532276397, 292845.40895793668, 288562.44342502725, 279753.70997966686])
Ls_south  = np.array([424777.2650658561, 394942.08036138373, 332430.9181651594, 303738.499327732, 296659.0156905292, 284686.5963970118])

age = np.loadtxt('transform_long/center2_new/opt1/opt_age.txt')
L1 = np.loadtxt('transform_long/center3_minimal/opt2/opt_L.txt')
L2 = np.loadtxt('transform_long/south3_minimal/opt2/opt_L.txt')

y_age = np.loadtxt('paleo_inputs/y_ages.txt')
y_c = np.loadtxt('paleo_inputs/y_c.txt')
y_s = np.loadtxt('paleo_inputs/y_s.txt')
Py_c = np.loadtxt('paleo_inputs/Py_c.txt')
Py_s = np.loadtxt('paleo_inputs/Py_s.txt')

v1 = 2.0*np.sqrt(Py_c[range(len(Py_c)), range(len(Py_s))])
v2 = 2.0*np.sqrt(Py_c[range(len(Py_c)), range(len(Py_s))])

"""
for j in [1,2,3,4]:
    index = np.where(L1 > Ls_south[j])[0].max()
    time = age[index]
    print(time)
"""

plt.subplot(2,1,1)
plt.plot(age, L1, 'r')
plt.plot(y_age, y_c, 'k')
plt.plot(y_age, y_c - v1, 'k')
plt.plot(y_age, y_c + v1, 'k')
plt.plot(age, Ls_center[0]*np.ones(len(age)), 'b')
plt.plot(age, Ls_center[1]*np.ones(len(age)), 'b')
plt.plot(age, Ls_center[2]*np.ones(len(age)), 'b')
plt.plot(age, Ls_center[3]*np.ones(len(age)), 'b')
plt.plot(age, Ls_center[4]*np.ones(len(age)), 'b')


plt.subplot(2,1,2)
plt.plot(age, L2, 'r')
plt.plot(y_age, y_s, 'k')
plt.plot(y_age, y_s - v2, 'k')
plt.plot(y_age, y_s + v2, 'k')

plt.show()



