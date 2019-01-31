import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
import seaborn as sns
import matplotlib
matplotlib.rcParams.update({'font.size': 16})


y = np.loadtxt('../paleo_inputs/y_c.txt')
y_ages = np.loadtxt('../paleo_inputs/y_ages.txt')

Pyy = np.loadtxt('../paleo_inputs/Py_c.txt')
v = np.sqrt(np.diag(Pyy))

Y = np.loadtxt('../transform_start/center2_random/Y.txt')
ages = np.loadtxt('../transform_start/center2_random/age_0.txt')


for i in range(len(Y)):
    plt.plot(ages, Y[i])

plt.plot(y_ages, y, 'k', linewidth = 2)
plt.plot(y_ages, y - 2.*v, 'k', linewidth = 2)
plt.plot(y_ages, y + 2.*v, 'k', linewidth = 2)

plt.show()



