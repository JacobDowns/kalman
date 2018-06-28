import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

ages = np.loadtxt('filter/3/ages.txt')
Ls = np.loadtxt('filter/3/Ls.txt')
Ls_smoothed = np.loadtxt('filter/3/Ls_smoothed.txt')

ages_data = np.array([-11.6, -10.2, -9.2, -8.2, -7.3])*1e3
Ls_data = [406878.12855486432, 396313.20004890749, 321224.04532276397, 292845.40895793668, 288562.44342502725]
L_interp = interp1d(ages_data, Ls_data, kind = 'linear')

#np.savetxt('filter/3/Ls_smoothed.txt', Ls)
plt.plot(ages, Ls)
plt.plot(ages, Ls_smoothed)
plt.plot(ages, L_interp(ages))
plt.show()
