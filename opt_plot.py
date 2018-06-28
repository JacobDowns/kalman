import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

# Model
ages = np.loadtxt('opt_ages.txt')
Ls = np.loadtxt('opt_L.txt')

# Observations
obs_ages = np.array([-11.6, -10.2, -9.2, -8.2, -7.3])*1e3
obs_Ls = [406878.12855486432, 396313.20004890749, 321224.04532276397, 292845.40895793668, 288562.44342502725]
L_interp = interp1d(obs_ages, obs_Ls, kind = 'linear')


plt.plot(ages, Ls, 'r')
plt.plot(ages, L_interp(ages), 'k')
plt.show()







