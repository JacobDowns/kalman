import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

dt1 = np.loadtxt('filter/center_prior2/opt_m.txt')
dt2 = np.loadtxt('filter/south_prior4/opt_m.txt')
#ages = np.loadtxt('filter/center_prior2/ages_0.txt')

#obs_ages = np.array([-11.6, -10.2, -9.2, -8.2, -7.3])*1e3
#obs_Ls = np.array([406878.12855486432, 396313.20004890749, 321224.04532276397, 292845.40895793668, 288562.44342502725])

#np.savetxt('filter/3/Ls_smoothed.txt', Ls)
plt.plot(dt1, 'k')
plt.plot(dt2, 'r')
#plt.plot(ages, Ls1, 'b')a
#plt.plot(ages, Ls2, 'g')
#plt.plot(obs_ages, obs_Ls, 'ko-')
#plt.plot(ages, L_interp(ages))
plt.show()
