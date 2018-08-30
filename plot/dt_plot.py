import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
import seaborn as sns
import matplotlib
matplotlib.rcParams.update({'font.size': 16})


fig = plt.figure(figsize=(9,6))
ax = fig.add_subplot(111)

dt1 = np.loadtxt('transform/north1/opt2/opt_m.txt')
#v1 = np.loadtxt('filter/center_prior2/var.txt')
dt2 = np.loadtxt('transform/center1/opt2/opt_m.txt')
#v2 = np.loadtxt('filter/south_prior4/var.txt')
dt3 = np.loadtxt('transform/south1/opt2/opt_m.txt')
#v3 = np.loadtxt('filter/north_prior2/var.txt')
ages = np.loadtxt('transform/center1/sigma_ts.txt')

#obs_ages = np.array([-11.6, -10.2, -9.2, -8.2, -7.3])*1e3
#obs_Ls = np.array([406878.12855486432, 396313.20004890749, 321224.04532276397, 292845.40895793668, 288562.44342502725])

#np.savetxt('filter/3/Ls_smoothed.txt', Ls)

#ages / 1000.
plt.plot(ages, dt3, '#d62728', lw = 2, label = 'North')
#plt.plot(ages, dt3 - 2.0*np.sqrt(v3), '#d62728', lw = 2, linestyle = '--')
#plt.plot(ages, dt3 + 2.0*np.sqrt(v3), '#d62728', lw = 2, linestyle = '--')
plt.plot(ages, dt1, '#2ca02c', lw = 2, label = 'Center')
#plt.plot(ages, dt1 - 2.0*np.sqrt(v1), '#2ca02c', lw = 2, linestyle = '--')
#plt.plot(ages, dt1 + 2.0*np.sqrt(v1), '#2ca02c', lw = 2, linestyle = '--')
plt.plot(ages, dt2, '#1f77b4', lw = 2, label = 'South')
#plt.plot(ages, dt2 - 2.0*np.sqrt(v2), '#1f77b4', lw = 2, linestyle = '--')
#plt.plot(ages, dt2 + 2.0*np.sqrt(v2), '#1f77b4', lw = 2, linestyle = '--')
plt.legend()
plt.xlabel('Age (ka BP)')
plt.ylabel(r'$\Delta T$')
plt.xlim([ages.min(), ages.max()])
ax.set_xticks([-11600, -11000., -10000., -9000., -8000, -7300.])
ticks = ax.get_xticks()
#rint ticks
plt.grid(color='slategray', linestyle=':', linewidth=1)
ax.set_xticklabels([int(abs(tick / 1000.)) for tick in ticks])
plt.tight_layout()

plt.savefig('dts.png', dpi=700)