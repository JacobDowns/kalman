import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
import seaborn as sns
import matplotlib
matplotlib.rcParams.update({'font.size': 16})


fig = plt.figure(figsize=(9,6))
ax = fig.add_subplot(111)

#ages = np.loadtxt('transform_long/center1/opt1/opt_age.txt')
p1 = np.loadtxt('transform_long/center1/opt1/opt_m.txt')
p2 = np.loadtxt('transform_long/center2/opt1/opt_m.txt')

plt.plot(p1, 'r')
plt.plot(p2, 'g')
plt.show()
quit()
#v1 = np.loadtxt('transform/north2/opt2/v.txt')
#p2 = np.loadtxt('transform_long/beta2_0/opt1/opt_m.txt')
#p3 = np.loadtxt('transform_long/beta2_1/opt1/opt_m.txt')
#v2 = np.loadtxt('transform/center2/opt2/v.txt')
#p3 = np.loadtxt('transform/south2/opt2/opt_m.txt')
#v3 = np.loadtxt('transform/south2/opt2/v.txt')
#ts = np.loadtxt('transform/center1/sigma_ts.txt')

plt.plot(p1, 'r')
plt.plot(p2, 'g')
#plt.plot(ages, p3, 'b')
plt.show()


quit()
"""
p1_interp = interp1d(ts, p1, kind = 'linear')
p2_interp = interp1d(ts, p2, kind = 'linear')
p3_interp = interp1d(ts, p3, kind = 'linear')

ts_fine = np.loadtxt('transform/center2/age_0.txt')
Ls1 = np.loadtxt('transform/north2/opt1/opt_L.txt')
Ls2 = np.loadtxt('transform/center2/opt1/opt_L.txt')
Ls3 = np.loadtxt('transform/south2/opt1/opt_L.txt')

p1_fine = p1_interp(ts_fine)
p2_fine = p2_interp(ts_fine)
p3_fine = p3_interp(ts_fine)
"""

#obs_ages = np.array([-11.6, -10.2, -9.2, -8.2, -7.3])*1e3
#obs_Ls = np.array([406878.12855486432, 396313.20004890749, 321224.04532276397, 292845.40895793668, 288562.44342502725])

#np.savetxt('filter/3/Ls_smoothed.txt', Ls)

#ages / 1000.
plt.plot(ts, p3, '#1f77b4', lw = 2.5, label = 'South', marker = 'o', ms = 10)
plt.plot(ts, p3 - 2.0*np.sqrt(v3), '#1f77b4', lw = 2.5, linestyle = '--')
plt.plot(ts, p3 + 2.0*np.sqrt(v3), '#1f77b4', lw = 2.5, linestyle = '--')
plt.plot(ts, p2, '#2ca02c', lw = 2.5, label = 'Center', marker = 'o', ms = 10)
plt.plot(ts, p2 - 2.0*np.sqrt(v2), '#2ca02c', lw = 2.5, linestyle = '--')
plt.plot(ts, p2 + 2.0*np.sqrt(v2), '#2ca02c', lw = 2.5, linestyle = '--')
plt.plot(ts, p1, '#d62728', lw = 2.5, label = 'North', marker = 'o', ms = 10)
plt.plot(ts, p1 - 2.0*np.sqrt(v1), '#d62728', lw = 2.5, linestyle = '--')
plt.plot(ts, p1 + 2.0*np.sqrt(v1), '#d62728', lw = 2.5, linestyle = '--')


#plt.plot(ages, dtb, color = 'k', lw = 1.5, label = 'Buizert Dye-3', marker = 'o', ms = 2)
plt.legend()
plt.xlabel('Age (ka BP)')
plt.ylabel(r'Avg. Add. Precip. m.w.e')
plt.xlim([ts.min(), ts.max()])
ax.set_xticks([-11600, -11000., -10000., -9000., -8000, -7300.])
ticks = ax.get_xticks()
#rint ticks
plt.grid(color='slategray', linestyle=':', linewidth=1)
ax.set_xticklabels([int(abs(tick / 1000.)) for tick in ticks])
plt.tight_layout()
plt.show()
#plt.savefig('dts.png', dpi=700)
