import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
import seaborn as sns
import matplotlib
matplotlib.rcParams.update({'font.size': 16})

fig = plt.figure(figsize=(9,6))
ax = fig.add_subplot(111)

# Default params
p1a = np.loadtxt('transform_long/south2_seasonal/opt1/opt_m.txt')
L1 = np.loadtxt('transform_long/south2_seasonal/opt1/opt_L.txt')
p1 = np.loadtxt('transform_long/south2_seasonal/opt1/opt_p.txt') / L1
v1 = np.loadtxt('transform_long/south2_seasonal/opt1/v.txt')

# beta2 0
p2a = np.loadtxt('transform_long/center2_seasonal/opt1/opt_m.txt')
L2 = np.loadtxt('transform_long/center2_seasonal/opt1/opt_L.txt')
p2 = np.loadtxt('transform_long/center2_seasonal/opt1/opt_p.txt') / L2
v2 = np.loadtxt('transform_long/center2_seasonal/opt1/v.txt')

"""
# beta2 1
p3a = np.loadtxt('transform_long/beta2_1/opt1/opt_m.txt')
L3 = np.loadtxt('transform_long/beta2_1/opt1/opt_L.txt')
p3 = np.loadtxt('transform_long/beta2_1/opt1/opt_p.txt') / L3
v3 = np.loadtxt('transform_long/beta2_1/opt1/v.txt')

# P_frac 0
p4a = np.loadtxt('transform_long/P_frac_0/opt1/opt_m.txt')
L4 = np.loadtxt('transform_long/P_frac_0/opt1/opt_L.txt')
p4 = np.loadtxt('transform_long/P_frac_0/opt1/opt_p.txt') / L4
v4 = np.loadtxt('transform_long/P_frac_0/opt1/v.txt')

# P_frac 1
p5a = np.loadtxt('transform_long/P_frac_1/opt1/opt_m.txt')
L5 = np.loadtxt('transform_long/P_frac_1/opt1/opt_L.txt')
p5 = np.loadtxt('transform_long/P_frac_1/opt1/opt_p.txt') / L5
v5 = np.loadtxt('transform_long/P_frac_1/opt1/v.txt')"""

tsa = np.loadtxt('transform_long/center3/sigma_ts.txt')
ts = np.loadtxt('transform_long/center3/opt1/opt_age.txt')

"""
p1_interp = interp1d(ts, p1, kind = 'linear')
p2_interp = interp1d(ts, p2, kind = 'linear')
p3_interp = interp1d(ts, p3, kind = 'linear')

ts_fine = np.loadtxt('transform_long/center3/age_0.txt')

p1_fine = p1_interp(ts_fine)
p2_fine = p2_interp(ts_fine)
p3_fine = p3_interp(ts_fine)"""

#obs_ages = np.array([-11.6, -10.2, -9.2, -8.2, -7.3])*1e3
#obs_Ls = np.array([406878.12855486432, 396313.20004890749, 321224.04532276397, 292845.40895793668, 288562.44342502725])

#np.savetxt('filter/3/Ls_smoothed.txt', Ls)

#plt.plot(tsa, p1a, 'k', lw = 2.5, label = 'Default', marker = 'o', ms = 5, linestyle = '-')
plt.plot(ts, p1, '#d62728', lw = 2.5, ms = 10, alpha = 0.7, linestyle = '--',  dashes=(1, 0))
plt.plot(tsa, p1a, '#1f77b4', lw = 2.5, marker = 'o', ms = 5, linestyle = '-')
#plt.plot(tsa, p3a, '#1f77b4', lw = 2.5, label = r'$\beta^2 = 1.25 \times 10^{-3}$', marker = 'o', ms = 5, linestyle = '--')
plt.plot(ts, p2,  '#1f77b4', lw = 2.5, ms = 10, alpha = 0.7, linestyle = '--', dashes=(1, 0))
plt.plot(tsa, p2a, '#d62728', lw = 2.5, marker = 'o', ms = 5, linestyle = '-')
#plt.plot(tsa, p5a, '#d62728', lw = 2.5, label = r'$\beta^2 = 1.25 \times 10^{-3}$', marker = 'o', ms = 5, linestyle = '--')


#plt.plot(ages, dtb, color = 'k', lw = 1.5, label = 'Buizert Dye-3', marker = 'o', ms = 2)
plt.legend()
plt.xlabel('Age (ka BP)')
plt.ylabel(r'$\Delta P$ (m.w.e. a$^{-1}$)')
plt.xlim([ts.min(), ts.max()])
#ax.set_xticks([-11600, -11000., -10000., -9000., -8000, -7000., -6000., -5000., -4000., -3000., -2000., -1000., 0.])
ticks = ax.get_xticks()
#rint ticks
plt.grid(color='slategray', linestyle=':', linewidth=1)
labels = [int(abs(tick / 1000.)) for tick in ticks]
labels[0] = '11.6'
ax.set_xticklabels(labels)
plt.tight_layout()
#plt.show()
plt.savefig('seasonal_precip.png', dpi=700)
