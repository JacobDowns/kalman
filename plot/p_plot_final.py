import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
import seaborn as sns
import matplotlib
matplotlib.rcParams.update({'font.size': 18})

fig = plt.figure(figsize=(12,7))
ax = fig.add_subplot(111)

current_palette =  sns.color_palette()#sns.color_palette("deep", 8)

ages = np.loadtxt('../transform_final/center1/sigma_ts.txt')
ages1 = np.loadtxt('../transform_final/center2/opt1/opt_age.txt')

L1 = np.loadtxt('../transform_final/center2/opt1/opt_L.txt')
L2 = np.loadtxt('../transform_final/south2/opt1/opt_L.txt')

p1 = np.loadtxt('../transform_final/center2/opt1/opt_m.txt')
p2 = np.loadtxt('../transform_final/south2/opt1/opt_m.txt')
p3 = np.loadtxt('../transform_final/center3/opt1/opt_m.txt')

plt.plot(p1, 'k')
plt.plot(p3, 'r')
plt.show()


quit()

v1 = np.loadtxt('../transform_final/center2/opt1/v.txt')
v2 = np.loadtxt('../transform_final/south2/opt1/v.txt')

p1t = np.loadtxt('../transform_final/center2/opt1/opt_p.txt')/ L1
p2t = np.loadtxt('../transform_final/south2/opt1/opt_p.txt') / L2


plt.plot(ages, p1, color = 'k', marker = 'o', lw = 4.5, ms=10)
plt.plot(ages, p1, color = current_palette[0], marker = 'o', lw = 3.5, ms=8, label = 'North')
#plt.fill_between(ages, p1 - 2.0*np.sqrt(v1), p1 + 2.0*np.sqrt(v1), color = 'k', alpha = 0.5)
plt.plot(ages1, p1t, color = 'k', lw = 3.2, alpha = 0.8)
plt.plot(ages1, p1t, color = current_palette[0], lw = 2.5, alpha = 0.8)

plt.plot(ages, p2, color = 'k', marker = 'o', lw = 4.5, ms=10)
plt.plot(ages, p2, color = current_palette[3], marker = 'o', lw = 3.2, ms=8, label = 'South')
#plt.fill_between(ages, p2 - 2.0*np.sqrt(v2), p2 + 2.0*np.sqrt(v2), color = 'k', alpha = 0.5)"""

plt.plot(ages1, p2t, color = 'k', lw = 3, alpha = 0.8)
plt.plot(ages1, p2t, color = current_palette[3], lw = 2.5, alpha = 0.8)
plt.grid(color='slategray', linestyle=':', linewidth=1)

plt.xlabel('Age (ka BP)')
plt.ylabel(r'$\Delta P$ (m.w.e. a$^{-1}$)')
plt.xlim([ages.min(), ages.max()])
#plt.plot(ages, p2, 'r')

ax.set_xticks([-11500, -10000., -8000, -6000.,-4000., -2000., 0.])
ticks = ax.get_xticks()
#rint ticks
plt.grid(color='slategray', linestyle=':', linewidth=1)
labels = [int(abs(tick / 1000.)) for tick in ticks]
labels[0] = '11.5'
plt.legend()
ax.set_xticklabels(labels)
plt.tight_layout()
plt.savefig('images/deltap_final.png', dpi = 500)
plt.show()
