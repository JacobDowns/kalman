import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
import seaborn as sns
import matplotlib
matplotlib.rcParams.update({'font.size': 16})

fig = plt.figure(figsize=(9,5))
ax = fig.add_subplot(111)

current_palette =  sns.color_palette()#sns.color_palette("deep", 8)

ages = np.loadtxt('transform/center1/sigma_ts.txt')
ages1 = np.loadtxt('transform/center2_new/opt1/opt_age.txt')

L1 = np.loadtxt('transform/center2_new/opt1/opt_L.txt')
L2 = np.loadtxt('transform/south2_new/opt1/opt_L.txt')

p1 = np.loadtxt('transform/center2_new/opt1/opt_m.txt')
p2 = np.loadtxt('transform/south2_new/opt1/opt_m.txt')

p1t = np.loadtxt('transform/center2_new/opt1/opt_p.txt')/ L1
p2t = np.loadtxt('transform/south2_new/opt1/opt_p.txt') / L2

#plt.plot(ages, p1, color = current_palette[0], marker = 'o', lw = 2, ms=5, label = 'North')
plt.plot(ages1, p1t, color = current_palette[0], lw = 3, alpha = 0.9, label = 'North Flowline')
#plt.plot(ages, p2, color = current_palette[3], marker = 'o', lw = 2, ms=5, label = 'South')
plt.plot(ages1, p2t, color = current_palette[3], lw = 3, alpha = 0.9, label = 'South Flowline')
plt.grid(color='slategray', linestyle=':', linewidth=1)

plt.xlabel('Age (ka BP)')
plt.ylabel(r'Avg. Precip. (m.w.e. a$^{-1}$)')
plt.xlim([ages.min(), -7000.])
#plt.plot(ages, p2, 'r')
ax.set_xticks([-11600, -10000., -8000, -7000.])
ticks = ax.get_xticks()
#rint ticks
plt.grid(color='slategray', linestyle=':', linewidth=1)
labels = [int(abs(tick / 1000.)) for tick in ticks]
labels[0] = '11.6'
plt.legend()
ax.set_xticklabels(labels)
plt.tight_layout()
plt.savefig('deltap_poster.png', dpi = 500)
plt.show()
