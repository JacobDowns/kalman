import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
import seaborn as sns
import matplotlib
matplotlib.rcParams.update({'font.size': 18})

fig = plt.figure(figsize=(12,7))
ax = fig.add_subplot(111)

current_palette =  sns.color_palette()#sns.color_palette("deep", 8)

ages = np.loadtxt('../transform_final/center3/sigma_ts.txt')
ages1 = np.loadtxt('../transform_final/center3/opt1/opt_age.txt')

L1 = np.loadtxt('../transform_final/center3/opt1/opt_L.txt')
L2 = np.loadtxt('../transform_final/south2/opt1/opt_L.txt')

p1 = np.loadtxt('../transform_final/center3/opt1/opt_m.txt')
p2 = np.loadtxt('../transform_final/south2/opt1/opt_m.txt')
p1h = np.loadtxt('../transform_final/center3_high/opt1/opt_m.txt')
p2h = np.loadtxt('../transform_final/south2_high/opt1/opt_m.txt')
p1l = np.loadtxt('../transform_final/center3_low/opt1/opt_m.txt')
p2l = np.loadtxt('../transform_final/south2_low/opt1/opt_m.txt') 
v1 = np.loadtxt('../transform_final/center3/opt1/v.txt')
v2 = np.loadtxt('../transform_final/south2/opt1/v.txt')
v1h = np.loadtxt('../transform_final/center3_high/opt1/v.txt')
v2h = np.loadtxt('../transform_final/south2_high/opt1/v.txt')
v1l = np.loadtxt('../transform_final/center3_low/opt1/v.txt')
v2l = np.loadtxt('../transform_final/south2_low/opt1/v.txt')

u1 = np.maximum(np.maximum(p1 + 2.*np.sqrt(v1), p1h + 2.*np.sqrt(v1h)), p1l + 2.*np.sqrt(v1l))
l1 = np.minimum(np.minimum(p1 - 2.*np.sqrt(v1), p1h - 2.*np.sqrt(v1h)), p1l - 2.*np.sqrt(v1l))
p1f = (1./3.)*(p1 + p1h + p1l)

u2 = np.maximum(np.maximum(p2 + 2.*np.sqrt(v2), p2h + 2.*np.sqrt(v2h)), p2l + 2.*np.sqrt(v2l))
l2 = np.minimum(np.minimum(p2 - 2.*np.sqrt(v2), p2h - 2.*np.sqrt(v2h)), p2l - 2.*np.sqrt(v2l))
p2f = (1./3.)*(p2 + p2h + p2l)

p1t = np.loadtxt('../transform_final/center3/opt1/opt_p.txt')/ L1
p2t = np.loadtxt('../transform_final/south2/opt1/opt_p.txt') / L2


plt.plot(ages, p1f, color = 'k', marker = 'o', lw = 3.5, ms=6)
plt.plot(ages, p1f, color = current_palette[0], marker = 'o', lw = 2.25, ms=5, label = 'North')
plt.fill_between(ages, l1, u1, color = 'gray', alpha = 0.75)
plt.plot(ages1, p1t, color = 'k', lw = 3.2, alpha = 0.8)
plt.plot(ages1, p1t, color = current_palette[0], lw = 2.5, alpha = 0.8)

plt.plot(ages, p2f, color = 'k', marker = 'o', lw = 3.5, ms=6)
plt.plot(ages, p2f, color = current_palette[3], marker = 'o', lw = 2.25, ms=5, label = 'South')
plt.fill_between(ages, l2, u2, color = 'gray', alpha =0.75)
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
