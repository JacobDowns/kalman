import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
import seaborn as sns
import matplotlib
matplotlib.rcParams.update({'font.size': 18})

fig = plt.figure(figsize=(9,6))
ax = fig.add_subplot(111)

# Default params

p1a = np.loadtxt('transform_long/north2_seasonal/opt1/opt_m.txt')
L1 = np.loadtxt('transform_long/north2_seasonal/opt1/opt_L.txt')
p1 = np.loadtxt('transform_long/north2_seasonal/opt1/opt_p.txt') / L1
v1 = np.loadtxt('transform_long/north2_seasonal/opt1/v.txt')

p2a = np.loadtxt('transform_long/center2_seasonal/opt1/opt_m.txt')
p2a0 = np.loadtxt('transform_long/center1_seasonal/opt1/opt_m.txt')
L2 = np.loadtxt('transform_long/center2_seasonal/opt1/opt_L.txt')
p2 = np.loadtxt('transform_long/center2_seasonal/opt1/opt_p.txt') / L2
v2 = np.loadtxt('transform_long/center2_seasonal/opt1/v.txt')
prior_m = np.loadtxt('transform_long/center1_seasonal/prior_m.txt')
prior_ts = np.loadtxt('transform_long/center1_seasonal/sigma_ts.txt')

p3a = np.loadtxt('transform_long/south2_seasonal/opt1/opt_m.txt')
p3a0 = np.loadtxt('transform_long/south1_seasonal/opt1/opt_m.txt')
L3 = np.loadtxt('transform_long/south2_seasonal/opt1/opt_L.txt')
p3 = np.loadtxt('transform_long/south2_seasonal/opt1/opt_p.txt') / L3
v3 = np.loadtxt('transform_long/south2_seasonal/opt1/v.txt')

tsa = np.loadtxt('transform_long/center2_seasonal/sigma_ts.txt')
ts = np.loadtxt('transform_long/center2_seasonal/opt1/opt_age.txt')

current_palette = sns.color_palette()
#sns.palplot(current_palette)
#plt.show()
#quit()


#obs_ages = np.array([-11.6, -10.2, -9.2, -8.2, -7.3])*1e3
#obs_Ls = np.array([406878.12855486432, 396313.20004890749, 321224.04532276397, 292845.40895793668, 288562.44342502725])

#np.savetxt('filter/3/Ls_smoothed.txt', Ls)

#plt.plot(tsa, p1a, 'k', lw = 2.5, label = 'Default', marker = 'o', ms = 5, linestyle = '-')
#plt.plot(ts, p1, color = current_palette[0], lw = 2.5, ms = 10, alpha = 0.7, linestyle = '--',  dashes=(1, 0))
#plt.plot(tsa, p1a, color = current_palette[0], lw = 2.5, marker = 'o', ms = 5, linestyle = '-')
plt.plot(prior_ts, prior_m, color = 'gray', lw = 3, marker = 'o', ms = 6, linestyle = '-', label = 'Prior')
#plt.plot(tsa, p2a0,  color = current_palette[0], lw = 3, ms = 10, alpha = 0.75, linestyle = '--', dashes=(1, 0))
plt.plot(ts, p2, color = current_palette[0], lw = 3, ms = 10, alpha = 0.8, linestyle = '--', dashes=(1, 0))
plt.plot(tsa, p2a, color = current_palette[0], lw = 3, marker = 'o', ms = 6, linestyle = '-', label = 'North')
#plt.plot(tsa, p3a0, color = current_palette[3], lw = 3, ms = 10, alpha = 0.75, linestyle = '--', dashes=(1, 0))
plt.plot(ts, p3,  color = current_palette[3], lw = 3, ms = 10, alpha = 0.8, linestyle = '--', dashes=(1, 0))
plt.plot(tsa, p3a, color = current_palette[3], lw = 3, marker = 'o', ms = 6, linestyle = '-', label = 'South')


plt.legend(loc = 3)
plt.xlabel('Age (ka BP)')
plt.ylabel(r'$\Delta P$ (m.w.e. a$^{-1}$)')
plt.xlim([ts.min(), ts.max()])
ticks = ax.get_xticks()
plt.grid(color='slategray', linestyle=':', linewidth=1)
labels = [int(abs(tick / 1000.)) for tick in ticks]
labels[0] = '11.6'
ax.set_xticklabels(labels)
plt.tight_layout()
plt.savefig('p_plot_seasonal.png', dpi = 700)
