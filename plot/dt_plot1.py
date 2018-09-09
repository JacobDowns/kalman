import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
import seaborn as sns
import matplotlib
matplotlib.rcParams.update({'font.size': 16})

#fig = plt.figure(figsize=(9,6))
#ax = fig.add_subplot(111)

dt1 = np.loadtxt('transform/center1/opt2/opt_m.txt')
dt2 = np.loadtxt('transform/center1_new/opt1/opt_m.txt')
dt3 = np.loadtxt('transform/center2_new/opt1/opt_m.txt')
dt4 = np.loadtxt('transform/center2_new/opt2/opt_m.txt')
ages = np.loadtxt('transform/center2/sigma_ts.txt')


"""
v1 = np.loadtxt('transform/north2/opt1/v.txt')
dt2 = np.loadtxt('transform/center2/opt1/opt_m.txt')
v2 = np.loadtxt('transform/center2/opt1/v.txt')
dt3 = np.loadtxt('transform/south2/opt1/opt_m.txt')
v3 = np.loadtxt('transform/south2/opt1/v.txt')

"""
# Load Buizert

data = np.loadtxt('paleo_data/buizert_dye3.txt')
years = -data[:,0][::-1]
temps = data[:,1][::-1]
delta_temp_interp = interp1d(years, temps - temps[-1], kind = 'linear')
dtb = delta_temp_interp(ages)
print dtb

#obs_ages = np.array([-11.6, -10.2, -9.2, -8.2, -7.3])*1e3
#obs_Ls = np.array([406878.12855486432, 396313.20004890749, 321224.04532276397, 292845.40895793668, 288562.44342502725])

#np.savetxt('filter/3/Ls_smoothed.txt', Ls)

#ages / 1000.
plt.plot(ages, dt1, '#d62728', lw = 1.5, label = '1', marker = 'o', ms = 2)
plt.plot(ages, dt2, '#2ca02c', lw = 1.5, label = '2', marker = 'o', ms = 2)
plt.plot(ages, dt3, color = 'k', lw = 1.5, label = '3', marker = 'o', ms = 2)
plt.plot(ages, dt4, color = 'y', lw = 1.5, label = '3', marker = 'o', ms = 2)
plt.legend()
plt.xlabel('Age (ka BP)')
plt.ylabel(r'$\Delta T$')
plt.xlim([ages.min(), ages.max()])
#ax.set_xticks([-11600, -11000., -10000., -9000., -8000, -7300.])
#ticks = ax.get_xticks()
plt.grid(color='slategray', linestyle=':', linewidth=1)
#ax.set_xticklabels([int(abs(tick / 1000.)) for tick in ticks])
plt.tight_layout()
plt.show()