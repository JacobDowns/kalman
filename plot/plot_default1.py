import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
import matplotlib
import seaborn as sns

matplotlib.rcParams.update({'font.size': 16})
current_palette = sns.color_palette("RdBu_r", 12)
fig = plt.figure(figsize=(12,16))

data = np.loadtxt('paleo_data/buizert_full.txt')
years = -data[:,0][::-1]
temps_ann = data[:,1][::-1]
temps_djf = data[:,2][::-1]
temps_mam = data[:,3][::-1]
temps_jja = data[:,4][::-1]
temps_son = data[:,5][::-1]

#self.delta_temp_ann = interp1d(years, temps_ann - temps_ann[-1], kind = 'linear')
delta_temp_djf = interp1d(years, temps_djf - temps_djf[-1], kind = 'linear')
delta_temp_mam = interp1d(years, temps_mam - temps_mam[-1], kind = 'linear' )
delta_temp_jja = interp1d(years, temps_jja - temps_jja[-1], kind = 'linear' )
delta_temp_son = interp1d(years, temps_son - temps_son[-1], kind = 'linear')


ax = plt.subplot(3,1,1)
plt.title('(a)')
ts = np.linspace(-11.6e3, -7.3e3, 750)
plt.plot(ts, delta_temp_djf(ts), color = current_palette[0], lw = 2.5, label = 'DJF')
plt.plot(ts, delta_temp_mam(ts), color = current_palette[-3], lw = 2.5, label = 'MAM')
plt.plot(ts, delta_temp_jja(ts), color = current_palette[-1], lw = 2.5, label = 'JJA')
plt.plot(ts, delta_temp_son(ts), color = current_palette[2], lw = 2.5, label = 'SON')
plt.grid(color='lightgray', linestyle=':', linewidth=2.5)
plt.xticks([-11e3, -10e3, -9e3, -8e3, -7e3])
plt.xlim([ts.min(), ts.max()])
plt.ylabel(r'$\Delta T$ ($^{\circ{}}$ C)')
ticks = ax.get_xticks()
ax.set_xticklabels([int(abs(tick / 1000.)) for tick in ticks])
plt.grid(True)
plt.legend(loc = 4)


### Terminus positon
#################################################################
ages = np.loadtxt('paleo_runs/center_seasonal/age.txt')
Ls1 = np.loadtxt('paleo_runs/center_seasonal/L.txt') / 1000.
Ls2 = np.loadtxt('paleo_runs/south_seasonal/L.txt') / 1000.

obs_ages = np.array([-11.6, -10.2, -9.2, -8.2, -7.3, 0.0])*1e3
obs_Ls1 = np.array([406878, 396313, 321224, 292845, 288562, 279753]) / 1000.
obs_Ls2 = np.array([424777, 394942, 332430, 303738, 296659, 284686]) / 1000. 

meas_ages = np.loadtxt('paleo_inputs/measure_ages.txt')

y_center = np.loadtxt('paleo_inputs/y_center.txt') / 1e3
Py_center = np.loadtxt('paleo_inputs/Py_center.txt')
sd_center = np.sqrt(Py_center[range(len(Py_center)), range(len(Py_center))]) / 1e3

y_south = np.loadtxt('paleo_inputs/y_south.txt') / 1e3
Py_south = np.loadtxt('paleo_inputs/Py_south.txt')
sd_south = np.sqrt(Py_south[range(len(Py_south)), range(len(Py_south))]) / 1e3

obs_indexes = [0, 28, 48, 68, 86, -1]


ax = plt.subplot(3,1,2)
plt.title('(b)')
plt.grid(color='lightgray', linestyle=':', linewidth=2.5)
plt.plot(ages, Ls1, 'r-', lw = 3)
plt.plot(meas_ages, y_center, 'k', lw = 3, ms = 8)
plt.xlim([ages.min(), ages.max()])
plt.grid(color='lightgray', linestyle=':', linewidth=2.5)
plt.xticks([-11e3, -10e3, -9e3, -8e3, -7e3])
plt.xlim([ages.min(), ages.max()])
ticks = ax.get_xticks()
ax.set_xticklabels([int(abs(tick / 1000.)) for tick in ticks])

plt.fill_between(meas_ages, y_center - 2.0*sd_center, y_center + 2.0*sd_center, facecolor='gray', alpha = 0.5)
#plt.fill_between(meas_ages, y_center - sd_center    , y_center + sd_center    , facecolor='gray', alpha = 0.5)

ax = plt.subplot(3,1,3)
plt.title('(c)')
plt.plot(ages, Ls2, 'r-', lw = 3)
plt.plot(meas_ages, y_south, 'k', lw = 3, ms = 8)
plt.xlim([ages.min(), ages.max()])
plt.grid(color='lightgray', linestyle=':', linewidth=2.5)
plt.xticks([-11e3, -10e3, -9e3, -8e3, -7e3])
plt.xlim([ages.min(), ages.max()])
plt.ylabel('Glacier Length (km)')
ticks = ax.get_xticks()
ax.set_xticklabels([int(abs(tick / 1000.)) for tick in ticks])
plt.fill_between(meas_ages, y_south - 2.0*sd_south, y_south + 2.0*sd_south, facecolor='gray', alpha = 0.5)
#plt.fill_between(meas_ages, y_south - sd_south    , y_south + sd_south    , facecolor='gray', alpha = 0.5)
plt.xlabel('Age (ka BP)')

plt.tight_layout()
plt.savefig('seasonal_default.png', dpi=550)
