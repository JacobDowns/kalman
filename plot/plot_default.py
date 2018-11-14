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
ages = np.loadtxt('paleo_runs/north_seasonal/age.txt')
Ls1 = np.loadtxt('paleo_runs/north_seasonal/L.txt')  / 1000.
Ls2 = np.loadtxt('paleo_runs/center_seasonal/L.txt') / 1000.
Ls3 = np.loadtxt('paleo_runs/south_seasonal/L.txt') / 1000.

#P = np.loadtxt('paleo_runs/north_seasonal/P.txt') / Ls1
#plt.plot(P)
#plt.show()
#quit()

obs_ages = np.array([-11.6, -10.2, -9.2, -8.2, -7.3, 0.0])*1e3
obs_Ls1 = np.array([470376, 442567, 351952, 313633, 307167, 300896]) / 1000. 
obs_Ls2 = np.array([406878, 396313, 321224, 292845, 288562, 279753]) / 1000.
obs_Ls3 = np.array([424777, 394942, 332430, 303738, 296659, 284686]) / 1000. 

ages_fine = np.linspace(-11.6, 0.0, 1000)*1e3
min_err = 5.**2
max_err = 50.**2
error_ts = np.array([-11.6e3, -10.9e3, -10.2e3, -9.7e3, -9.2e3, -8.7e3, -8.2e3, -7.75e3, -7.3e3, -3650., 0.])
dif1 = (obs_ages[1:] - obs_ages[:-1])
dif1 /= dif1.max()
Ls1_interp = interp1d(obs_ages, obs_Ls1)
Ls2_interp = interp1d(obs_ages, obs_Ls2)
Ls3_interp = interp1d(obs_ages, obs_Ls3)
dif1 = dif1*max_err
error_vs = np.array([min_err,  dif1[0],  min_err,   dif1[1],  min_err,   dif1[2],  min_err,   dif1[3],  min_err, dif1[4], min_err])
error_interp = interp1d(error_ts, error_vs, kind = 'linear')
errors1 = error_interp(ages_fine)

ax = plt.subplot(4,1,2)
plt.title('(b)')
plt.grid(color='lightgray', linestyle=':', linewidth=2.5)
plt.plot(ages, Ls1, 'r-', lw = 3)
plt.plot(obs_ages, obs_Ls1, 'ko--', lw = 3, ms = 8, dashes = (2,2))
plt.xlim([ages.min(), ages.max()])
plt.grid(color='lightgray', linestyle=':', linewidth=2.5)
plt.xticks([-11e3, -10e3, -9e3, -8e3, -7e3])
plt.xlim([ages.min(), ages.max()])
ticks = ax.get_xticks()
ax.set_xticklabels([int(abs(tick / 1000.)) for tick in ticks])
plt.fill_between(ages_fine, Ls1_interp(ages_fine) - 2.0*(np.sqrt(errors1)), Ls1_interp(ages_fine) + 2.0*(np.sqrt(errors1)), facecolor='gray', alpha = 0.5)
plt.fill_between(ages_fine, Ls1_interp(ages_fine) - (np.sqrt(errors1)), Ls1_interp(ages_fine) + (np.sqrt(errors1)), facecolor='gray', alpha = 0.5)

ax = plt.subplot(4,1,3)
plt.title('(c)')
plt.plot(ages, Ls2, 'r-', lw = 3)
plt.plot(obs_ages, obs_Ls2, 'ko--', lw = 3, ms = 8, dashes = (2,2))
plt.xlim([ages.min(), ages.max()])
plt.grid(color='lightgray', linestyle=':', linewidth=2.5)
plt.xticks([-11e3, -10e3, -9e3, -8e3, -7e3])
plt.xlim([ages.min(), ages.max()])
plt.ylabel('Glacier Length (km)')
ticks = ax.get_xticks()
ax.set_xticklabels([int(abs(tick / 1000.)) for tick in ticks])
plt.fill_between(ages_fine, Ls2_interp(ages_fine) - 2.0*(np.sqrt(errors1)), Ls2_interp(ages_fine) + 2.0*(np.sqrt(errors1)), facecolor='gray', alpha = 0.5)
plt.fill_between(ages_fine, Ls2_interp(ages_fine) - (np.sqrt(errors1)), Ls2_interp(ages_fine) + (np.sqrt(errors1)), facecolor='gray', alpha = 0.5)

ax = plt.subplot(4,1,4)
plt.title('(d)')
plt.plot(ages, Ls3, 'r-', lw = 3)
plt.plot(obs_ages, obs_Ls3, 'ko--', lw = 3, ms = 8, dashes = (2,2))
plt.grid(color='lightgray', linestyle=':', linewidth=2.5)
plt.xticks([-11e3, -10e3, -9e3, -8e3, -7e3])
plt.xlim([ages.min(), ages.max()])
ticks = ax.get_xticks()
ax.set_xticklabels([int(abs(tick / 1000.)) for tick in ticks])
plt.fill_between(ages_fine, Ls3_interp(ages_fine) - 2.0*(np.sqrt(errors1)), Ls3_interp(ages_fine) + 2.0*(np.sqrt(errors1)), facecolor='gray', alpha = 0.5)
plt.fill_between(ages_fine, Ls3_interp(ages_fine) - (np.sqrt(errors1)), Ls3_interp(ages_fine) + (np.sqrt(errors1)), facecolor='gray', alpha = 0.5)
plt.grid(True)
plt.xlabel('Age (ka BP)')

plt.tight_layout()
plt.savefig('seasonal_default.png', dpi=550)
