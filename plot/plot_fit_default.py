import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
import matplotlib

matplotlib.rcParams.update({'font.size': 18})
fig = plt.figure(figsize=(12,17.5))

ages = np.loadtxt('paleo_runs/north_dj/age.txt')

Ls1 = np.loadtxt('paleo_runs/north_dj/L.txt') / 1000.
Ls2 = np.loadtxt('paleo_runs/north_buizert/L.txt') / 1000.

Ls3 = np.loadtxt('paleo_runs/center_dj/L.txt') / 1000.

ages1 = np.loadtxt('paleo_runs/center_buizert/age.txt')
Ls4 = np.loadtxt('paleo_runs/center_buizert/L.txt') / 1000.

Ls5 = np.loadtxt('paleo_runs/south_dj/L.txt') / 1000.
Ls6 = np.loadtxt('paleo_runs/south_buizert/L.txt') / 1000.

obs_ages = np.array([-11.6, -10.2, -9.2, -8.2, -7.3])*1e3
obs_Ls1 = np.array([443746.66897917818, 397822.86008538032, 329757.49741948338, 292301.29712071194, 285478.05793305294]) / 1000.
obs_Ls2 = np.array([406878.12855486432, 396313.20004890749, 321224.04532276397, 292845.40895793668, 288562.44342502725]) / 1000.
obs_Ls3 = np.array([424777.2650658561, 394942.08036138373, 332430.91816515941, 303738.49932773202, 296659.0156905292]) / 1000.

### Plot dt forcings
########################################################

data = np.loadtxt('paleo_data/jensen_dye3.txt')
years = data[:,0] - 2000.0
temps = data[:,1]
delta_temp1 = interp1d(years, temps - temps[-1], kind = 'linear')(ages)

data = np.loadtxt('paleo_data/buizert_dye3.txt')
years = -data[:,0][::-1]
temps = data[:,1][::-1]
delta_temp2 = interp1d(years, temps - temps[-1], kind = 'linear')(ages)

ax = plt.subplot(4,1,1)
plt.title('(a)')
plt.plot(ages, delta_temp1, 'r--', lw = 3, label = 'Dahl-Jensen')
plt.plot(ages, delta_temp2, 'r-', lw = 3, label = 'Buizert')
plt.xlim([ages.min(), ages.max()])
plt.grid(color='lightgray', linestyle=':', linewidth=2.5)
plt.xticks([-11e3, -10e3, -9e3, -8e3, -7e3])
plt.xlim([ages.min(), ages.max()])
plt.ylabel(r'$\Delta T$  ($^{\circ}$ C)')
plt.legend()
ticks = ax.get_xticks()
ax.set_xticklabels([int(abs(tick / 1000.)) for tick in ticks])
plt.grid(True)


### South
########################################################

ax = plt.subplot(4,1,2)
plt.title('(b)')
plt.plot(ages, Ls1, 'r--', lw = 3)
plt.plot(ages, Ls2, 'r-', lw = 3)
plt.plot(obs_ages, obs_Ls1, 'ko--', lw = 3, ms = 8, dashes = (2,2))
plt.xlim([ages.min(), ages.max()])
plt.grid(color='lightgray', linestyle=':', linewidth=2.5)
plt.xticks([-11e3, -10e3, -9e3, -8e3, -7e3])
plt.xlim([ages.min(), ages.max()])
plt.ylim([250., 460.])
plt.ylabel('Glacier Length (km)')
ticks = ax.get_xticks()
ax.set_xticklabels([int(abs(tick / 1000.)) for tick in ticks])
plt.grid(True)

### Center
########################################################

error_ts = np.array([-11.6, -10.9, -10.2, -9.7, -9.2, -8.7, -8.2, -7.75, -7.3])*1e3
min_err = 5.**2
max_err = 50.**2
error_vs = np.array([min_err,  max_err,  min_err,   max_err,  min_err,   max_err,  min_err,   max_err,  min_err])
error_interp = interp1d(error_ts, error_vs, kind = 'linear')
errors = 2.*np.sqrt(error_interp(ages))
obs_Ls2_fine = interp1d(obs_ages, obs_Ls2)(ages)

ax = plt.subplot(4,1,3)
plt.title('(c)')
plt.plot(ages, Ls3, 'r--', lw = 3)
plt.plot(ages1, Ls4, 'r-', lw = 3)
plt.plot(obs_ages, obs_Ls2, 'ko--', lw = 3, ms = 8, dashes = (2,2))
#plt.plot(ages, obs_Ls2_fine + errors, 'r--', lw = 3, ms = 7)
#plt.plot(ages, obs_Ls2_fine - errors, 'r--', lw = 3, ms = 7)
plt.xlim([ages.min(), ages.max()])
plt.grid(color='lightgray', linestyle=':', linewidth=2.5)
plt.xticks([-11e3, -10e3, -9e3, -8e3, -7e3])
plt.xlim([ages.min(), ages.max()])
plt.ylim([250., 420.])
plt.ylabel('Glacier Length (km)')
ticks = ax.get_xticks()
ax.set_xticklabels([int(abs(tick / 1000.)) for tick in ticks])
plt.grid(True)

ax = plt.subplot(4,1,4)
plt.title('(d)')
plt.plot(ages, Ls5, 'r--', lw = 3)
plt.plot(ages, Ls6, 'r-', lw = 3)
plt.plot(obs_ages, obs_Ls3, 'ko--', lw = 3, ms = 8, dashes = (2,2))
plt.grid(color='lightgray', linestyle=':', linewidth=2.5)
plt.xticks([-11e3, -10e3, -9e3, -8e3, -7e3])
plt.xlim([ages.min(), ages.max()])
plt.ylim([250., 440.])
ticks = ax.get_xticks()
ax.set_xticklabels([int(abs(tick / 1000.)) for tick in ticks])
plt.grid(True)
plt.xlabel('Age (ka BP)')
plt.ylabel('Glacier Length (km)')
plt.savefig('default_fit.png', dpi=700)



"""
Ls =   np.loadtxt('paleo_runs/center_opt/opt_Ls.txt')
obs_Ls = np.array([406878.12855486432, 396313.20004890749, 321224.04532276397, 292845.40895793668, 288562.44342502725])
L_interp = interp1d(obs_ages, obs_Ls, kind = 'linear')

plt.plot(obs_ages, obs_Ls, color = '#2ca02c', lw = 2, ms = 8, linestyle = '--', marker = 'o')
plt.plot(ages, Ls, , lw = 2)

obs_Ls = np.array([424777.2650658561, 394942.08036138373, 332430.91816515941, 303738.49932773202, 296659.0156905292])
L_interp = interp1d(obs_ages, obs_Ls, kind = 'linear')

plt.plot(obs_ages, obs_Ls, 'ko--', lw = 2, ms = 8)
plt.plot(ages, Ls, 'r', lw = 2)


plt.grid(color='lightgray', linestyle=':', linewidth=1)
ticks = ax.get_xticks()
#plt.xlabel('Year Before Present')
#plt.ylabel(r'Glacier Length $L$')
ax.set_xticklabels([int(abs(tick)) for tick in ticks])
plt.tight_layout()

plt.savefig('test.png', dpi=700)"""
