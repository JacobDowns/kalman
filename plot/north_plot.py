import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
import matplotlib

matplotlib.rcParams.update({'font.size': 20})
fig = plt.figure(figsize=(12,14))


### Plot North
###############################################################

ax = fig.add_subplot(311)

plt.title('(a)')
ages = np.loadtxt('paleo_runs/north_opt1/opt_ages.txt')
Ls =   np.loadtxt('paleo_runs/north_opt1/opt_Ls.txt')
#Ls1 = np.loadtxt('paleo_runs/north_opt2/opt_Ls.txt')
Ls1 = np.loadtxt('paleo_runs/south_jensen_land/opt_L.txt')
obs_ages = np.array([-11.6, -10.2, -9.2, -8.2, -7.3])*1e3
obs_Ls = np.array([443746.66897917818, 397822.86008538032, 329757.49741948338, 292301.29712071194, 285478.05793305294])
L_interp = interp1d(obs_ages, obs_Ls, kind = 'linear')

error_ts = np.array([-11.6, -10.9, -10.2, -9.7, -9.2, -8.7, -8.2, -7.75, -7.3])*1e3
min_err = 5000.**2
max_err = 50000.**2
error_vs = np.array([min_err,  max_err,  min_err,   max_err,  min_err,   max_err,  min_err,   max_err,  min_err])
error_interp = interp1d(error_ts, error_vs, kind = 'linear')
errors = error_interp(ages[:-1])

obs_Ls /= 1000.
Ls /= 1000.
Ls1 /= 1000.

plt.plot(obs_ages, obs_Ls, 'ko--', lw = 3, ms = 10)
plt.plot(ages, Ls, '#d62728', lw = 1)
plt.plot(ages, Ls1, '#d62728', lw = 1, linestyle = ':')

"""
plt.plot(ages[:-1], L_interp(ages[:-1]) + 2.0*np.sqrt(errors), 'k--', lw = 2)
plt.plot(ages[:-1], L_interp(ages[:-1]) - 2.0*np.sqrt(errors), 'k--', lw = 2)
"""
plt.xlim([ages.min(), ages.max()])
plt.grid(color='lightgray', linestyle=':', linewidth=2.5)
ticks = ax.get_xticks()
#plt.xlabel('Year Before Present')
#plt.ylabel(r'Glacier Length $L$')
ax.set_xticklabels([int(abs(tick / 1000.)) for tick in ticks])


### Plot Center
###############################################################

ax = fig.add_subplot(312)
plt.title('(b)')
ages = np.loadtxt('paleo_runs/center_opt1/opt_ages.txt')
Ls = np.loadtxt('paleo_runs/center_opt1/opt_Ls.txt')
Ls1 = np.loadtxt('paleo_runs/center_opt_error/opt_Ls.txt')
obs_ages = np.array([-11.6, -10.2, -9.2, -8.2, -7.3])*1e3
obs_Ls = np.array([406878.12855486432, 396313.20004890749, 321224.04532276397, 292845.40895793668, 288562.44342502725])
L_interp = interp1d(obs_ages, obs_Ls, kind = 'linear')

error_ts = np.array([-11.6, -10.9, -10.2, -9.7, -9.2, -8.7, -8.2, -7.75, -7.3])*1e3
min_err = 5000.**2
max_err = 50000.**2
error_vs = np.array([min_err,  max_err,  min_err,   max_err,  min_err,   max_err,  min_err,   max_err,  min_err])
error_interp = interp1d(error_ts, error_vs, kind = 'linear')
errors = error_interp(ages[:-1])

obs_Ls /= 1000.
Ls /= 1000.
Ls1 /= 1000.

plt.plot(obs_ages, obs_Ls, 'ko--', lw = 2, ms = 10)
plt.plot(ages, Ls,  '#2ca02c', lw = 2)
plt.plot(ages, Ls1,  '#2ca02c', lw = 2, linestyle = ':')
"""
plt.plot(ages[:-1], L_interp(ages[:-1]) + 2.0*np.sqrt(errors), 'k--', lw = 2)
plt.plot(ages[:-1], L_interp(ages[:-1]) - 2.0*np.sqrt(errors), 'k--', lw = 2)
"""
plt.xlim([ages.min(), ages.max()])
plt.grid(color='lightgray', linestyle=':', linewidth=2.5)
ticks = ax.get_xticks()
#plt.xlabel('Year Before Present')
plt.ylabel(r'Glacier Length $L$ (km)')
ax.set_xticklabels([int(abs(tick / 1000.)) for tick in ticks])

### Plot South
###############################################################


ax = fig.add_subplot(313)
plt.title('(c)')
ages = np.loadtxt('paleo_runs/south_opt/opt_ages.txt')
Ls = np.loadtxt('paleo_runs/south_opt/opt_Ls.txt')
Ls1 = np.loadtxt('paleo_runs/south_jensen_land/opt_L.txt')
obs_ages = np.array([-11.6, -10.2, -9.2, -8.2, -7.3])*1e3
obs_Ls = np.array([424777.2650658561, 394942.08036138373, 332430.91816515941, 303738.49932773202, 296659.0156905292])
L_interp = interp1d(obs_ages, obs_Ls, kind = 'linear')

error_ts = np.array([-11.6, -10.9, -10.2, -9.7, -9.2, -8.7, -8.2, -7.75, -7.3])*1e3
min_err = 5000.**2
max_err = 50000.**2
error_vs = np.array([min_err,  max_err,  min_err,   max_err,  min_err,   max_err,  min_err,   max_err,  min_err])
error_interp = interp1d(error_ts, error_vs, kind = 'linear')
errors = error_interp(ages[:-1])

obs_Ls /= 1000.
Ls /= 1000.
Ls1 /= 1000.

plt.plot(obs_ages, obs_Ls, 'ko--', lw = 3, ms = 10)
plt.plot(ages, Ls, '#1f77b4', lw = 5)
plt.plot(ages, Ls1, '#1f77b4', lw = 5, linestyle = ':')
"""
plt.plot(ages[:-1], L_interp(ages[:-1]) + 2.0*np.sqrt(errors), 'k--', lw = 2)
plt.plot(ages[:-1], L_interp(ages[:-1]) - 2.0*np.sqrt(errors), 'k--', lw = 2)
"""
plt.xlim([ages.min(), ages.max()])
plt.grid(color='lightgray', linestyle=':', linewidth=2.5)
ticks = ax.get_xticks()
plt.xlabel('Age (ka BP)')
#plt.ylabel(r'Glacier Length $L$')
ax.set_xticklabels([int(abs(tick / 1000.)) for tick in ticks])


plt.tight_layout()
plt.show()
#plt.savefig('test.png', dpi=750)
