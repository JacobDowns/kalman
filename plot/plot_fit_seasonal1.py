import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
import matplotlib
import seaborn as sns

matplotlib.rcParams.update({'font.size': 18})
fig = plt.figure(figsize=(12,9))
current_palette = sns.color_palette()

ages = np.loadtxt('transform_long/south2_seasonal/opt1/opt_age.txt')
Ls1 = np.loadtxt('transform_long/north2_seasonal/opt1/opt_L.txt') / 1000.
Ls2 = np.loadtxt('transform_long/center2_seasonal/opt1/opt_L.txt') / 1000.
Ls3 = np.loadtxt('transform_long/south2_seasonal/opt1/opt_L.txt') / 1000.

obs_ages = np.array([-11.6, -10.2, -9.2, -8.2, -7.3, 0.0])*1e3
obs_Ls1 = np.array([470376, 442567, 351952, 313633, 307167, 300896]) / 1000. 
obs_Ls2 = np.array([406878, 396313, 321224, 292845, 288562, 279753]) / 1000.
obs_Ls3 = np.array([424777, 394942, 332430, 303738, 296659, 284686]) / 1000. 


### Error structure
######################################################################

ages_fine = np.linspace(-11.6, 0.0, 1000)*1e3
min_err = 5.**2
max_err = 60.**2
error_ts = np.array([-11.6e3, -10.9e3, -10.2e3, -9.7e3, -9.2e3, -8.7e3, -8.2e3, -7.75e3, -7.3e3, -3650., 0.])
dif1 = (obs_ages[1:] - obs_ages[:-1])
dif1 /= dif1.max()
Ls1_interp = interp1d(obs_ages, obs_Ls1)
Ls2_interp = interp1d(obs_ages, obs_Ls2)
Ls3_interp = interp1d(obs_ages, obs_Ls3)
dif1 = dif1*max_err
error_vs = np.array([min_err,  dif1[0],  min_err,   dif1[1],  min_err,   dif1[2],  min_err,   dif1[3],  min_err, dif1[4], min_err])
error_interp = interp1d(error_ts, error_vs, kind = 'linear')
errors1 = error_interp(ages)


"""
ax = plt.subplot(2,1,1)

plt.title('(a)')
plt.plot(ages, Ls1, 'r-', lw = 3)
plt.plot(obs_ages, obs_Ls1, 'ko--', lw = 3, ms = 8, dashes = (2,2))
plt.xlim([ages.min(), ages.max()])
plt.grid(color='lightgray', linestyle=':', linewidth=2.5)
plt.xticks([-11e3, -10e3, -9e3, -8e3, -7e3])
plt.xlim([ages.min(), ages.max()])
ticks = ax.get_xticks()
ax.set_xticklabels([int(abs(tick / 1000.)) for tick in ticks])
plt.grid(True)"""

ax = plt.subplot(2,1,1)

plt.title('(a)')
plt.plot(obs_ages, obs_Ls2, 'ko--', lw = 4, ms = 10, dashes = (2,2))
plt.plot(ages, Ls2, color = current_palette[3], lw = 4, alpha = 0.85)
plt.xlim([ages.min(), ages.max()])
plt.grid(color='lightgray', linestyle=':', linewidth=2.5)
plt.xlim([ages.min(), ages.max()])
plt.ylabel('Glacier Length (km)')
ticks = ax.get_xticks()
ax.set_xticklabels([int(abs(tick / 1000.)) for tick in ticks])
plt.fill_between(ages, Ls2_interp(ages) - 2.0*(np.sqrt(errors1)), Ls2_interp(ages) + 2.0*(np.sqrt(errors1)), facecolor='gray', alpha = 0.33)
plt.fill_between(ages, Ls2_interp(ages) - np.sqrt(errors1), Ls2_interp(ages) + np.sqrt(errors1), facecolor='gray', alpha = 0.5)
plt.ylim([240., 435.])

ax = plt.subplot(2,1,2)

plt.title('(b)')
plt.plot(obs_ages, obs_Ls3, 'ko--', lw = 4, ms = 10, dashes = (2,2))
plt.plot(ages, Ls3, color = current_palette[3], lw = 4, alpha = 0.85)
plt.grid(color='lightgray', linestyle=':', linewidth=2.5)
plt.xlim([ages.min(), ages.max()])
ticks = ax.get_xticks()
ax.set_xticklabels([int(abs(tick / 1000.)) for tick in ticks])
plt.xlabel('Age (ka BP)')
plt.ylabel('Glacier Length (km)')
plt.fill_between(ages, Ls3_interp(ages) - 2.0*(np.sqrt(errors1)), Ls3_interp(ages) + 2.0*(np.sqrt(errors1)), facecolor='gray', alpha = 0.33)
plt.fill_between(ages, Ls3_interp(ages) - np.sqrt(errors1), Ls3_interp(ages) + np.sqrt(errors1), facecolor='gray', alpha = 0.5)
plt.ylim([245., 450.])

plt.tight_layout()

#plt.show()
plt.savefig('seasonal_fit.png', dpi=700)
