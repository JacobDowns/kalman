import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
import matplotlib

matplotlib.rcParams.update({'font.size': 18})
fig = plt.figure(figsize=(12,14))

ages = np.loadtxt('transform_buizert/north2/opt1/opt_age.txt')
Ls1 = np.loadtxt('transform_buizert/north2/opt1/opt_L.txt') / 1000.
Ls2 = np.loadtxt('transform_buizert/center2/opt1/opt_L.txt') / 1000.
Ls3 = np.loadtxt('transform_buizert/south2/opt1/opt_L.txt') / 1000.

obs_ages = np.array([-11.6, -10.2, -9.2, -8.2, -7.3])*1e3
obs_Ls1 = np.array([443746.66897917818, 397822.86008538032, 329757.49741948338, 292301.29712071194, 285478.05793305294]) / 1000.
obs_Ls2 = np.array([406878.12855486432, 396313.20004890749, 321224.04532276397, 292845.40895793668, 288562.44342502725]) / 1000.
obs_Ls3 = np.array([424777.2650658561, 394942.08036138373, 332430.91816515941, 303738.49932773202, 296659.0156905292]) / 1000.

ax = plt.subplot(3,1,1)
plt.title('(a)')
plt.plot(ages, Ls1, 'r-', lw = 3)
plt.plot(obs_ages, obs_Ls1, 'ko--', lw = 3, ms = 8, dashes = (2,2))
plt.xlim([ages.min(), ages.max()])
plt.grid(color='lightgray', linestyle=':', linewidth=2.5)
plt.xticks([-11e3, -10e3, -9e3, -8e3, -7e3])
plt.xlim([ages.min(), ages.max()])
ticks = ax.get_xticks()
ax.set_xticklabels([int(abs(tick / 1000.)) for tick in ticks])
plt.grid(True)

ax = plt.subplot(3,1,2)
plt.title('(b)')
plt.plot(ages, Ls2, 'r-', lw = 3)
plt.plot(obs_ages, obs_Ls2, 'ko--', lw = 3, ms = 8, dashes = (2,2))
plt.xlim([ages.min(), ages.max()])
plt.grid(color='lightgray', linestyle=':', linewidth=2.5)
plt.xticks([-11e3, -10e3, -9e3, -8e3, -7e3])
plt.xlim([ages.min(), ages.max()])
plt.ylabel('Glacier Length (km)')
ticks = ax.get_xticks()
ax.set_xticklabels([int(abs(tick / 1000.)) for tick in ticks])
plt.grid(True)

ax = plt.subplot(3,1,3)
plt.title('(c)')
plt.plot(ages, Ls3, 'r-', lw = 3)
plt.plot(obs_ages, obs_Ls3, 'ko--', lw = 3, ms = 8, dashes = (2,2))
plt.grid(color='lightgray', linestyle=':', linewidth=2.5)
plt.xticks([-11e3, -10e3, -9e3, -8e3, -7e3])
plt.xlim([ages.min(), ages.max()])
ticks = ax.get_xticks()
ax.set_xticklabels([int(abs(tick / 1000.)) for tick in ticks])
plt.grid(True)
plt.xlabel('Age (ka BP)')

plt.savefig('buizert_fit.png', dpi=700)
