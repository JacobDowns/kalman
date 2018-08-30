import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
import matplotlib

matplotlib.rcParams.update({'font.size': 14})
#fig = plt.figure(figsize=(12,14))


ages = np.loadtxt('transform/north1/opt2/opt_age.txt')
Ls =   np.loadtxt('transform/north1/opt2/opt_L.txt')
obs_ages = np.array([-11.6, -10.2, -9.2, -8.2, -7.3])*1e3

#obs_Ls = np.array([424777.2650658561, 394942.08036138373, 332430.91816515941, 303738.49932773202, 296659.0156905292])
obs_Ls = np.array([443746.66897917818, 397822.86008538032, 329757.49741948338, 292301.29712071194, 285478.05793305294])
L_interp = interp1d(obs_ages, obs_Ls, kind = 'linear')

plt.plot(obs_ages, obs_Ls, '#d62728', lw = 2, ms = 8, linestyle = '--', marker = 'o')
plt.plot(ages, Ls, '#d62728', lw = 2)
plt.show()

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

plt.xlim([ages.min(), ages.max()])
plt.grid(color='lightgray', linestyle=':', linewidth=1)
ticks = ax.get_xticks()
#plt.xlabel('Year Before Present')
#plt.ylabel(r'Glacier Length $L$')
ax.set_xticklabels([int(abs(tick)) for tick in ticks])
plt.tight_layout()

plt.savefig('test.png', dpi=700)"""
