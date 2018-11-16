import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
import matplotlib

matplotlib.rcParams.update({'font.size': 18})
fig = plt.figure(figsize=(12,14))

ages = np.loadtxt('transform_long/center2_seasonal/opt1/opt_age.txt')
Ls1 = np.loadtxt('transform_long/center2_seasonal/opt1/opt_L.txt') / 1000.
#Ls2 = np.loadtxt('transform_long/south2_seasonal/opt1/opt_L.txt') / 1000.

meas_ages = np.loadtxt('paleo_inputs/measure_ages.txt')
y_ages = np.loadtxt('paleo_inputs/y_ages.txt') * 1000.
ys1 = np.loadtxt('paleo_inputs/y_c.txt') / 1000.
#ys2 = np.loadtxt('transform_long/paleo_inputs/y_s.txt') / 1000.

obs_ages = np.array([-11.6e3, -10.2e3, -9.2e3, -8.2e3, -7.3e3, 0.])
obs_Ls1 = np.array([406878.12855486432, 396313.20004890749, 321224.04532276397, 292845.40895793668, 288562.44342502725, 279753.70997966686]) / 1000.

#obs_Ls1 = interp1d(obs_ages, obs_Ls1)(ages)
#obs_Ls2 = interp1d(obs_ages, obs_Ls2)(ages)
#obs_Ls3 = interp1d(obs_ages, obs_Ls3)(ages)

#plt.plot(obs_ages, obs_Ls1, 'ko--')
plt.plot(ages, Ls1, 'r')
#plt.plot(ages, Ls2, 'g')
plt.plot(y_ages, ys1, 'r--')
#plt.plot(meas_ages, ys2, 'g--')
plt.show()

quit()

ax = plt.subplot(3,1,1)
plt.title('(a)')
plt.plot(ages, obs_Ls1 - Ls1, 'k-', lw = 3)
plt.xlim([ages.min(), ages.max()])
plt.grid(color='lightgray', linestyle=':', linewidth=2.5)
plt.xticks([-11e3, -10e3, -9e3, -8e3, -7e3])
plt.plot(obs_ages, np.zeros(len(obs_ages)), 'ko')
plt.xlim([ages.min(), ages.max()])
ticks = ax.get_xticks()
ax.set_xticklabels([int(abs(tick / 1000.)) for tick in ticks])
plt.grid(True)

ax = plt.subplot(3,1,2)
plt.title('(b)')
plt.plot(ages, obs_Ls2 - Ls2, 'k-', lw = 3)
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
plt.plot(ages, obs_Ls3 - Ls3, 'k-', lw = 3)
plt.grid(color='lightgray', linestyle=':', linewidth=2.5)
plt.xticks([-11e3, -10e3, -9e3, -8e3, -7e3])
plt.xlim([ages.min(), ages.max()])
ticks = ax.get_xticks()
ax.set_xticklabels([int(abs(tick / 1000.)) for tick in ticks])
plt.grid(True)
plt.xlabel('Age (ka BP)')

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


plt.grid(color='lightgray', linestyle=':', linewidth=1)
ticks = ax.get_xticks()
#plt.xlabel('Year Before Present')
#plt.ylabel(r'Glacier Length $L$')
ax.set_xticklabels([int(abs(tick)) for tick in ticks])
plt.tight_layout()

plt.savefig('test.png', dpi=700)"""
