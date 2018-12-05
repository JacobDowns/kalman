import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
import matplotlib
import seaborn as sns

matplotlib.rcParams.update({'font.size': 18})
current_palette = sns.color_palette("RdBu_r", 12)
fig = plt.figure(figsize=(10,15))

dt_years = np.loadtxt('paleo_data/buizert_ages.txt')
dt_vals = np.loadtxt('paleo_data/buizert_dts.txt')

 # Interpolate the anomalies
dt_functions = []
for i in range(12):
    dt_functions.append(interp1d(dt_years, dt_vals[:,i], kind = 'linear'))

ax = plt.subplot(3,1,1)
plt.title('(a)')
ts = np.linspace(-11.6e3, -7.3e3, 300)
#[11, 2, 5, 8]
plt.plot(ts, dt_functions[11](ts), color = 'k', lw = 2.75)
plt.plot(ts, dt_functions[11](ts), color = current_palette[0], lw = 2.25, label = 'Dec.')
plt.plot(ts, dt_functions[2](ts), color = 'k', lw = 2.75)
plt.plot(ts, dt_functions[2](ts), color = current_palette[-3], lw = 2.25, label = 'Mar.')
plt.plot(ts, dt_functions[5](ts), color = 'k', lw = 2.75)
plt.plot(ts, dt_functions[5](ts), color = current_palette[-1], lw = 2.25, label = 'Jun.')
plt.plot(ts, dt_functions[8](ts), color = 'k', lw = 2.75)
plt.plot(ts, dt_functions[8](ts), color = current_palette[2], lw = 2.25, label = 'Sep.')
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

# Observation ages
obs_ages = np.array([-11554., -10284., -9024., -8064., -7234., 0.])
# Observation variances
obs_sigmas = np.array([0.4, 0.2, 0.2, 0.3, 0.3, 0.1])*1e3 / 2.
# Center 
L1_obs = np.array([406878.12855486432, 396313.20004890749, 321224.04532276397, 292845.40895793668, 288562.44342502725, 279753.70997966686]) / 1e3
# south
L2_obs  = np.array([424777.2650658561, 394942.08036138373, 332430.9181651594, 303738.499327732, 296659.0156905292, 284686.5963970118]) / 1e3


ages = np.loadtxt('default_runs/center/age.txt')
Ls1 = np.loadtxt('default_runs/center/L.txt') / 1000.
Ls2 = np.loadtxt('default_runs/south/L.txt') / 1000.


### North
#################################################################

ax = plt.subplot(3,1,2)
plt.title('(b)')

for i in [0, 1, 2, 3, 4]:
    plt.plot([ages.min(), -7350.], [L1_obs[i], L1_obs[i]], 'gray', linestyle=':', alpha = 1., lw = 2)

plt.plot(ages, Ls1, 'k', lw = 3.5)
#plt.grid(color='lightgray', linestyle=':', linewidth=2.5)
plt.xticks([-11e3, -10e3, -9e3, -8e3, -7e3])
plt.xlim([obs_ages.min(), -7350.])
plt.ylim([225., 420.])
plt.ylabel('Glacier Length (km)')
ticks = ax.get_xticks()
ax.set_xticklabels([int(abs(tick / 1000.)) for tick in ticks])

for i in range(len(obs_ages) - 1):
    plt.plot([obs_ages[i] - 2.0*obs_sigmas[i], obs_ages[i] + 2.0*obs_sigmas[i]], [L1_obs[i], L1_obs[i]], 'k', lw = 5, ms = 6, alpha = 1.)
    plt.plot([obs_ages[i] - 2.0*obs_sigmas[i], obs_ages[i] + 2.0*obs_sigmas[i]], [L1_obs[i], L1_obs[i]], 'r', lw = 3.5, ms = 6, alpha = 1.)
    plt.plot(obs_ages[i], L1_obs[i], 'ko', ms = 12)
    plt.plot(obs_ages[i], L1_obs[i], 'ro', ms = 10)

    
### South
#################################################################
    
ax = plt.subplot(3,1,3)
plt.title('(c)')

for i in [0, 1, 2, 3, 4]:
    plt.plot([ages.min(), -7350.], [L2_obs[i], L2_obs[i]], 'gray', linestyle=':', alpha = 1., lw = 2)

plt.plot(ages, Ls2, 'k', lw = 3.5)
#plt.grid(color='lightgray', linestyle=':', linewidth=2.5)
plt.xticks([-11e3, -10e3, -9e3, -8e3, -7e3])
plt.xlim([obs_ages.min(), -7350.])
plt.ylim([225., 440.])
plt.ylabel('Glacier Length (km)')
plt.xlabel('Age (ka BP)')
ticks = ax.get_xticks()
ax.set_xticklabels([int(abs(tick / 1000.)) for tick in ticks])

for i in range(len(obs_ages) - 1):
    plt.plot([obs_ages[i] - 2.0*obs_sigmas[i], obs_ages[i] + 2.0*obs_sigmas[i]], [L2_obs[i], L2_obs[i]], 'k', lw = 5, ms = 6, alpha = 1.)
    plt.plot([obs_ages[i] - 2.0*obs_sigmas[i], obs_ages[i] + 2.0*obs_sigmas[i]], [L2_obs[i], L2_obs[i]], 'r', lw = 3.5, ms = 6, alpha = 1.)
    plt.plot(obs_ages[i], L2_obs[i], 'ko', ms = 12)
    plt.plot(obs_ages[i], L2_obs[i], 'ro', ms = 10)


plt.tight_layout()
plt.savefig('default_final.png', dpi=500)    
