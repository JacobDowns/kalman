import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
import seaborn as sns
import matplotlib

matplotlib.rcParams.update({'font.size': 18})
fig = plt.figure(figsize=(10,10))

# Center 
L1_obs = np.array([406878.12855486432, 396313.20004890749, 321224.04532276397, 292845.40895793668, 288562.44342502725, 279753.70997966686]) / 1e3
# south
L2_obs  = np.array([424777.2650658561, 394942.08036138373, 332430.9181651594, 303738.499327732, 296659.0156905292, 284686.5963970118]) / 1e3

# Model ages
ages = np.loadtxt('transform_final/center2/opt1/opt_age.txt')
# Optimized lengths + errors
L1 = np.loadtxt('transform_final/center2/opt1/opt_L.txt') / 1e3
v1 = np.loadtxt('transform_final/center2/opt1/y_v.txt') / 1e3
L2 = np.loadtxt('transform_final/south2/opt1/opt_L.txt') / 1e3
v2 = np.loadtxt('transform_final/south2/opt1/y_v.txt') / 1e3

yc = np.loadtxt('paleo_inputs/y_c.txt') / 1e3
ys = np.loadtxt('paleo_inputs/y_s.txt') / 1e3

# Measurement ages
meas_indexes = range(0, len(ages), 25*3)
y_ages = ages[meas_indexes]
# Observation ages
obs_ages = np.array([-11554., -10284., -9024., -8064., -7234., 0.])
# Observation variances
obs_sigmas = np.array([0.4, 0.2, 0.2, 0.3, 0.3, 0.1])*1e3/2.


### Plot center
#################################################
ax = plt.subplot(2,1,1)
plt.title('(a)')

for i in [0, 1, 2, 3, 4]:
    plt.plot([ages.min(), ages.max()], [L1_obs[i], L1_obs[i]], 'gray', linestyle=':', alpha = 0.9, lw = 2)

plt.plot(ages, L1, 'k', lw=3.5)
plt.plot(y_ages, yc, 'k--', lw = 2)
#plt.plot(y_ages, yc, 'k--', lw=3.5)
#plt.plot(age[meas_indexes], y_c, 'k--', lw = 3)
#plt.grid(color='slategray', linestyle=':', linewidth=3, axis='x')

#plt.fill_between(y_ages, L1[meas_indexes]-v1, L1[meas_indexes]+v1,
#                     color='gray', alpha=.5)
#plt.fill_between(y_ages, L1[meas_indexes]-2.*v1, L1[meas_indexes]+2.*v1,
#                     color='gray', alpha=.5)

plt.xlim([ages.min(), ages.max()])
plt.ylabel('Glacier Length (km)')

"""
for i in [1,2,3,4]:
    index = np.where(L > Ls_center[i])[0].max()
    time = age[index]
    print(time)
    #plt.plot(time, Ls_center[i], 'rd', ms = 10)
"""

for i in range(len(obs_ages) - 1):
    plt.plot([obs_ages[i] - 2.0*obs_sigmas[i], obs_ages[i] + 2.0*obs_sigmas[i]], [L1_obs[i], L1_obs[i]], 'k', lw = 5, ms = 6, alpha = 1.)
    plt.plot([obs_ages[i] - 2.0*obs_sigmas[i], obs_ages[i] + 2.0*obs_sigmas[i]], [L1_obs[i], L1_obs[i]], 'r', lw = 3.5, ms = 6, alpha = 1.)
    plt.plot(obs_ages[i], L1_obs[i], 'ko', ms = 12)
    plt.plot(obs_ages[i], L1_obs[i], 'ro', ms = 10)

plt.plot(obs_ages[-1]-20., L1_obs[-1], 'ko', ms = 12)
plt.plot(obs_ages[-1]-20., L1_obs[-1], 'ro', ms = 10)

ticks = ax.get_xticks()
ax.set_xticklabels([int(abs(tick / 1000.)) for tick in ticks])


### Plot south
#################################################
ax = plt.subplot(2,1,2)

plt.title('(b)')

for i in [0, 1, 2, 3, 4]:
    plt.plot([ages.min(), ages.max()], [L2_obs[i], L2_obs[i]], 'gray', linestyle=':', alpha = 0.9, lw = 2)

plt.plot(ages, L2, 'k', lw=3.5)
plt.plot(y_ages, ys, 'k--', lw=3.5)

plt.xlim([ages.min(), ages.max()])

for i in range(len(obs_ages) - 1):
    plt.plot([obs_ages[i] - 2.0*obs_sigmas[i], obs_ages[i] + 2.0*obs_sigmas[i]], [L2_obs[i], L2_obs[i]], 'k', lw = 5, ms = 6, alpha = 1.)
    plt.plot([obs_ages[i] - 2.0*obs_sigmas[i], obs_ages[i] + 2.0*obs_sigmas[i]], [L2_obs[i], L2_obs[i]], 'r', lw = 3.5, ms = 6, alpha = 1.)
    plt.plot(obs_ages[i], L2_obs[i], 'ko', ms = 12)
    plt.plot(obs_ages[i], L2_obs[i], 'ro', ms = 10)

plt.plot(obs_ages[-1]-20., L2_obs[-1], 'ko', ms = 12)
plt.plot(obs_ages[-1]-20., L2_obs[-1], 'ro', ms = 10)

ticks = ax.get_xticks()
ax.set_xticklabels([int(abs(tick / 1000.)) for tick in ticks])

plt.ylabel('Glacier Length (km)')
plt.xlabel('Age (ka BP)')
    
plt.tight_layout()
plt.savefig('fit_final.png', dpi=500)    
#plt.show()

