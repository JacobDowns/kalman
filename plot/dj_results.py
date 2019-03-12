import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
import seaborn as sns
import matplotlib
from scipy.signal import savgol_filter
from scipy import signal
from data_loader import DataLoader

matplotlib.rcParams.update({'font.size': 18})

fig = plt.figure(figsize=(12,16.33))
ksmooth = signal.hann(11) / signal.hann(11).sum()
start = -10.

### Buizert temp.
############################################################

ax = fig.add_subplot(4,1,1)
#ax.axvspan(-10., -8., alpha=0.33, color='gray')
current_palette = sns.color_palette("coolwarm", 12)

indexes = [0, 1, 2, 3, 4, 5, 6,  7,  8, 9, 10, 11]
indexes = [0, 2, 4, 7, 8, 9, 10, 8, 6, 5, 3,  1]

d_ages = np.loadtxt('../paleo_data/dj_ages_seasonal.txt') / 1e3
d_dts = np.loadtxt('../paleo_data/dj_dts_seasonal.txt')
d_avg = d_dts.mean(axis = 1)

plt.fill_between([-8.5, -3.], [-1e6, -1e6], [1e6, 1e6], color = 'gray', alpha = 0.3)

for i in range(12):
    dt_smooth = np.convolve(d_dts[:,i], ksmooth, 'same')
    plt.plot(d_ages, dt_smooth, color = 'k', lw = 2)
    plt.plot(d_ages, dt_smooth, color = current_palette[indexes[i]], lw = 1.75)

plt.plot(d_ages, d_avg, 'k', lw = 3)
#plt.legend()
plt.xlim([start, 0.])
plt.ylim([-5.5, 5.5])
plt.grid(color='slategray', linestyle=':', linewidth=1)
plt.ylabel(r'$\Delta T$ ($^{\circ}$ C)')


### Optimized delta P
##########################################################

ax = fig.add_subplot(4,1,2)
#ax.axvspan(-10., -8., alpha=0.33, color='gray')
current_palette =  sns.color_palette()

data1 = DataLoader('../transform_dj_seasonal/center3/', 'opt1/')
data2 = DataLoader('../transform_dj_seasonal/south/', 'opt1/')

plt.fill_between([-8.5, -3.], [-1e6, -1e6], [1e6, 1e6], color = 'gray', alpha = 0.3)

# North
plt.plot(data1.sigma_ages, data1.deltap, color = 'k', marker = 'o', lw = 3.5, ms=8)
plt.plot(data1.sigma_ages, data1.deltap, color = current_palette[0], marker = 'o', lw = 2.25, ms=6, label = 'North')

# South
plt.plot(data2.sigma_ages, data2.deltap, color = 'k', marker = 'o', lw = 3.5, ms=8)
plt.plot(data2.sigma_ages, data2.deltap, color = current_palette[3], marker = 'o', lw = 2.25, ms=6, label = 'North')

plt.ylim([-0.05, .55])
plt.xlim([start, 0.])
plt.grid(color='slategray', linestyle=':', linewidth=1)
plt.ylabel(r'$\Delta P$ (m.w.e. a$^{-1}$)')



### Snowfall
##########################################################


ax = fig.add_subplot(4,1,3)

plt.fill_between([-8.5, -3.], [-1e6, -1e6], [1e6, 1e6], color = 'gray', alpha = 0.3)

# North
plt.plot(data1.ages, data1.precip / data1.precip[-1], color = 'k', lw = 3.)
plt.plot(data1.ages, data1.precip / data1.precip[-1], color = current_palette[0], lw = 2.)

# South
plt.plot(data2.ages, data2.precip / data2.precip[-1], color = 'k', lw = 3.)
plt.plot(data2.ages, data2.precip / data2.precip[-1], color = current_palette[3], lw = 2.)

plt.ylim([0.25, 1.6])
plt.xlim([start, 0.])
plt.grid(color='slategray', linestyle=':', linewidth=1)
plt.ylabel('Snowfall (% of modern)')

### Glacier lengths
##########################################################

ax = plt.subplot(4,1,4)
#ax.axvspan(-10., -8., alpha=0.33, color='gray')
plt.fill_between([-8.5, -3.], [-1e6, -1e6], [1e6, 1e6], color = 'gray', alpha = 0.3)

# Center 
L1_obs = np.array([406878, 396313, 321224, 292845, 288562, 279753]) / 1e3
# south
L2_obs  = np.array([424777, 394942, 332430, 303738, 296659, 284686]) / 1e3
# Observation ages
obs_ages = np.array([-11554., -10284., -9024., -8064., -7234., 0.]) / 1e3
# Observation variances
obs_sigmas = np.array([0.4, 0.2, 0.2, 0.3, 0.3, 0.1])/2.

plt.plot(data1.ages, data1.L, color = 'k', lw = 5)
plt.plot(data1.ages, data1.L, color = current_palette[0], lw = 4, label = 'North')
plt.xlim([start, 0.])

plt.plot(data2.ages, data2.L, color = 'k', lw = 5)
plt.plot(data2.ages, data2.L, color = current_palette[3], lw = 4, label = 'South')

for i in range(len(obs_ages) - 1):
    plt.plot([obs_ages[i] - 2.0*obs_sigmas[i], obs_ages[i] + 2.0*obs_sigmas[i]], [L1_obs[i], L1_obs[i]], 'k', lw = 5, ms = 6, alpha = 1.)
    #plt.plot([obs_ages[i] - 2.0*obs_sigmas[i], obs_ages[i] + 2.0*obs_sigmas[i]], [L1_obs[i], L1_obs[i]], color = current_palette[0], lw = 2, ms = 6, alpha = 1.)
    plt.plot(obs_ages[i], L1_obs[i], 'ko', ms = 10)
    #plt.plot(obs_ages[i], L1_obs[i], 'ro', ms = 10)

plt.plot(obs_ages[-1], L1_obs[-1], 'ko', ms = 10)


for i in range(len(obs_ages) - 1):
    plt.plot([obs_ages[i] - 2.0*obs_sigmas[i], obs_ages[i] + 2.0*obs_sigmas[i]], [L2_obs[i], L2_obs[i]], 'k', lw = 5, ms = 6, alpha = 1.)
    #plt.plot([obs_ages[i] - 2.0*obs_sigmas[i], obs_ages[i] + 2.0*obs_sigmas[i]], [L2_obs[i], L2_obs[i]], color = current_palette[3], lw = 2, ms = 6, alpha = 1.)
    plt.plot(obs_ages[i], L2_obs[i], 'ko', ms = 10)
    #plt.plot(obs_ages[i], L2_obs[i], 'ro', ms = 10)

plt.plot(obs_ages[-1], L2_obs[-1], 'ko', ms = 10)

plt.ylim([275, 390])
plt.xlim([start, 0.])
plt.grid(color='slategray', linestyle=':', linewidth=1)


plt.ylabel('Glacier Length (km)')
plt.xlabel('Age (ka BP)')
#plt.legend()

plt.tight_layout()

plt.savefig('images/dj_results.png', dpi=500)  
