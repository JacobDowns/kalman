import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
import seaborn as sns
import matplotlib
from scipy.signal import savgol_filter
from scipy import signal
from data_loader import DataLoader
from matplotlib.ticker import FormatStrFormatter

matplotlib.rcParams.update({'font.size': 18})

fig = plt.figure(figsize=(12,13))
ksmooth = signal.hann(7) / signal.hann(7).sum()
start = -11.6
end = -10.


### Buizert temp.
############################################################

ax = fig.add_subplot(3,1,1)
ax.set_title('(a)')
#ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))

current_palette = sns.color_palette("coolwarm", 12)

indexes = [0, 1, 2, 3, 4, 5, 6,  7,  8, 9, 10, 11]
indexes = [0, 2, 4, 7, 8, 9, 10, 8, 6, 5, 3,  1]

d_ages = np.loadtxt('../paleo_data/dj_ages_seasonal.txt') / 1e3
d_dts = np.loadtxt('../paleo_data/dj_dts_seasonal.txt')
d_avg = d_dts.mean(axis = 1)

for i in range(12):
    dt_smooth = np.convolve(d_dts[:,i], ksmooth, 'same')
    plt.plot(d_ages, dt_smooth, color = 'k', lw = 4)
    plt.plot(d_ages, dt_smooth, color = current_palette[indexes[i]], lw = 3)

plt.plot(d_ages, d_avg, 'k', lw = 3)
plt.xlim([start, end])
plt.ylim([-12., 3.])
plt.grid(color='slategray', linestyle=':', linewidth=1)
plt.ylabel(r'$\Delta T$ ($^{\circ}$ C)')

current_palette =  sns.color_palette()

ages = np.loadtxt('../default_runs/center1/opt_age.txt')/ 1e3
L1 = np.loadtxt('../default_runs/center1/opt_L.txt') / 1e3
L2 = np.loadtxt('../default_runs/south1/opt_L.txt') / 1e3

plt.xticks([-11.5, -11., -10.5, -10.])


### Glacier Length North
##########################################################

ax = plt.subplot(3,1,2)
ax.set_title('(b)')
# Center 
L1_obs = np.array([406878, 396313, 321224, 292845, 288562, 279753]) / 1e3
# south
L2_obs  = np.array([424777, 394942, 332430, 303738, 296659, 284686]) / 1e3
# Observation ages
obs_ages = np.array([-11554., -10284., -9024., -8064., -7234., 0.]) / 1e3
# Observation variances
obs_sigmas = np.array([0.4, 0.2, 0.2, 0.3, 0.3, 0.1])/2.

plt.plot(ages, L1, color = 'k', lw = 6.5) 
plt.plot(ages, L1, color = current_palette[0], lw = 4.5, label = 'North')

plt.xlim([start, 0.])

for i in range(len(obs_ages) - 1):
    plt.plot([obs_ages[i] - 2.0*obs_sigmas[i], obs_ages[i] + 2.0*obs_sigmas[i]], [L1_obs[i], L1_obs[i]], 'k', lw = 5.5, ms = 4, alpha = 0.9)
    plt.plot(obs_ages[i], L1_obs[i], 'ko', ms = 10)

plt.plot(obs_ages[-1], L1_obs[-1], 'ko', ms = 10)
plt.ylim([300, 420])
plt.xlim([start, end])
plt.grid(color='slategray', linestyle=':', linewidth=1)
plt.xticks([-11.5, -11., -10.5, -10.])
plt.ylabel('Glacier Length (km)')


### Glacier Length South
##########################################################


ax = plt.subplot(3,1,3)
ax.set_title('(c)')
plt.plot(ages, L2, color = 'k', lw = 6.5) 
plt.plot(ages, L2, color = current_palette[3], lw = 4.5, label = 'South')

for i in range(len(obs_ages) - 1):
    plt.plot([obs_ages[i] - 2.0*obs_sigmas[i], obs_ages[i] + 2.0*obs_sigmas[i]], [L2_obs[i], L2_obs[i]], 'k', lw = 5.5, ms = 4, alpha = 0.9)
    plt.plot(obs_ages[i], L2_obs[i], 'ko', ms = 10)

plt.plot(obs_ages[-1], L2_obs[-1], 'ko', ms = 10)

plt.ylim([300, 440])
plt.xlim([start, end])
plt.grid(color='slategray', linestyle=':', linewidth=1)
plt.xticks([-11.5, -11., -10.5, -10.])
plt.ylabel('Glacier Length (km)')
plt.xlabel('Age (ka BP)')


plt.tight_layout()

plt.savefig('images/default_results.png', dpi=500)  
