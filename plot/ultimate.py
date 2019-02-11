import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
import seaborn as sns
import matplotlib
from scipy.signal import savgol_filter
from scipy import signal
from data_loader import DataLoader

matplotlib.rcParams.update({'font.size': 18})

fig = plt.figure(figsize=(12,13))
ksmooth = signal.hann(11) / signal.hann(11).sum()
start = -9.5


### Buizert temp.
############################################################

ax = fig.add_subplot(3,1,1)
current_palette = sns.color_palette("coolwarm", 12)

indexes = [0, 1, 2, 3, 4, 5, 6,  7,  8, 9, 10, 11]
indexes = [0, 2, 4, 7, 8, 9, 10, 8, 6, 5, 3,  1]

dt_years = np.loadtxt('../paleo_data/buizert_ages.txt') / 1e3
dt_vals = np.loadtxt('../paleo_data/buizert_dts.txt').T

dt_w = [0, 1, 2]


dt_avg = np.zeros(len(dt_years))
for i in range(12):
    dt_smooth = signal.convolve(dt_vals[i], ksmooth, mode = 'same') #savgol_filter(dt_vals[i], 21, 2, mode = 'constant')
    #plt.plot(dt_years, dt_smooth, color = 'k', lw = 2, alpha = 1)
    #plt.plot(dt_years, dt_smooth, color = current_palette[indexes[i]], lw = 1.4, alpha = 1.)
    dt_avg += (1./12.)*dt_vals[i]

dt_smooth = signal.convolve(dt_vals[i], ksmooth, mode = 'same')
#plt.plot(dt_years, dt_smooth, color = 'w', lw = 5.5)
plt.plot(dt_years, dt_smooth, color = 'k', lw = 3)

plt.xlim([start, 0.])
plt.ylim([-6., 3.5])
plt.grid(color='slategray', linestyle=':', linewidth=1)
plt.ylabel(r'$\Delta T$ ($^{\circ}$ C)')


### Optimized delta P
##########################################################

ax = fig.add_subplot(3,1,2)
current_palette =  sns.color_palette()

data1 = DataLoader('../transform_final/center3/', 'opt1/')
data2 = DataLoader('../transform_final/south2/', 'opt1/')
data3 = DataLoader('../transform_final/center_sensitivity/', 'opt1/')

def_age1 = np.loadtxt('../default_runs/center1/age.txt') / 1e3
def_L1 = np.loadtxt('../default_runs/center1/L.txt')
def_precip1 = np.loadtxt('../default_runs/center1/P.txt') / def_L1

def_age2 = np.loadtxt('../default_runs/south1/age.txt') / 1e3
def_L2 = np.loadtxt('../default_runs/south1/L.txt')
def_precip2 = np.loadtxt('../default_runs/south1/P.txt') / def_L2


#plt.plot(def_age1, def_precip1, color = current_palette[0], alpha = 0.8, lw = 2)
#plt.plot(def_age2, def_precip2, color = current_palette[3], alpha = 0.8, lw = 2)

# North
plt.plot(data1.sigma_ages, data1.deltap, color = 'k', marker = 'o', lw = 3.5, ms=8)
plt.plot(data1.sigma_ages, data1.deltap, color = current_palette[0], marker = 'o', lw = 2.25, ms=6, label = 'North')
plt.plot(data1.ages, data1.precip, color = 'k', lw = 3.)
plt.plot(data1.ages, data1.precip, color = current_palette[0], lw = 2.)

# South
plt.plot(data2.sigma_ages, data2.deltap, color = 'k', marker = 'o', lw = 3.5, ms=8)
plt.plot(data2.sigma_ages, data2.deltap, color = current_palette[3], marker = 'o', lw = 2.25, ms=6, label = 'North')
plt.plot(data2.ages, data2.precip, color = 'k', lw = 3.)
plt.plot(data2.ages, data2.precip, color = current_palette[3], lw = 2.)

plt.plot(data3.sigma_ages, data3.deltap, color = 'k', marker = 'o', lw = 3.5, ms=8)
plt.plot(data3.sigma_ages, data3.deltap, color = 'k', marker = 'o', lw = 2.25, ms=6, label = 'North')
plt.fill_between(data3.sigma_ages, data3.deltap_l, data3.deltap_u, color = 'gray', alpha = 0.75)

plt.ylim([-0.05, 0.65])
plt.xlim([start, 0.])
plt.grid(color='slategray', linestyle=':', linewidth=1)
plt.ylabel(r'$\Delta P$ (m.w.e. a$^{-1}$)')



### Glacier lengths
##########################################################

plt.subplot(3,1,3)
plt.plot(data1.ages, data1.L, color = 'k', lw = 5)
plt.plot(data1.ages, data1.L, color = current_palette[0], lw = 4)

plt.plot(data2.ages, data2.L, color = 'k', lw = 5)
plt.plot(data2.ages, data2.L, color = current_palette[3], lw = 4)

plt.ylim([275, 360])
plt.xlim([start, 0.])
plt.grid(color='slategray', linestyle=':', linewidth=1)

plt.ylabel('Glacier Length (km)')
plt.xlabel('Age (ka BP)')

plt.tight_layout()

plt.savefig('images/ultimate.png', dpi=500)    
