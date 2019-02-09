import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
import seaborn as sns
import matplotlib
from scipy.signal import savgol_filter
matplotlib.rcParams.update({'font.size': 18})

fig = plt.figure(figsize=(12,7))




### Buizert temp.
############################################################

ax = fig.add_subplot(111)
current_palette = sns.color_palette("coolwarm", 12)

indexes = [0, 1, 2, 3, 4, 5, 6,  7,  8, 9, 10, 11]
indexes = [0, 2, 4, 7, 8, 9, 10, 8, 6, 5, 3,  1]

dt_years = np.loadtxt('../paleo_data/buizert_ages.txt')
dt_vals = np.loadtxt('../paleo_data/buizert_dts.txt').T

dt_avg = np.zeros(len(dt_years))
for i in range(12):
    dt_smooth = savgol_filter(dt_vals[i], 91, 4)
    plt.plot(dt_years, dt_smooth, color = 'k', lw = 2, alpha = 1)
    plt.plot(dt_years, dt_smooth, color = current_palette[indexes[i]], lw = 1.4, alpha = 1.)
    dt_avg += (1./12.)*dt_smooth


plt.plot(dt_years, dt_avg, color = 'w', lw = 5.5)
plt.plot(dt_years, dt_avg, color = 'k', lw = 4)

plt.xlim([-11.6e3, 0.])
plt.ylim([-10., 3.])
plt.grid(True)
#plt.show()

plt.savefig('images/ultimate.png', dpi=500)    
