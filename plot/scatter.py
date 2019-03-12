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

### Buizert temp.
############################################################

ax = fig.add_subplot(1,1,1)
#ax.axvspan(-10., -8., alpha=0.33, color='gray')
current_palette = sns.color_palette("coolwarm", 12)

d_ages = np.loadtxt('../paleo_data/dj_ages_seasonal.txt') / 1e3
d_dts = np.loadtxt('../paleo_data/dj_dts_seasonal.txt')
d_avg = d_dts.mean(axis = 1)

d_avg_interp = interp1d(d_ages, d_avg)

data1 = DataLoader('../transform_dj_seasonal/center3/', 'opt1/')
data2 = DataLoader('../transform_dj_seasonal/south2/', 'opt1/')

data3 = DataLoader('../transform_final/center3/', 'opt1/')
data4 = DataLoader('../transform_final/south2/', 'opt1/')

#plt.plot(data1.ages, data1.precip / data1.precip[-1], color = 'k', lw = 3.)
#plt.plot(data1.ages, data1.precip / data1.precip[-1], color = current_palette[0], lw = 2.)

plt.scatter(d_avg_interp(data1.ages), data1.precip)
plt.scatter(d_avg_interp(data2.ages), data2.precip)
plt.scatter(d_avg_interp(data3.ages[10:-10]), data3.precip[10:-10])
plt.scatter(d_avg_interp(data4.ages[10:-10]), data4.precip[10:-10])

plt.xlim([0.,2])
plt.show()
