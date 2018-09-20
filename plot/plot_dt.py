import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
import matplotlib

matplotlib.rcParams.update({'font.size': 18})

### Plot dt forcings
########################################################

ages = np.linspace(-11.6e3, -200., 1000)

data = np.loadtxt('paleo_data/jensen_dye3.txt')
years = data[:,0] - 2000.0
temps = data[:,1]
delta_temp1 = interp1d(years, temps - temps[-1], kind = 'linear')(ages)

data = np.loadtxt('paleo_data/buizert_dye3.txt')
years = -data[:,0][::-1]
temps = data[:,1][::-1]
delta_temp2 = interp1d(years, temps - temps[-1], kind = 'linear')(ages)

plt.plot(ages, delta_temp1)
plt.plot(ages, delta_temp2)
plt.plot([-7.3e3, -7.3e3], [-10., 10.], 'k--')
plt.ylim([-7., 4.])
plt.show()
