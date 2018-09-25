import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

### Load delta temp. record
########################################################################

data = np.loadtxt('paleo_data/buizert_full.txt')
years = -data[:,0][::-1]
temps_ann = data[:,1][::-1]
temps_djf = data[:,2][::-1]
temps_mam = data[:,3][::-1]
temps_jja = data[:,4][::-1]
temps_son = data[:,5][::-1]

delta_temp_ann = interp1d(years, temps_ann - temps_ann[-1], kind = 'linear')
delta_temp_djf = interp1d(years, temps_djf - temps_djf[-1], kind = 'linear')
delta_temp_mam = interp1d(years, temps_mam - temps_mam[-1], kind = 'linear')
delta_temp_jja = interp1d(years, temps_jja - temps_jja[-1], kind = 'linear')
delta_temp_son = interp1d(years, temps_son - temps_son[-1], kind = 'linear')


ages = -11.6e3 + np.linspace(0., 11600., 1000)

plt.plot(ages, delta_temp_djf(ages), 'r')
plt.plot(ages, delta_temp_mam(ages), 'g')
plt.plot(ages, delta_temp_jja(ages), 'b')
plt.plot(ages, delta_temp_son(ages), 'y')
plt.plot(ages, delta_temp_ann(ages), 'k', lw = 2)
plt.xlim(ages.min(), ages.max())
plt.show()
