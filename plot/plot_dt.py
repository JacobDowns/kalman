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


ts = np.linspace(-12600, -11554.+1., 500)
#[11, 2, 5, 8]
plt.plot(ts, dt_functions[11](ts), color = 'k', lw = 2.75)
plt.plot(ts, dt_functions[11](ts), color = current_palette[0], lw = 2.1, label = 'Dec.')
plt.plot(ts, dt_functions[2](ts), color = 'k', lw = 2.75)
plt.plot(ts, dt_functions[2](ts), color = current_palette[-3], lw = 2.1, label = 'Mar.')
plt.plot(ts, dt_functions[6](ts), color = 'k', lw = 2.75)
plt.plot(ts, dt_functions[6](ts), color = current_palette[-1], lw = 2.1, label = 'Jun.')
plt.plot(ts, dt_functions[8](ts), color = 'k', lw = 2.75)
plt.plot(ts, dt_functions[8](ts), color = current_palette[2], lw = 2.1, label = 'Sep.')
plt.plot([-12554., -12554.], [-30., 0.], 'k--')
plt.grid(color='lightgray', linestyle=':', linewidth=2.5)
#plt.xticks([-11e3, -10e3, -9e3, -8e3, -7e3])
plt.xlim([ts.min(), ts.max()])
plt.ylabel(r'$\Delta T$ ($^{\circ{}}$ C)')
#ticks = ax.get_xticks()
#ax.set_xticklabels([int(abs(tick / 1000.)) for tick in ticks])
plt.grid(True)
plt.legend(loc = 4)
plt.show()
