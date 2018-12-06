import numpy as np
import matplotlib.pyplot as plt

ages1 = np.loadtxt('paleo_data/badgeley_ages.txt')
dts1 = np.loadtxt('paleo_data/badgeley_dts.txt')

ages2 = np.loadtxt('paleo_data/buizert_ages.txt')
dts2 = np.loadtxt('paleo_data/buizert_dts.txt')

ts = np.linspace(-11554.0, -7300., 1000)

dts1 = dts1[:,11]
dts2 = dts2[:,11]

plt.plot(ts, np.interp(ts, ages1, dts1), 'r')
plt.plot(ts, np.interp(ts, ages2, dts2), 'k')
plt.show()
