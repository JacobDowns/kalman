import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
import seaborn as sns
import matplotlib
matplotlib.rcParams.update({'font.size': 16})

ages = np.loadtxt('transform_long/south1_new/sigma_ts.txt')
p1 = np.loadtxt('transform_long/center3_minimal/opt1/opt_m.txt')
p2 = np.loadtxt('transform_long/center3_minimal/opt2/opt_m.txt')
p3 = np.loadtxt('transform_long/south3_minimal/opt1/opt_m.txt')
p4 = np.loadtxt('transform_long/south3_minimal/opt2/opt_m.txt')

#plt.plot(ages, p1)
#plt.plot(ages, p1, 'k')
#plt.plot(ages, p2, 'k--')
plt.plot(ages, p1, 'r:')
plt.plot(ages, p2, 'r')
plt.plot(ages, p3, 'k:')
plt.plot(ages, p4, 'k')
plt.show()
