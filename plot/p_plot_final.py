import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
import seaborn as sns
import matplotlib
matplotlib.rcParams.update({'font.size': 16})

ages = np.loadtxt('transform/center1/sigma_ts.txt')
p1 = np.loadtxt('transform/center1/opt1/opt_m.txt')
p2 = np.loadtxt('transform/center2/opt1/opt_m.txt')

plt.plot(ages, p1, 'r:')
plt.plot(ages, p2, 'r')
plt.show()
