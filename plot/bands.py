import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
import seaborn as sns
import matplotlib
matplotlib.rcParams.update({'font.size': 16})

ages = np.loadtxt('../transform_start/center2/sigma_ts.txt')

m1 = np.loadtxt('../transform_start/center2/opt/opt_m.txt')
P1 = np.loadtxt('../transform_start/center2/opt/opt_P.txt')
v1 = np.sqrt(np.diag(P1))

m2 = np.loadtxt('../transform_start/center2/opt1/opt_m.txt')
P2 = np.loadtxt('../transform_start/center2/opt1/opt_P.txt')
v2 = np.sqrt(np.diag(P2))


plt.plot(m1, 'k', linewidth = 3)
plt.plot(m1 - 2.*v1, 'k', linewidth = 3)
plt.plot(m1 + 2.*v1, 'k', linewidth = 3)

plt.plot(m2, 'r', linewidth = 3)
plt.plot(m2 - 2.*v2, 'r', linewidth = 3)
plt.plot(m2 + 2.*v2, 'r', linewidth = 3)

plt.show()

#np.savetxt('../transform_start/center2/opt/opt_m.txt', m)
#np.savetxt('../transform_start/center2/opt/opt_P.txt', P)


#samples = np.random.multivariate_normal(m, P, 45)


