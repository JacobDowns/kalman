import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

plt.rcParams.update({'font.size': 22})


plt.subplot(2,1,1)
opt_P = np.loadtxt('filter/prior5/opt_P1.txt')
vs = 2.0*np.sqrt(opt_P[range(len(opt_P)), range(len(opt_P))])
# Dahl Jensen delta T
m = np.loadtxt('filter/prior5/prior_m.txt')
# Optimal delta T
m_opt = np.loadtxt('filter/prior5/opt_m.txt')
# Ages
ages = np.loadtxt('opt_ages.txt')
sigma_ts = np.loadtxt('filter/prior5/sigma_ts.txt')


plt.plot(sigma_ts, m, 'k', linewidth = 2, label = r'Dahl Jensen')
plt.plot(sigma_ts, m_opt, 'r', linewidth = 2, label = 'Inversion')
plt.plot(sigma_ts, m_opt + vs, 'r:', linewidth = 2)
plt.plot(sigma_ts, m_opt - vs, 'r:', linewidth = 2)
plt.xlim(sigma_ts.min(), sigma_ts.max())

plt.ylabel(r'$\Delta T$')
plt.legend(loc = 2)

Ls = np.loadtxt('opt_L1.txt')
# L0
L0 = np.loadtxt('filter/prior5/Y_0.txt')



# Observations
obs_ages = np.array([-11.6, -10.2, -9.2, -8.2, -7.3])*1e3
obs_Ls = [406878.12855486432, 396313.20004890749, 321224.04532276397, 292845.40895793668, 288562.44342502725]
L_interp = interp1d(obs_ages, obs_Ls, kind = 'linear')

ages = ages[1:-1]

plt.subplot(2,1,2)

plt.plot(ages, Ls[1:-1], 'r', linewidth = 2, label = 'Inversion')
plt.plot(ages, L_interp(ages), 'k--', linewidth = 2)
plt.plot(obs_ages, obs_Ls, 'kx', ms = 15, linewidth = 2, label = 'Observations')
plt.plot(ages, L0[1:-1], 'k', linewidth = 2, label = 'Dahl Jensen')
plt.xlim(ages.min(), ages.max())
plt.ylabel('Glacier Length (m)')
plt.xlabel('Thousands of Years Before Present')
plt.legend()
plt.show()

#plt.plot(ages[1:-1], Ls[1:-1] - L_interp(ages[1:-1]))
#plt.show()







