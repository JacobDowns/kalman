import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
import seaborn as sns
import matplotlib
matplotlib.rcParams.update({'font.size': 18})

fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111)

current_palette =  sns.color_palette()#sns.color_palette("deep", 8)

ages = np.loadtxt('transform_uw/center1/sigma_ts.txt')
p = np.loadtxt('transform_uw/center1/opt1/opt_m.txt')


#plt.plot(ages, p)


plt.plot(ages, p, color = 'k', marker = 'o', lw = 4.5, ms=10)
plt.plot(ages, p, color = current_palette[0], marker = 'o', lw = 3.5, ms=8)
plt.plot(ages, p*0., 'k--', lw = 2)


plt.xlabel('Age (ka BP)')
plt.ylabel(r'$\Delta P$ (m.w.e. a$^{-1}$)')
plt.xlim([ages.min(), ages.max()])

plt.show()
