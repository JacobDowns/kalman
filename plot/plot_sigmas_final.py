import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.interpolate import interp1d



plt.rcParams.update({'font.size': 18})
#plt.rc('text', usetex=True)
#plt.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]


fig, ax1 = plt.subplots(figsize=(10, 8))
current_palette = sns.color_palette("deep", 8)

x_ts = np.loadtxt('/home/jake/kalman/transform/center1/sigma_ts.txt')
X = np.loadtxt('/home/jake/kalman/transform/center1/X.txt')
Y = np.loadtxt('/home/jake/kalman/transform/center1/Y.txt') / 1000.
y_ts = np.loadtxt('/home/jake/kalman/transform/center1/age_0.txt')
indexes = [0, 10, 20, 30]

### Plot sigma points
##########################################################
j = 0
for i in indexes:
    ax1.plot(x_ts, X[i], color = 'k', linewidth = 4.5, marker = 'o', alpha = 0.85, ms = 8)
    ax1.plot(x_ts, X[i], color = current_palette[j], linewidth = 4, marker = 'o', alpha = 0.85, ms = 7)
    j += 1
ax1.set_ylim([-0.1, 0.55])
ax1.set_ylabel(r'$\Delta P$ (m.w.e. a$^{-1}$)')
ax1.set_xlabel('Age (ka BP)')
    
### Plot transformed points
##########################################################
ax2 = ax1.twinx()
j = 0
for i in indexes:
    #ax2.plot(y_ts, Y[i], color = 'k', linewidth = 4.5, dashes = (5,1), alpha = 0.85)
    ax2.plot(y_ts, Y[i], color = current_palette[j], linewidth = 4, dashes = (5,1), alpha = 0.9)
    j += 1

ax2.set_ylim([Y[0].min() - 4., Y[0].max() + 35.])
ax2.set_ylabel('Glacier Length (km)')

plt.xlim([x_ts.min(), x_ts.max()])
ticks = ax2.get_xticks()
ax2.set_xticklabels([int(abs(tick / 1000.)) for tick in ticks])
#plt.xlabel()


#plt.text(-8000., 372. + 5., r'$\pmb{\mathcal{P}_i}$', fontsize=26, horizontalalignment='center', verticalalignment='center', multialignment='center')
#plt.text(-8000., 362. + 5., r'$\downarrow$', fontsize=35, horizontalalignment='center', verticalalignment='center', multialignment='center')
#plt.text(-8000., 350. + 5., r'$\pmb{\mathcal{L}_i} = \mathcal{F}(\pmb{\mathcal{P}_i})$', fontsize=26, horizontalalignment='center', verticalalignment='center', multialignment='center')

plt.savefig('sigmas_final0.png', dpi=500)


