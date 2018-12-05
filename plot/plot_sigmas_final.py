import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.interpolate import interp1d




plt.rcParams.update({'font.size': 18})
#plt.rc('text', usetex=True)


#plt.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
#plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
#plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})


#plt.rc('mathtext', fontset='stixsans')
#plt.rcParams['mathtext.fontset'] = 'custom'
#plt.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
#plt.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
#plt.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'

fig, ax1 = plt.subplots(figsize=(10, 8))
current_palette = sns.color_palette("deep", 8)

x_ts = np.loadtxt('/home/jake/kalman/transform_final/center1/sigma_ts.txt')
X = np.loadtxt('/home/jake/kalman/transform_final/center1/X.txt')
Y = np.loadtxt('/home/jake/kalman/transform_final/center1/Y.txt') / 1000.
y_ts = np.loadtxt('/home/jake/kalman/transform_final/center1/age_0.txt')
indexes = [0, 10, 20, 30]

### Plot sigma points
##########################################################
j = 0
for i in indexes:
    ax1.plot(x_ts, X[i], color = 'k', linewidth = 5, marker = 'o', alpha = 0.9, ms = 8)
    ax1.plot(x_ts, X[i], color = current_palette[j], linewidth = 4, marker = 'o', alpha = 0.9, ms = 7)
    j += 1

ax1.set_yticks([0., 0.1, 0.2, 0.3])
ticks = ax1.get_yticks()
#print(ticks)
#ax1.set_yticklabels([str(tick) for tick in ticks])


ax1.set_ylim([-0.1, 0.375])
ax1.set_ylabel(r'$\Delta P$ (m.w.e. a$^{-1}$)')
ax1.set_xlabel('Age (ka BP)')


    
### Plot transformed points
##########################################################
ax2 = ax1.twinx()
j = 0
for i in indexes:
    #ax2.plot(y_ts, Y[i], color = 'k', linewidth = 4.5, dashes = (5,1), alpha = 0.85)
    ax2.plot(y_ts, Y[i], color = 'k', linewidth = 5, linestyle = '-', alpha = 0.9)
    ax2.plot(y_ts, Y[i], color = current_palette[j], linewidth = 4, linestyle = '-', alpha = 0.9)
    j += 1

ax2.set_ylim([Y[0].min() - 4., Y[0].max() + 55.])
ax2.set_ylabel('Glacier Length (km)')

plt.xlim([x_ts.min(), x_ts.max()])
ticks = ax2.get_xticks()
ax2.set_xticklabels([int(abs(tick / 1000.)) for tick in ticks])
#plt.xlabel()

ticks = ax2.get_yticks()
print(ticks)
#ax2.set_yticklabels([int(tick) for tick in ticks])


#plt.text(-6500., 372. + 5., r'$\pmb{\mathcal{P}_i}$', fontsize=26, horizontalalignment='center', verticalalignment='center', multialignment='center')
#plt.text(-6500., 360. + 5., r'$\downarrow$', fontsize=35, horizontalalignment='center', verticalalignment='center', multialignment='center')
#plt.text(-6500., 345. + 5., r'$\pmb{\mathcal{L}_i} = \mathcal{F}(\pmb{\mathcal{P}_i})$', fontsize=26, horizontalalignment='center', verticalalignment='center', multialignment='center')

plt.tight_layout()
plt.savefig('sigmas_final0.png', dpi=500)


