import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from filterpy.kalman import JulierSigmaPoints
import seaborn as sns

plt.rcParams.update({'font.size': 18})

fig, ax1 = plt.subplots(figsize=(12, 10))
#fig = plt.figure(figsize=(10,12))
current_palette = sns.color_palette()
#sns.palplot(current_palette)
#plt.show()
#current_palette = np.array(current_palette)
#current_palette = current_palette[[0,1,2,3,7,9]]
#current_palette = sns.color_palette("dark", 10)

x_ts = np.loadtxt('/home/jake/kalman/transform_long/center1_seasonal/sigma_ts.txt')
X = np.loadtxt('/home/jake/kalman/transform_long/center1_seasonal/X.txt')
Y = np.loadtxt('/home/jake/kalman/transform_long/center1_seasonal/Y.txt') / 1000.
y_ts = np.loadtxt('/home/jake/kalman/transform_long/center1_seasonal/age_0.txt')
indexes = [0, 30, 45+22, 15, 45+7]

### Plot sigma points
##########################################################
j = 0
for i in indexes:
    ax1.plot(x_ts, X[i], color = current_palette[j], linewidth = 3, marker = 'o', alpha = 0.85)
    j += 1
ax1.set_ylim([-0.1, 0.48])
ax1.set_ylabel(r'$\Delta P$ (m.w.e. a$^{-1}$)')
ax1.set_xlabel('Age (ka BP)')
    
### Plot transformed points
##########################################################
ax2 = ax1.twinx()
j = 0
for i in indexes:
    ax2.plot(y_ts, Y[i], color = current_palette[j], linewidth = 4, dashes = (5,1), alpha = 0.85)
    j += 1

ax2.set_ylim([Y[0].min() - 4., Y[0].max() + 7.])
ax2.set_ylabel('Glacier Length (km)')


plt.xlim([x_ts.min(), x_ts.max()])
ticks = ax2.get_xticks()
ax2.set_xticklabels([int(abs(tick / 1000.)) for tick in ticks])
#plt.xlabel()

plt.text(-5500., 372. + 5., r'$\mathcal{P}_i$', fontsize=25, horizontalalignment='center', verticalalignment='center', multialignment='center')
plt.text(-5600., 365. + 5., r'$\downarrow$', fontsize=32, horizontalalignment='center', verticalalignment='center', multialignment='center')
plt.text(-5500., 358. + 5., r'$\mathcal{L}_i = \mathcal{F}(\mathcal{P}_i)$', fontsize=25, horizontalalignment='center', verticalalignment='center', multialignment='center')


plt.savefig('sigmas.png', dpi=700)



quit()






plt.subplot(2,1,2)

plt.title('(b)')

### Compute sigma points
##########################################################################
# Generate Julier sigma points
points = JulierSigmaPoints(N, kappa=N)
sigma_points = points.sigma_points(x, P)


j = 0
#[0, N+28, 24, N+20, 16, N+12, 8, N + 4]



#indexes = np.array(zip([0, 15, 30], [45 + 15, 45 + 30])).flatten()  #[0, N+36, 27, N+18, 18]
indexes = [0, 15, 45 + 15, 30, 45 + 30]
print indexes
#quit()
labels = [r'$\chi_{{{}}}$'.format(index) for index in indexes]
alphas = [0.9, 0.9, 0.9, 0.9, 0.9]
dashes = [(5, 0), (5, 0), (5, 0), (5, 0), (5, 0)]

for i in indexes:
    print i
    plt.plot(ts, sigma_points[i], linewidth = 1.5, marker = 'o', ms = 2,  color = current_palette[j], label = labels[j])
    j += 1

plt.xlim([ts.min(), ts.max()])
#plt.ylim([-0.021, 0.021])
#plt.show()
plt.ylabel(r'$\Delta P$ (m.w.e. a$^{-1}$)')

#handles, labels = plt.gca().get_legend_handles_labels()
#order = [0, 4, 2, 3, 1]
#plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order], loc = 3)
plt.legend()

#plt.yticks([-0.02, -0.01, 0., 0.01, 0.02])
#plt.xticks([-10., -6., -2.])
#plt.grid(True)
plt.grid(color='slategray', linestyle=':', linewidth=0.75)
plt.xlabel('Age (ka BP)')

plt.tight_layout()
plt.savefig('sigmas.png', dpi=700)
