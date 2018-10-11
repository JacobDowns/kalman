import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from filterpy.kalman import JulierSigmaPoints
import seaborn as sns

plt.rcParams.update({'font.size': 16})


current_palette = np.array(sns.color_palette())
#current_palette = np.array(sns.color_palette("muted", 10)[1:])
current_palette = current_palette[[0,2,3]]
#sns.palplot(current_palette)
#plt.show()
#quit()

### Prior mean
##########################################################################

N = 45
x = np.zeros(N)
ts = (-11.6e3 + np.linspace(0., 11590, N)) / 1000.
chi = np.linspace(0., 1., len(ts))
x = 0.45*np.ones(len(ts)) - 0.45*(chi)**4


### Prior covariance
##########################################################################

delta = 250e3
# Covariance matrix
P = np.zeros((N, N))
P[range(N), range(N)] = 2.
P[range(1,N), range(N-1)] = -1.
P[range(N-1), range(1,N)] = -1.
P = delta*P
P = np.linalg.inv(P)


### Sigma points
##########################################################################

# Sigma point indexes 
#indexes = [0, 30, 45 + 7 + 15, 15, 45 + 7]
indexes = [0, 45 + 20, 15]
# Generate Julier sigma points
points = JulierSigmaPoints(N, kappa=N)
sigma_points = points.sigma_points(x, P)


### Prior plot
##########################################################################

"""
plt.subplot(3,1,1)

plt.title('(a)')

current_palette = sns.color_palette("deep", 12)
samples = np.random.multivariate_normal(x, P, 11)

for i in range(samples.shape[0]):
    plt.plot(ts, samples[i], ms = 2, linewidth = 1.5, color = current_palette[i])

plt.grid(color='slategray', linestyle=':', linewidth=0.75)
plt.xlim([ts.min(), ts.max()])
plt.ylabel(r'$\Delta P$ (m.w.e. a$^{-1}$)')"""


### Sigma points plot
##########################################################################

fig, ax1 = plt.subplots(figsize=(8,6.5))
j = 0
labels = [r'$\chi_{{{}}}$'.format(index) for index in indexes]
for i in indexes:
    print i
    ax1.plot(ts, sigma_points[i], linewidth = 2.75, marker = 'o', ms = 3,  color = current_palette[j], label = labels[j])
    j += 1

plt.xlim([ts.min(), ts.max()])
ax1.set_ylim([0., 0.475])
plt.ylabel(r'$\Delta P$ (m.w.e. a$^{-1}$)')
plt.xlabel('Age (ka BP)')

handles, labels = plt.gca().get_legend_handles_labels()
#order = [0, 3, 1, 4, 2]
#order = [0, 1, 2]
#ax1.legend([handles[idx] for idx in order],[labels[idx] for idx in order], loc = 1)
#ax1.grid(color='slategray', linestyle=':', linewidth=0.75)
#plt.text(-7., 0.25, r'$\mathcal{Y}_i = \mathcal{F}(\chi_i)$', fontsize=14)
#plt.text(-6.6, 0.275, r'$\Downarrow$', fontsize=30)
#plt.text(-6.6, 0.2, r'$\Downarrow$', fontsize=30)

plt.text(-6.2, 0.31, r'$\chi_i$', fontsize=21, horizontalalignment = 'center', verticalalignment = 'center')
plt.text(-6.4, 0.25, r'$\Downarrow$', fontsize=33, horizontalalignment = 'center', verticalalignment = 'center')
plt.text(-6.2, 0.2, r'$\mathcal{Y}_i = \mathcal{F}(\chi_i)$', fontsize=21, horizontalalignment = 'center', verticalalignment = 'center')


### Transformed sigma points plot
##########################################################################

ax2 = ax1.twinx()
ts = np.loadtxt('transform_long/center2_seasonal/age_0.txt') / 1000.
j = 0
labels = [r'$\mathcal{{Y}}_{{{}}}$'.format(index) for index in indexes]
for i in indexes:
    Ls = np.loadtxt('transform_long/center2_seasonal/Y_{}.txt'.format(i)) / 1000.
    ax2.plot(ts, Ls, linewidth = 2.75, color = current_palette[j], label = labels[j], alpha = 1., dashes = (4,1))
    j += 1
plt.xlim([ts.min(), ts.max()])
ax2.set_ylabel('Glacier Length (km)')
ax2.set_ylim([290., 410.])
#plt.grid(color='slategray', linestyle=':', linewidth=0.75)
#plt.xlabel('Age (ka BP)')
#plt.ylabel('Glacier Length (km)')
plt.tight_layout()
handles, labels = plt.gca().get_legend_handles_labels()
#ax2.legend([handles[idx] for idx in order],[labels[idx] for idx in order], loc = 3)

plt.savefig('sigmas.png', dpi=700)
