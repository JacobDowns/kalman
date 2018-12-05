import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from filterpy.kalman import JulierSigmaPoints
import seaborn as sns

plt.rcParams.update({'font.size': 16})

fig = plt.figure(figsize=(10,12))
#current_palette = sns.color_palette()
current_palette = sns.color_palette()

### Prior mean
####################################################
N = 45
#x = np.zeros(N)
ts = -11.6e3 + np.linspace(0., 11590, N)
chi = np.linspace(0., 1., len(ts))
x = 0.2*np.ones(len(ts)) - 0.2*(chi)**4

### Prior covariance
####################################################

delta = 250e3
# Covariance matrix
P = np.zeros((N, N))
P[range(N), range(N)] = 2.
P[range(1,N), range(N-1)] = -1.
P[range(N-1), range(1,N)] = -1.
P = delta*P
P = np.linalg.inv(P)

plt.subplot(2,1,1)

plt.title('(a)')

current_palette = sns.color_palette("deep", 12)
samples = np.random.multivariate_normal(x, P, 11)

for i in range(samples.shape[0]):
    plt.plot(ts, samples[i], ms = 2, linewidth = 1.5, color = current_palette[i], marker = 'o')

plt.grid(color='slategray', linestyle=':', linewidth=0.75)
plt.xlim([ts.min(), ts.max()])
plt.ylabel(r'$\Delta P$ (m.w.e. a$^{-1}$)')
#plt.show()



plt.subplot(2,1,2)

plt.title('(b)')

### Compute sigma points
##########################################################################
# Generate Julier sigma points
points = JulierSigmaPoints(N, kappa=N)
sigma_points = points.sigma_points(x, P)
linestyles = [':', '--', ':', '--', ':', '--', '--', '-']

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
