import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from filterpy.kalman import JulierSigmaPoints
import seaborn as sns

plt.rcParams.update({'font.size': 16})

fig = plt.figure(figsize=(10,6))
current_palette = sns.color_palette()
#sns.palplot(current_palette)
#plt.show()
#quit()
#print current_palette
#quit()


### Prior mean
####################################################
N = 45
x = np.zeros(N)
ts = np.linspace(-11.6, 0., N)

### Prior covariance
####################################################

delta = 250e3
# Covariance matrix
P = np.zeros((N, N))
P[range(N), range(N)] = 2.
P[range(1,N), range(N-1)] = -1.
P[range(N-1), range(1,N)] = -1.
P[N-1, N-1] = 2.
P = delta*P
P = np.linalg.inv(P)

### Compute sigma points
##########################################################################
# Generate Julier sigma points
points = JulierSigmaPoints(N, kappa=N)
sigma_points = points.sigma_points(x, P)
linestyles = [':', '--', ':', '--', ':', '--', '--', '-']
colors = [current_palette[0], current_palette[1], current_palette[2], 'k', current_palette[1]] 

j = 0
#[0, N+28, 24, N+20, 16, N+12, 8, N + 4]

#[0, 24, N+20, 8, N + 4]
labels = [r'$\chi_0$', r'$\chi_{34}$', r'$\chi_{67}$', r'$\chi_{11}$', r'$\chi_{49}$']
alphas = [1., 0.7, .55, .4, .6]
dashes = [(5, 0), (5, 0), (5, 0), (5, 0), (5, 4)]

for i in [0, 34, N+22, 11]:
    print i
    plt.plot(ts, sigma_points[i], linewidth = 3, marker = 'o', ms = 4, alpha = alphas[j], dashes = dashes[j], color = colors[j], label = labels[j])
    j += 1

plt.xlim([ts.min(), ts.max()])
#plt.ylim([-0.021, 0.021])
#plt.show()
plt.ylabel(r'$\Delta P$ (m.w.e. a$^{-1}$)')
plt.xlabel('Age (ka BP)')
plt.legend(loc = 4)
plt.yticks([-0.02, -0.01, 0., 0.01, 0.02])
#plt.xticks([-10., -6., -2.])
#plt.grid(True)
plt.grid(color='slategray', linestyle=':', linewidth=0.75)
plt.tight_layout()
plt.savefig('sigmas.png', dpi=700)

