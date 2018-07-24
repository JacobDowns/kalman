import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from filterpy.kalman import JulierSigmaPoints
import seaborn as sns

plt.rcParams.update({'font.size': 22})

# Output directory
out_dir = 'filter/south_prior3/'
# Load Jensen dye3 temp.
data = np.loadtxt('paleo_data/jensen_dye3.txt')
# Years before present (2000)
years = data[:,0] - 2000.0
# Temps. in K
temps = data[:,1]
# Delta temps. 
delta_temp_interp = interp1d(years, temps - temps[-1], kind = 'linear')
# Delta temp. grid years
years = -11.6e3 + np.linspace(0., 4300, 87)


### Mean and covariance of prior
##########################################################################
N = len(years)
x = delta_temp_interp(years)
P = np.zeros((N, N))
P[range(N), range(N)] = 2.
P[range(1,N), range(N-1)] = -1.
P[range(N-1), range(1,N)] = -1.
P[N-1, N-1] = 1.
P = 1500.*P
P = np.linalg.inv(P)

np.savetxt(out_dir + 'prior_m.txt', x)
np.savetxt(out_dir + 'prior_P.txt', P)


### Plot samples from prior
##########################################################################
samples = np.random.multivariate_normal(x, P, 15)

sns.set_palette("muted")
#sns.palplot(sns.color_palette("hls", 8))
current_palette = sns.color_palette()
#sns.palplot(current_palette)

print 2*N +  1

fig, ax = plt.subplots()

for i in range(samples.shape[0]):
    sns.lineplot(years, samples[i], lw = 4)


#quit()
plt.plot(years, x, 'k', lw = 6, label = 'Dahl-Jensen')
plt.title('Prior Samples')
plt.xlim(years.min(), years.max())
plt.xlabel('Year Before Present')
plt.ylabel(r'$\Delta T$')
ticks = ax.get_xticks()
ax.set_xticklabels([int(abs(tick)) for tick in ticks])
plt.legend()
plt.show()

quit()


### Compute sigma points
##########################################################################
points = JulierSigmaPoints(N, kappa=20*len(years))

print points.weights()[0]
print points.weights()[1]

sigma_points = points.sigma_points(x, P)

plt.title('Sigma Points')
plt.xlabel('Year Before Present')
plt.ylabel(r'$\Delta T$')
plt.plot(years, sigma_points[50], 'ro-', lw = 2)
plt.plot(years, sigma_points[50 + 88], 'bo-', lw = 2)
plt.plot(years, sigma_points[0], 'ko-', lw = 2)
plt.show()

"""
indexes = range(len(points.weights()[0]))[::5]
for i in indexes:
    plt.plot(sigma_points[i])
plt.show()"""
