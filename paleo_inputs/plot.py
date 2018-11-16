import matplotlib.pyplot as plt
import numpy as np

# Observation times
ts = -np.array([11.6, 10.3, 9.0, 8.1, 7.3, 0.])
# Observaion lengths
Ls = np.array([406878.128554864, 396313.200048907, 321224.045322763, 292845.408957936, 288562.443425027, 279753.709979666])
# Time standard deviation
sigmas = np.array([0.1,  0.2,  0.2, 0.3, 0.3, 0.1]) / 2.
# Learned mean
y = np.loadtxt('y_s.txt')
# Learned standard deviation
vs = np.loadtxt('yv_s.txt')
# Times for mean
N = len(y)
path_ts = np.linspace(-11.6, 0., N)

### Generate random paths
#########################################################################################

# Random samples

thing = []
for i in range(2500):
    path = y + np.random.randn(len(y))*vs

    plt.plot(path)
    L_indexes = np.where(path > Ls[4])
    # Get the time at which the moraine formed
    formation_time = path_ts[L_indexes[0].max()]
    # How far off is this formation time from the predicted time?
    #time_dif = formation_time
    thing.append(formation_time)
    #print(time_dif)
    #plt.plot(time_dif, [0.], 'ko')

#plt.show()
#quit()
plt.show()
quit()
thing = np.array(thing)
print(thing.mean())
print(np.std(thing))
quit()

plt.plot(path_ts, y, 'k', lw = 3)

plt.plot([path_ts.min(), path_ts.max()], [Ls[0], Ls[0]], 'k--', lw = 3)
plt.plot([path_ts.min(), path_ts.max()], [Ls[1], Ls[1]], 'k--', lw = 3)
plt.plot([path_ts.min(), path_ts.max()], [Ls[2], Ls[2]], 'k--', lw = 3)
plt.plot([path_ts.min(), path_ts.max()], [Ls[3], Ls[3]], 'k--', lw = 3)

plt.plot(path_ts, y + 2.0*vs, 'k', lw = 3)
plt.plot(path_ts, y - 2.0*vs, 'k', lw = 3)
plt.xlim([path_ts.min(), path_ts.max()])
plt.show()
quit()
"""
plt.plot([path_ts.min(), path_ts.max()], [Ls[4], Ls[4]], 'ko-')
plt.show()
quit()

print(thing.mean())
"""
#quit()
thing = np.array(thing)
print(thing.mean())
print(np.std(thing))
plt.hist(thing, bins = 'auto')
plt.show()
quit()
v = np.sqrt(P[range(N), range(N)])
plt.plot(path_ts, x, 'k', lw = 5)
plt.plot(path_ts, x + v, 'k--', lw = 5)
plt.plot(path_ts, x - v, 'k--', lw = 5)
plt.plot(path_ts, x + 2.*v, 'k--', lw = 5)
plt.plot(path_ts, x - 2.*v, 'k--', lw = 5)
plt.plot(ts, Ls, 'ro-', lw = 5, ms = 5)
plt.show()


