import matplotlib.pyplot as plt
import numpy as np
from filterpy.kalman import MerweScaledSigmaPoints
from filterpy.kalman import JulierSigmaPoints
from filterpy.kalman import unscented_transform

in_dir = 'prior1'

X = np.loadtxt('jensen_sigma_points.txt')
N = sigma_points.shape[0]

ages = np.loadtxt('filter/jensen_sigmas/ages_0.txt')
#indexes = np.array([0, 1400, 2400, 3400, 4289])*3

#print len(ages[::60])
#quit()

#all_Ls = np.zeros((N, len(indexes)))
all_Ls = np.zeros((N, len(ages[::5])))

for i in range(N):
    print i
    Ls = np.loadtxt('filter/jensen_sigmas/Ls_' + str(i) + '.txt')
    all_Ls[i,:] = Ls[::5] #Ls[indexes]
    plt.plot(Ls)

np.savetxt('filter/jensen_sigmas/all_Ls.txt', all_Ls)
plt.show()

quit()

sigma_points = np.loadtxt('jensen_sigma_points.txt')
N = sigma_points.shape[0]
Ls = np.loadtxt('filter/jensen_sigmas/all_Ls.txt')
points = MerweScaledSigmaPoints(429, alpha=.1, beta=2., kappa=-1)
points = JulierSigmaPoints(429, kappa=-300.)


x, P = unscented_transform(Ls, points.weights()[0], points.weights()[1])


plt.plot(x)
plt.show()
#print points.weights()[0]

quit()
print np.dot(points.weights()[0], Ls)

#for i in range(len(Ls)):
#    plt.plot(Ls[i])

#plt.show()

