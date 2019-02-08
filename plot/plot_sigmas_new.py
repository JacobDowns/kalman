
import numpy as np
import matplotlib.pyplot as plt


X = np.loadtxt('../transform_final/center3/X.txt')
m = np.loadtxt('../transform_final/center3/prior_m.txt')
P = np.loadtxt('../transform_final/center3/prior_P.txt')
v = 2.0*np.sqrt(np.diag(P))
Y = np.loadtxt('../transform_final/center3/Y.txt')
    

plt.subplot(2,1,1)
for i in range(len(X)):
    plt.plot(X[i])

plt.plot(m, 'k')
plt.plot(m + v, 'k')
plt.plot(m - v, 'k')

plt.subplot(2,1,2)

for i in range(len(Y)):
    plt.plot(Y[i])


plt.show()
                        
