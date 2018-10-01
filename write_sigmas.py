import matplotlib.pyplot as plt
import numpy as np
import os.path

in_dir = 'transform_long/lambda_ice_1/'
X = np.loadtxt(in_dir + 'X.txt')
Y0 = np.loadtxt(in_dir + 'Y_0.txt')
Y = np.zeros((X.shape[0], len(Y0)))

for i in range(X.shape[0]):
    if os.path.isfile(in_dir + 'Y_' + str(i) + '.txt'):
        print i
        Y_i = np.loadtxt(in_dir + 'Y_' + str(i) + '.txt')
        Y[i,:] = Y_i
        plt.plot(Y_i)
    else:
        print "missing " + str(i)

np.savetxt(in_dir + 'Y.txt', Y)
plt.show()
