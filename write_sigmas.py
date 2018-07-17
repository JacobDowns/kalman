import matplotlib.pyplot as plt
import numpy as np
import os.path

in_dir = 'filter/south_prior/'

X = np.loadtxt(in_dir + 'X.txt')
Y0 = np.loadtxt(in_dir + 'Y_0.txt')
Y = np.zeros((X.shape[0], len(Y0)))

for i in range(X.shape[0]):
    print i
    Y_i = np.loadtxt(in_dir + 'Y_' + str(i) + '.txt')
    Y[i,:] = Y_i
    plt.plot(Y_i) 

np.savetxt(in_dir + 'Y.txt', Y)
plt.show()
