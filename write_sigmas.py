import matplotlib.pyplot as plt
import numpy as np
import sys
import os.path 

in_dir = sys.argv[1]
X = np.loadtxt(in_dir + 'X.txt')
Y0 = np.loadtxt(in_dir + 'Y_0.txt')
Y = np.zeros((X.shape[0], len(Y0)))

for i in range(X.shape[0]):
    if os.path.isfile(in_dir + 'Y_' + str(i) + '.txt'):
        print(i)
        Y_i = np.loadtxt(in_dir + 'Y_' + str(i) + '.txt')
        X_i = X[i, :]
        Y[i,:] = Y_i

        print(X_i[-5:])
        #print()
        print(Y_i.max())
        print ()
        plt.plot(Y_i)
    else:
        print("missing " + str(i))

np.savetxt(in_dir + 'Y.txt', Y)
plt.show()
