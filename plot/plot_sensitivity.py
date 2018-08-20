import numpy as np
import matplotlib.pyplot as plt
import os

# Number of sensitivity experiments
num_experiments = 324
ages = np.loadtxt('../sensitivity/ages_0.txt')
indexes = range(num_experiments)

all_params = []
for i in indexes:
    file_name = '../sensitivity/params_' + str(i) + '.txt'
    params = np.loadtxt('../sensitivity/params_' + str(i) + '.txt')

for i in indexes:
    file_name = '../sensitivity/Ls_' + str(i) + '.txt'
    
    if os.path.isfile(file_name):
        print i
        params = np.loadtxt('../sensitivity/params_' + str(i) + '.txt')
        print params
        Ls = np.loadtxt('../sensitivity/Ls_' + str(i) + '.txt')
        plt.plot(ages, Ls)
        plt.legend()
        L_last.append(Ls[-1])

L_last = np.array(L_last)
indexes = L_last.argsort()
print L_last[indexes]

print
print indexes
print

for j in indexes[0:30]:
    params = np.loadtxt('../sensitivity/params_' + str(j) + '.txt')
    print params

plt.show()
    
    
