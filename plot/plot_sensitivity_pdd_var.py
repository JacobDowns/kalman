import numpy as np
import matplotlib.pyplot as plt
import os

"""
Sensitivity experiment varying pdd variance between 5 and 5.5.
"""

# Number of sensitivity experiments
num_experiments = 324
# Ages
ages = np.loadtxt('../sensitivity/ages_0.txt')
# All indexes
indexes = range(num_experiments)

### Find the parameter sets we want
#########################################################################

all_params = []
for i in indexes:
    file_name = '../sensitivity/params_' + str(i) + '.txt'
    params = np.loadtxt('../sensitivity/params_' + str(i) + '.txt')
    all_params.append(params)

all_params = np.array(all_params)

params1 = [5., 0.005, 0.008, 0.07, 0.75, 1e-3]
params2 = [5.5, 0.005, 0.008, 0.07, 0.75, 1e-3]

index1 = ((all_params - params1)**2).sum(axis = 1).argmin()
index2 = ((all_params - params2)**2).sum(axis = 1).argmin()


### Plot the sensitivity experiment results for these parameter sets
#########################################################################

indexes = [index1, index2]
for i in indexes:
    file_name = '../sensitivity/Ls_' + str(i) + '.txt'
    
    params = np.loadtxt('../sensitivity/params_' + str(i) + '.txt')
    print params
    Ls = np.loadtxt('../sensitivity/Ls_' + str(i) + '.txt')
    plt.plot(ages, Ls, label = 'pdd var = ' + str(params[0]))

plt.legend()
plt.show()