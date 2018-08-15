import numpy as np
import matplotlib.pyplot as plt

# Number of sensitivity experiments
num_experiments = 324

ages = np.loadtxt('../sensitivity/ages_0.txt')

for i in range(num_experiments):
    print i
    Ls = np.loadtxt('../sensitivity/Ls_' + str(i) + '.txt')
    plt.plot(ages, Ls)

plt.show()
    
    
