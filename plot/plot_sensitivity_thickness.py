import sys
sys.path.append('../')
from model.inputs.paleo_inputs import *
from model.forward_model.forward_ice_model import *
import matplotlib.pyplot as plt
import numpy as np
import sys
import os.path
import matplotlib

matplotlib.rcParams.update({'font.size': 20})
#fig = plt.figure(figsize=(12,14))

# Plot a steady state profile
def plot_transient(name, index, color):
    Hs = np.loadtxt('../sensitivity/' + name + '/Hs_' + str(index) + '.txt')
    for i in range(len(Hs)):
        plt.plot(Hs[i], color)
    #print Hs

plot_transient('test_p_frac', 0, 'r')
plot_transient('test_p_frac', 1, 'g')
plot_transient('test_p_frac', 2, 'b')

plt.legend()
plt.show()
