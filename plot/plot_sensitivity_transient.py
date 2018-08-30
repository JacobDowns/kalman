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
def plot_transient(name, index, color, linestyle, label = ''):
    ages = np.loadtxt('../sensitivity/' + name + '/ages_' + str(index) + '.txt')
    Ls = np.loadtxt('../sensitivity/' + name + '/Ls_' + str(index) + '.txt')
    plt.plot(ages, Ls, color = color, linestyle = linestyle, lw = 3.5, label = label)

plot_transient('test_lambda_ice', 0, '#d62728', '-', label = r'$\lambda_i = 0.004$')
plot_transient('test_lambda_ice', 2, '#d62728', '--', label = r'$\lambda_i = 0.006$')


plot_transient('test_lambda_snow', 0, '#2ca02c', '-', label = r'$\lambda_s = 0.0064$')
#plot_transient('test_lambda_snow', 1, 'g', '-.')
plot_transient('test_lambda_snow', 2, '#2ca02c', '--', label = r'$\lambda_s = 0.0096$')


plot_transient('test_pdd_var', 0, 'b', '-', label = r'$\sigma = 5.5C$')
#plot_transient('test_pdd_var', 1, 'b', '--')

#plot_transient('test_lambda_precip', 0, 'c', '-')
#plot_transient('test_lambda_precip', 1, 'c', '-', label = r'$\sigma_p = 5C$')

plot_transient('test_p_frac', 0, '#1f77b4', '-', label = r'$P_{frac} = 0.6$')
plot_transient('test_p_frac', 2, '#1f77b4', '--', label = r'$P_{frac} = 0.9$')

plot_transient('test_beta2', 0, 'navy', '--', label = r'$\beta^2 = 0.0008$')
plot_transient('test_beta2', 2, 'navy', '-', label = r'$\beta^2 = 0.0012$')

plot_transient('test_hardness', 0, 'y', '--', label = r'$\beta^2 = 0.0008$')
plot_transient('test_hardness', 1, 'y', '-', label = r'$\beta^2 = 0.0008$')
#plot_transient('test_hardness', 2, 'y', '-', label = r'$\beta^2 = 0.0012$')

plot_transient('test_lambda_ice', 1, 'k', '-')

plt.legend()
plt.show()