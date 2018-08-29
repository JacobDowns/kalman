import numpy as np
from model.transient_runner import *
import matplotlib.pyplot as plt
import sys

"""
Perform a transient sensitivity run with given name and index.
"""

# Sensitivity experiment name
name = str(sys.argv[1])
# Run index
index = int(sys.argv[2])


### Model inputs
#######################################################

# Input dictionary
inputs = {}
# Input file name
inputs['in_file'] = 'sensitivity/' + name + '/steady_' + str(index) + '.h5'
# Time step
inputs['dt'] = 1./3.
# Number of model time steps
inputs['N'] = 4300*3


### Delta temp. function
#######################################################

data = np.loadtxt('paleo_data/jensen_dye3.txt')
# Years before present (2000)
years = data[:,0] - 2000.0
# Temps. in K
temps = data[:,1]
# Interp. delta temp. 
inputs['delta_temp_func'] = interp1d(years, temps - temps[-1], kind = 'linear')

np.set_printoptions(1, suppress = True)
print years
quit()


### Perform model run 
#######################################################

# Load the sensitivity parameters

param_sets = np.loadtxt('sensitivity/' + name + '/param_sets.txt')
params = param_sets[index,:]
inputs['b'] = params[0]
inputs['pdd_var'] = params[1]
inputs['lambda_snow'] = params[2]
inputs['lambda_ice'] = params[3]
inputs['lambda_precip'] = params[4]
inputs['P_frac'] = params[5]
inputs['beta2'] = params[6]

model_runner = TransientRunner(inputs)
ages, Ls, Hs = model_runner.run()

np.savetxt('sensitivity/' + name + '/ages_' + str(index) + '.txt', ages)
np.savetxt('sensitivity/' + name + '/Ls_' + str(index) + '.txt', Ls)
np.savetxt('sensitivity/' + name + '/Hs_' + str(index) + '.txt', Hs)
