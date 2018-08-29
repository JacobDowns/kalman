import numpy as np
from model.steady_runner import *
import matplotlib.pyplot as plt
import sys

# Sensitivity experiment name
name = str(sys.argv[1])
# Run index
index = int(sys.argv[2])


### Model inputs
#######################################################

# Input dictionary
inputs = {}
# Input file name
inputs['in_file'] = 'paleo_inputs/center_paleo_steady_11_6.h5'
# Steady state name
inputs['steady_file_name'] = 'sensitivity/' + name + '/steady_' + str(index)
# Time step
inputs['dt'] = 3.
# Number of model time steps
inputs['N'] = 10000
# Initial delta temp mean
inputs['delta_temp_mu'] = -8.
# Initial delta temp variance  
inputs['delta_temp_sigma2'] = 1.
# Observation mean
inputs['L_mu'] = 406878.
# Observation variance
inputs['L_sigma2'] = 100.**2
# Process noise
inputs['Q'] = 0.1**2


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

model_runner = SteadyRunner(inputs)
print model_runner
model_runner.run()
