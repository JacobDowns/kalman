import numpy as np
from model.steady_runner import *
import matplotlib.pyplot as plt
import sys

# Parameter name
param_name = str(sys.argv[1])
# Run index
index = int(sys.argv[2])
# Load the param value
in_dir = 'sensitivity/' + param_name + '/'
params = np.loadtxt(in_dir + 'params.txt')
param_value = params[index]


### Model inputs
#######################################################

# Input dictionary
inputs = {}
# Input file name
inputs['in_file'] = 'paleo_inputs/center_steady.h5'
# Steady state name
inputs['steady_file_name'] = in_dir + 'steady_' + str(index)
# Time step
inputs['dt'] = 3.
# Number of model time steps
inputs['N'] = 10000
# Desired terminus position
inputs['L_mu'] = 406893.


### Perform model run 
#######################################################

# Load the sensitivity parameters
inputs[param_name] = param_value
model_runner = SteadyRunner(inputs)
model_runner.run()
