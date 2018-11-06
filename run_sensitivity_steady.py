import numpy as np
from model.steady_runner import *
import matplotlib.pyplot as plt
import sys

""" 
Generate steady states for sensitivity experiment. 
"""

# Sensitivity experiment name
index = int(sys.argv[1])
# Input dictionary
inputs = {}
inputs['in_file'] = 'paleo_inputs/center_steady_seasonal.h5'
inputs['steady_file_name'] = 'transform_long/sensitivity/steady_' + str(index+1)
inputs['L_mu'] = 406893.


### Model inputs
#######################################################

#### Parameter names
params = np.loadtxt('transform_long/sensitivity/sensitivity_params.txt', dtype = str)
param_names = np.array([params, params]).flatten()
param_name = param_names[index]

### Parameter values
X = np.loadtxt('transform_long/sensitivity/X.txt')
param_vals = np.array([X[:, -6:].min(axis = 0), X[:, -6:].max(axis = 0)]).flatten()
param_val = param_vals[index]

print(param_name, param_val)
    
# Time step
inputs['dt'] = 3.
# Number of model time steps
inputs['N'] = 9000
# Tuned basal traction
inputs['beta2'] = 1.6e-3
# Start age 
inputs['start_age'] = -11.5e3
# Set the parameter
if param_name == 'A':
    inputs['b'] = (param_val*60**2*24*365)**(-1./3.)
else :
    inputs[param_name] = param_val

### Perform model run 
#######################################################

model_runner = SteadyRunner(inputs)
model_runner.run()


