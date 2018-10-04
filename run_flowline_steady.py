import numpy as np
from model.steady_runner import *
import matplotlib.pyplot as plt
import sys

""" 
Generate a steady state for each flowline by fixing delta temp. and adjusting precipitation to 
achieve the desired terminus position. 
"""

# Sensitivity experiment name
flowline = str(sys.argv[1])
# Input dictionary
inputs = {}

# Input file name
if flowline == 'south':
    inputs['in_file'] = 'paleo_inputs/south_paleo_steady_11_6.h5'
    inputs['steady_file_name'] = 'paleo_inputs/south_steady'
    inputs['L_mu'] = 424778.
if flowline == 'center':
    inputs['in_file'] = 'paleo_inputs/center_paleo_steady_11_6.h5'
    inputs['steady_file_name'] = 'paleo_inputs/center_steady'
    inputs['L_mu'] = 406893.
if flowline == 'north':
    inputs['in_file'] = 'paleo_inputs/north_paleo_steady_11_6.h5'
    inputs['steady_file_name'] = 'paleo_inputs/north_steady'
    inputs['L_mu'] = 443746.

    
### Model inputs
#######################################################

# Time step
inputs['dt'] = 3.
# Number of model time steps
inputs['N'] = 10000
# Tuned basal traction
inputs['beta2'] = 1.6e-3
# Start age 
inputs['start_age'] = -11.5e3


### Perform model run 
#######################################################

model_runner = SteadyRunner(inputs)
model_runner.run()
