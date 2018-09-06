import numpy as np
from model.steady_runner import *
import matplotlib.pyplot as plt
import sys

""" 
Generate steady states for the three flowlines.
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
if flowline == 'south_new':
    inputs['in_file'] = 'paleo_inputs/south_paleo_steady_11_6.h5'
    inputs['steady_file_name'] = 'paleo_inputs/south_new_steady'
    inputs['L_mu'] = 424778.
    inputs['P_frac'] = 0.85
if flowline == 'center':
    inputs['in_file'] = 'paleo_inputs/center_paleo_steady_11_6.h5'
    inputs['steady_file_name'] = 'paleo_inputs/center_steady'
    inputs['L_mu'] = 406893.
if flowline == 'center_new':
    inputs['in_file'] = 'paleo_inputs/center_paleo_steady_11_6.h5'
    inputs['steady_file_name'] = 'paleo_inputs/center_new_steady'
    inputs['L_mu'] = 406893.
    inputs['P_frac'] = 0.85
if flowline == 'north':
    inputs['in_file'] = 'paleo_inputs/north_paleo_steady_11_6.h5'
    inputs['steady_file_name'] = 'paleo_inputs/north_steady'
    inputs['L_mu'] = 443746.
if flowline == 'north_new':
    inputs['in_file'] = 'paleo_inputs/north_paleo_steady_11_6.h5'
    inputs['steady_file_name'] = 'paleo_inputs/north_new_steady'
    inputs['L_mu'] = 443746.
    inputs['P_frac'] = 0.85

    
### Model inputs
#######################################################

# Time step
inputs['dt'] = 3.
# Number of model time steps
inputs['N'] = 10000
# Initial delta temp mean
inputs['delta_temp_mu'] = -8.
# Initial delta temp variance  
inputs['delta_temp_sigma2'] = 1.
# Observation variance
inputs['L_sigma2'] = 100.**2
# Process noise
inputs['Q'] = 0.1**2


### Perform model run 
#######################################################

model_runner = SteadyRunner(inputs)
model_runner.run()
