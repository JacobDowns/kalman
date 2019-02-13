import numpy as np
from stats.optimizer import *
import sys

# Flowline
flowline = sys.argv[1]
# Input dictionary
inputs = {}
    
### Center
#############################################################

if flowline == 'center_init':
    inputs['in_dir'] = 'transform_dj/center1/'
    inputs['y_ages'] = np.loadtxt('paleo_inputs/y_ages_init.txt')
    inputs['y'] = np.loadtxt('paleo_inputs/y_init.txt')
    inputs['Py'] = 0.1*np.loadtxt('paleo_inputs/Py_init.txt')
    opt = Optimizer(inputs)
    opt.optimize(out_dir = 'transform_dj/center1/opt1/')

if flowline == 'center_init1':
    inputs['in_dir'] = 'transform_dj/center2/'
    inputs['y_ages'] = np.loadtxt('paleo_inputs/y_ages_init.txt')
    inputs['y'] = np.loadtxt('paleo_inputs/y_init.txt')
    inputs['Py'] = 0.1*np.loadtxt('paleo_inputs/Py_init.txt')
    opt = Optimizer(inputs)
    opt.optimize(out_dir = 'transform_dj/center2/opt1/')

    
### South
#############################################################

if flowline == 'south_init':
    inputs['in_dir'] = 'transform_dj/south1/'
    inputs['y_ages'] = np.loadtxt('paleo_inputs/y_ages_init.txt')
    inputs['y'] = np.loadtxt('paleo_inputs/y_south_init.txt')
    inputs['Py'] = 0.1*np.loadtxt('paleo_inputs/Py_south_init.txt')
    opt = Optimizer(inputs)
    opt.optimize(out_dir = 'transform_dj/south1/opt1/')

if flowline == 'south_init1':
    inputs['in_dir'] = 'transform_dj/south2/'
    inputs['y_ages'] = np.loadtxt('paleo_inputs/y_ages_init.txt')
    inputs['y'] = np.loadtxt('paleo_inputs/y_south_init.txt')
    inputs['Py'] = 0.1*np.loadtxt('paleo_inputs/Py_south_init.txt')
    opt = Optimizer(inputs)
    opt.optimize(out_dir = 'transform_dj/south2/opt1/')
