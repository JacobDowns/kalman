import numpy as np
from stats.optimizer import *
import sys

# Flowline
flowline = sys.argv[1]
# Input dictionary
inputs = {}
    
### Center
#############################################################

if flowline == 'center1':
    inputs['in_dir'] = 'transform_start/center1/'
    inputs['y_ages'] = np.loadtxt('paleo_inputs/y_ages.txt')
    inputs['y'] = np.loadtxt('paleo_inputs/y_c.txt')
    inputs['Py'] = .1*np.loadtxt('paleo_inputs/Py_c.txt')
    opt = Optimizer(inputs)
    opt.optimize(out_dir = 'transform_final/center1/opt1/')

if flowline == 'center2':
    inputs['in_dir'] = 'transform_start/center2/'
    inputs['y_ages'] = np.loadtxt('paleo_inputs/y_ages.txt')
    inputs['y'] = np.loadtxt('paleo_inputs/y_c.txt')
    inputs['Py'] = 1.*np.loadtxt('paleo_inputs/Py_c.txt')
    opt = Optimizer(inputs)
    opt.optimize(out_dir = 'transform_final/center2/opt1/')


### South
#############################################################

if flowline == 'south1':
    inputs['in_dir'] = 'transform_start/south1/'
    inputs['y_ages'] = np.loadtxt('paleo_inputs/y_ages.txt')
    inputs['y'] = np.loadtxt('paleo_inputs/y_s.txt')
    inputs['Py'] = .1*np.loadtxt('paleo_inputs/Py_s.txt')
    opt = Optimizer(inputs)
    opt.optimize(out_dir = 'transform_final/south1/opt1/')

if flowline == 'south2':
    inputs['in_dir'] = 'transform_start/south2/'
    inputs['y_ages'] = np.loadtxt('paleo_inputs/y_ages.txt')
    inputs['y'] = np.loadtxt('paleo_inputs/y_s.txt')
    inputs['Py'] = 1.*np.loadtxt('paleo_inputs/Py_s.txt')
    opt = Optimizer(inputs)
    opt.optimize(out_dir = 'transform_final/south2/opt1/')
