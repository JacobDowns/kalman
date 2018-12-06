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
    inputs['in_dir'] = 'transform_uw/center1/'
    inputs['y'] = np.loadtxt('paleo_inputs/y_c_uw.txt')
    inputs['Py'] = 0.1*np.loadtxt('paleo_inputs/Py_c_uw.txt')
    opt = Optimizer(inputs)
    opt.optimize(out_dir = 'transform_uw/center1/opt1/')

if flowline == 'south1':
    inputs['in_dir'] = 'transform_final/south1/'
    inputs['y'] = np.loadtxt('paleo_inputs/y_sf.txt')
    inputs['Py'] = 0.1*np.loadtxt('paleo_inputs/Py_sf.txt')
    opt = Optimizer(inputs)
    opt.optimize(out_dir = 'transform_final/south1/opt1/')

if flowline == 'center2':
    inputs['in_dir'] = 'transform_final/center2/'
    inputs['y'] = np.loadtxt('paleo_inputs/y_c_uw.txt')
    inputs['Py'] = np.loadtxt('paleo_inputs/Py_c_uw1.txt')
    opt = Optimizer(inputs)
    opt.optimize(out_dir = 'transform_final/center2/opt1/')

if flowline == 'south2':
    inputs['in_dir'] = 'transform_final/south2/'
    inputs['y'] = np.loadtxt('paleo_inputs/y_sf.txt')
    inputs['Py'] = np.loadtxt('paleo_inputs/Py_sf.txt')
    opt = Optimizer(inputs)
    opt.optimize(out_dir = 'transform_final/south2/opt1/')
