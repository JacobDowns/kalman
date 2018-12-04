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
    inputs['in_dir'] = 'transform/center1/'
    inputs['y'] = np.loadtxt('paleo_inputs/y_cf.txt')
    inputs['Py'] = 0.1*np.loadtxt('paleo_inputs/Py_cf.txt')
    opt = Optimizer(inputs)
    opt.optimize(out_dir = 'transform/center1/opt1/')

if flowline == 'south1':
    inputs['in_dir'] = 'transform/south1/'
    inputs['y'] = np.loadtxt('paleo_inputs/y_sf.txt')
    inputs['Py'] = 0.1*np.loadtxt('paleo_inputs/Py_sf.txt')
    opt = Optimizer(inputs)
    opt.optimize(out_dir = 'transform/south1/opt1/')

if flowline == 'center2':
    inputs['in_dir'] = 'transform/center2/'
    inputs['y'] = np.loadtxt('paleo_inputs/y_cf.txt')
    inputs['Py'] = np.loadtxt('paleo_inputs/Py_cf.txt')
    opt = Optimizer(inputs)
    opt.optimize(out_dir = 'transform/center2/opt1/')

if flowline == 'south2':
    inputs['in_dir'] = 'transform/south2/'
    inputs['y'] = np.loadtxt('paleo_inputs/y_sf.txt')
    inputs['Py'] = np.loadtxt('paleo_inputs/Py_sf.txt')
    opt = Optimizer(inputs)
    opt.optimize(out_dir = 'transform/south2/opt1/')
