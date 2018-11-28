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
    inputs['y'] = np.loadtxt('paleo_inputs/y_c1.txt')
    inputs['Py'] = 0.25*np.loadtxt('paleo_inputs/Py_c1.txt')
    opt = Optimizer(inputs)
    opt.optimize(out_dir = 'transform/center1/opt1/')

if flowline == 'center1_opt2':
    inputs['in_dir'] = 'transform/center1/'
    inputs['y'] = np.loadtxt('paleo_inputs/y_c2.txt')
    inputs['Py'] = 0.25*np.loadtxt('paleo_inputs/Py_c2.txt')
    opt = Optimizer(inputs)
    opt.optimize(out_dir = 'transform/center1/opt2/')

if flowline == 'center2':
    inputs['in_dir'] = 'transform/center2/'
    inputs['y'] = np.loadtxt('paleo_inputs/y_c1.txt')
    inputs['Py'] = 1.*np.loadtxt('paleo_inputs/Py_c1.txt')
    opt = Optimizer(inputs)
    opt.optimize(out_dir = 'transform/center2/opt1/')

if flowline == 'south1':
    inputs['in_dir'] = 'transform/center1/'
    inputs['y'] = np.loadtxt('paleo_inputs/y_c1.txt')
    inputs['Py'] = 0.5*np.loadtxt('paleo_inputs/Py_c1.txt')
    opt = Optimizer(inputs)
    opt.optimize(out_dir = 'transform/center1/opt1/')

