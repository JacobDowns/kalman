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
    opt.optimize(out_dir = 'transform_start/center1/opt1/')

if flowline == 'center2':
    inputs['in_dir'] = 'transform_start/center2/'
    inputs['y_ages'] = np.loadtxt('paleo_inputs/y_ages.txt')
    inputs['y'] = np.loadtxt('paleo_inputs/y_c.txt')
    inputs['Py'] = 1.*np.loadtxt('paleo_inputs/Py_c.txt')
    opt = Optimizer(inputs)
    opt.optimize(out_dir = 'transform_start/center2/opt1/')

if flowline == 'center2_1':
    inputs['in_dir'] = 'transform_start/center2_1/'
    inputs['y_ages'] = np.loadtxt('paleo_inputs/y_ages.txt')
    inputs['y'] = np.loadtxt('paleo_inputs/y_c.txt')
    inputs['Py'] = 1.*np.loadtxt('paleo_inputs/Py_c.txt')
    opt = Optimizer(inputs)
    opt.optimize(out_dir = 'transform_start/center2_1/opt1/')

if flowline == 'center2_low1':
    inputs['in_dir'] = 'transform_start/center2_low1/'
    inputs['y_ages'] = np.loadtxt('paleo_inputs/y_ages.txt')
    inputs['y'] = np.loadtxt('paleo_inputs/y_c.txt')
    inputs['Py'] = 1.*np.loadtxt('paleo_inputs/Py_c.txt')
    opt = Optimizer(inputs)
    opt.optimize(out_dir = 'transform_start/center2_low1/opt1/')

if flowline == 'center2_high1':
    inputs['in_dir'] = 'transform_start/center2_high1/'
    inputs['y_ages'] = np.loadtxt('paleo_inputs/y_ages.txt')
    inputs['y'] = np.loadtxt('paleo_inputs/y_c.txt')
    inputs['Py'] = 1.*np.loadtxt('paleo_inputs/Py_c.txt')
    opt = Optimizer(inputs)
    opt.optimize(out_dir = 'transform_start/center2_high1/opt1/')
