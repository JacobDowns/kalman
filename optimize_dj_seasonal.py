import numpy as np
from stats.optimizer import *
import sys

# Flowline
flowline = sys.argv[1]
# Input dictionary
inputs = {}
    
### Center
#############################################################

if flowline == 'center':
    inputs['in_dir'] = 'transform_dj_seasonal/center/'
    inputs['y_ages'] = np.loadtxt('paleo_inputs/y_ages.txt')
    inputs['y'] = np.loadtxt('paleo_inputs/y_c.txt')
    inputs['Py'] = .25*np.loadtxt('paleo_inputs/Py_c.txt')
    opt = Optimizer(inputs)
    opt.optimize(out_dir = 'transform_dj_seasonal/center/opt1/')

if flowline == 'center1':
    inputs['in_dir'] = 'transform_dj_seasonal/center1/'
    inputs['y_ages'] = np.loadtxt('paleo_inputs/y_ages.txt')
    inputs['y'] = np.loadtxt('paleo_inputs/y_c.txt')
    inputs['Py'] = 1.*np.loadtxt('paleo_inputs/Py_c.txt')
    opt = Optimizer(inputs)
    opt.optimize(out_dir = 'transform_dj_seasonal/center1/opt1/')

if flowline == 'south':
    inputs['in_dir'] = 'transform_dj_seasonal/south/'
    inputs['y_ages'] = np.loadtxt('paleo_inputs/y_ages.txt')
    inputs['y'] = np.loadtxt('paleo_inputs/y_s.txt')
    inputs['Py'] = .25*np.loadtxt('paleo_inputs/Py_s.txt')
    opt = Optimizer(inputs)
    opt.optimize(out_dir = 'transform_dj_seasonal/south/opt1/')

if flowline == 'south1':
    inputs['in_dir'] = 'transform_dj_seasonal/south1/'
    inputs['y_ages'] = np.loadtxt('paleo_inputs/y_ages.txt')
    inputs['y'] = np.loadtxt('paleo_inputs/y_s.txt')
    inputs['Py'] = .25*np.loadtxt('paleo_inputs/Py_s.txt')
    opt = Optimizer(inputs)
    opt.optimize(out_dir = 'transform_dj_seasonal/south1/opt1/')