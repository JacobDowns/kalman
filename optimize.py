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
    inputs['in_dir'] = 'transform_long/center1/'
    inputs['y'] = np.loadtxt('paleo_inputs/y_c.txt')
    inputs['Py'] = np.loadtxt('paleo_inputs/Py_c.txt')
    opt = Optimizer(inputs)
    opt.optimize(out_dir = 'transform_long/center1/opt1/')

if flowline == 'center1_new':
    inputs['in_dir'] = 'transform_long/center1_new/'
    inputs['y'] = np.loadtxt('paleo_inputs/y_c.txt')
    inputs['Py'] = np.loadtxt('paleo_inputs/Py_c.txt')
    opt = Optimizer(inputs)
    opt.optimize(out_dir = 'transform_long/center1_new/opt1/')

if flowline == 'center2':
    inputs['in_dir'] = 'transform_long/center2/'
    inputs['y'] = np.loadtxt('paleo_inputs/y_c.txt')
    inputs['Py'] = np.loadtxt('paleo_inputs/Py_c.txt')
    opt = Optimizer(inputs)
    opt.optimize(out_dir = 'transform_long/center2/opt1/')

if flowline == 'center1_seasonal':
    inputs['in_dir'] = 'transform_long/center1_seasonal/'
    inputs['y'] = np.loadtxt('paleo_inputs/y_c.txt')
    inputs['Py'] = np.loadtxt('paleo_inputs/Py_c.txt')
    opt = Optimizer(inputs)
    opt.optimize(out_dir = 'transform_long/center1_seasonal/opt1/')

if flowline == 'center2_seasonal':
    inputs['in_dir'] = 'transform_long/center2_seasonal/'
    inputs['y'] = np.loadtxt('paleo_inputs/y_c.txt')
    inputs['Py'] = np.loadtxt('paleo_inputs/Py_c.txt')
    opt = Optimizer(inputs)
    opt.optimize(out_dir = 'transform_long/center2_seasonal/opt1/')

    
### South
#############################################################

if flowline == 'south1':
    inputs['in_dir'] = 'transform_long/south1/'
    inputs['y'] = np.loadtxt('paleo_inputs/y_s.txt')
    inputs['Py'] = np.loadtxt('paleo_inputs/Py_s.txt')
    opt = Optimizer(inputs)
    opt.optimize(out_dir = 'transform_long/south1/opt1/')

if flowline == 'south2':
    inputs['in_dir'] = 'transform_long/south2/'
    inputs['y'] = np.loadtxt('paleo_inputs/y_s.txt')
    inputs['Py'] = np.loadtxt('paleo_inputs/Py_s.txt')
    opt = Optimizer(inputs)
    opt.optimize(out_dir = 'transform_long/south2/opt1/')

if flowline == 'south1_seasonal':
    inputs['in_dir'] = 'transform_long/south1_seasonal/'
    inputs['y'] = np.loadtxt('paleo_inputs/y_s.txt')
    inputs['Py'] = np.loadtxt('paleo_inputs/Py_s.txt')
    opt = Optimizer(inputs)
    opt.optimize(out_dir = 'transform_long/south1_seasonal/opt1/')

if flowline == 'south2_seasonal':
    inputs['in_dir'] = 'transform_long/south2_seasonal/'
    inputs['y'] = np.loadtxt('paleo_inputs/y_south.txt')
    inputs['Py'] = np.loadtxt('paleo_inputs/Py_south.txt')
    opt = Optimizer(inputs)
    opt.optimize(out_dir = 'transform_long/south2_seasonal/opt1/')
