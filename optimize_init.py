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
    inputs['in_dir'] = 'transform_dj_seasonal/center_init/'
    inputs['y_ages'] = np.loadtxt('paleo_inputs/y_ages_init.txt')
    inputs['y'] = np.loadtxt('paleo_inputs/y_init.txt')
    inputs['Py'] = 0.1*np.loadtxt('paleo_inputs/Py_init.txt')
    opt = Optimizer(inputs)
    opt.optimize(out_dir = 'transform_dj_seasonal/center_init/opt1/')

if flowline == 'center1_init':
    inputs['in_dir'] = 'transform_dj_seasonal/center1_init/'
    inputs['y_ages'] = np.loadtxt('paleo_inputs/y_ages_init.txt')
    inputs['y'] = np.loadtxt('paleo_inputs/y_init.txt')
    inputs['Py'] = 0.1*np.loadtxt('paleo_inputs/Py_init.txt')
    opt = Optimizer(inputs)
    opt.optimize(out_dir = 'transform_dj_seasonal/center1_init/opt1/')

if flowline == 'center2_init':
    inputs['in_dir'] = 'transform_dj_seasonal/center2_init/'
    inputs['y_ages'] = np.loadtxt('paleo_inputs/y_ages_init.txt')
    inputs['y'] = np.loadtxt('paleo_inputs/y_init.txt')
    inputs['Py'] = 0.1*np.loadtxt('paleo_inputs/Py_init.txt')
    opt = Optimizer(inputs)
    opt.optimize(out_dir = 'transform_dj_seasonal/center2_init/opt1/')

if flowline == 'center3_init':
    inputs['in_dir'] = 'transform_dj_seasonal/center3_init/'
    inputs['y_ages'] = np.loadtxt('paleo_inputs/y_ages_init.txt')
    inputs['y'] = np.loadtxt('paleo_inputs/y_init.txt')
    inputs['Py'] = 0.1*np.loadtxt('paleo_inputs/Py_init.txt')
    opt = Optimizer(inputs)
    opt.optimize(out_dir = 'transform_dj_seasonal/center3_init/opt1/')
    
    
### South
#############################################################

if flowline == 'south_init':
    inputs['in_dir'] = 'transform_dj_seasonal/south_init/'
    inputs['y_ages'] = np.loadtxt('paleo_inputs/y_ages_init.txt')
    inputs['y'] = np.loadtxt('paleo_inputs/y_south_init.txt')
    inputs['Py'] = 0.1*np.loadtxt('paleo_inputs/Py_south_init.txt')
    opt = Optimizer(inputs)
    opt.optimize(out_dir = 'transform_dj_seasonal/south_init/opt1/')

if flowline == 'south1_init':
    inputs['in_dir'] = 'transform_dj_seasonal/south1_init/'
    inputs['y_ages'] = np.loadtxt('paleo_inputs/y_ages_init.txt')
    inputs['y'] = np.loadtxt('paleo_inputs/y_south_init.txt')
    inputs['Py'] = 0.1*np.loadtxt('paleo_inputs/Py_south_init.txt')
    opt = Optimizer(inputs)
    opt.optimize(out_dir = 'transform_dj_seasonal/south1_init/opt1/')

if flowline == 'south2_init':
    inputs['in_dir'] = 'transform_dj_seasonal/south2_init/'
    inputs['y_ages'] = np.loadtxt('paleo_inputs/y_ages_init.txt')
    inputs['y'] = np.loadtxt('paleo_inputs/y_south_init.txt')
    inputs['Py'] = 0.1*np.loadtxt('paleo_inputs/Py_south_init.txt')
    opt = Optimizer(inputs)
    opt.optimize(out_dir = 'transform_dj_seasonal/south2_init/opt1/')

if flowline == 'south3_init':
    inputs['in_dir'] = 'transform_dj_seasonal/south3_init/'
    inputs['y_ages'] = np.loadtxt('paleo_inputs/y_ages_init.txt')
    inputs['y'] = np.loadtxt('paleo_inputs/y_south_init.txt')
    inputs['Py'] = 0.1*np.loadtxt('paleo_inputs/Py_south_init.txt')
    opt = Optimizer(inputs)
    opt.optimize(out_dir = 'transform_dj_seasonal/south3_init/opt1/')
