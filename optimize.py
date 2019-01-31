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
    inputs['in_dir'] = 'transform_init/center/'
    inputs['y_ages'] = np.loadtxt('paleo_inputs/y_ages_init.txt')
    inputs['y'] = np.loadtxt('paleo_inputs/y_init.txt')
    inputs['Py'] = 0.1*np.loadtxt('paleo_inputs/Py_init.txt')
    opt = Optimizer(inputs)
    opt.optimize(out_dir = 'transform_init/center/opt1/')

if flowline == 'center2_init':
    inputs['in_dir'] = 'transform_init/center2/'
    inputs['y_ages'] = np.loadtxt('paleo_inputs/y_ages_init.txt')
    inputs['y'] = np.loadtxt('paleo_inputs/y_init.txt')
    inputs['Py'] = 0.1*np.loadtxt('paleo_inputs/Py_init.txt')
    opt = Optimizer(inputs)
    opt.optimize(out_dir = 'transform_init/center2/opt1/')

if flowline == 'center3_init':
    inputs['in_dir'] = 'transform_init/center3/'
    inputs['y_ages'] = np.loadtxt('paleo_inputs/y_ages_init.txt')
    inputs['y'] = np.loadtxt('paleo_inputs/y_init.txt')
    inputs['Py'] = np.loadtxt('paleo_inputs/Py_init.txt')
    opt = Optimizer(inputs)
    opt.optimize(out_dir = 'transform_init/center3/opt1/')

if flowline == 'center6_init':
    inputs['in_dir'] = 'transform_init/center6/'
    inputs['y_ages'] = np.loadtxt('paleo_inputs/y_ages_init.txt')
    inputs['y'] = np.loadtxt('paleo_inputs/y_init.txt')
    inputs['Py'] = np.loadtxt('paleo_inputs/Py_init.txt')
    opt = Optimizer(inputs)
    opt.optimize(out_dir = 'transform_init/center6/opt1/')

    
### South
#############################################################

if flowline == 'south_init':
    inputs['in_dir'] = 'transform_init/south/'
    inputs['y_ages'] = np.loadtxt('paleo_inputs/y_ages_init.txt')
    inputs['y'] = np.loadtxt('paleo_inputs/y_south_init.txt')
    inputs['Py'] = 0.1*np.loadtxt('paleo_inputs/Py_south_init.txt')
    opt = Optimizer(inputs)
    opt.optimize(out_dir = 'transform_init/south/opt1/')

if flowline == 'south_init1':
    inputs['in_dir'] = 'transform_init/south1/'
    inputs['y_ages'] = np.loadtxt('paleo_inputs/y_ages_init.txt')
    inputs['y'] = np.loadtxt('paleo_inputs/y_south_init.txt')
    inputs['Py'] = 0.1*np.loadtxt('paleo_inputs/Py_south_init.txt')
    opt = Optimizer(inputs)
    opt.optimize(out_dir = 'transform_init/south1/opt1/')

if flowline == 'south_init2':
    inputs['in_dir'] = 'transform_init/south2/'
    inputs['y_ages'] = np.loadtxt('paleo_inputs/y_ages_init.txt')
    inputs['y'] = np.loadtxt('paleo_inputs/y_south_init.txt')
    inputs['Py'] = 0.1*np.loadtxt('paleo_inputs/Py_south_init.txt')
    opt = Optimizer(inputs)
    opt.optimize(out_dir = 'transform_init/south2/opt1/')

if flowline == 'south_init3':
    inputs['in_dir'] = 'transform_init/south3/'
    inputs['y_ages'] = np.loadtxt('paleo_inputs/y_ages_init.txt')
    inputs['y'] = np.loadtxt('paleo_inputs/y_south_init.txt')
    inputs['Py'] = 0.1*np.loadtxt('paleo_inputs/Py_south_init.txt')
    opt = Optimizer(inputs)
    opt.optimize(out_dir = 'transform_init/south3/opt1/')

if flowline == 'south_init4':
    inputs['in_dir'] = 'transform_init/south4/'
    inputs['y_ages'] = np.loadtxt('paleo_inputs/y_ages_init.txt')
    inputs['y'] = np.loadtxt('paleo_inputs/y_south_init.txt')
    inputs['Py'] = 0.1*np.loadtxt('paleo_inputs/Py_south_init.txt')
    opt = Optimizer(inputs)
    opt.optimize(out_dir = 'transform_init/south4/opt1/')
