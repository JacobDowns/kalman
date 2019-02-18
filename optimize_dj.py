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
    inputs['in_dir'] = 'transform_dj/center1/'
    inputs['y_ages'] = np.loadtxt('paleo_inputs/y_ages.txt')
    inputs['y'] = np.loadtxt('paleo_inputs/y_c.txt')
    inputs['Py'] = .25*np.loadtxt('paleo_inputs/Py_c.txt')
    opt = Optimizer(inputs)
    opt.optimize(out_dir = 'transform_dj/center1_1/opt1/')

if flowline == 'center1_1':
    inputs['in_dir'] = 'transform_dj/center1_1/'
    inputs['y_ages'] = np.loadtxt('paleo_inputs/y_ages.txt')
    inputs['y'] = np.loadtxt('paleo_inputs/y_c.txt')
    inputs['Py'] = .25*np.loadtxt('paleo_inputs/Py_c.txt')
    opt = Optimizer(inputs)
    opt.optimize(out_dir = 'transform_dj/center1_1/opt1/')

if flowline == 'center2':
    inputs['in_dir'] = 'transform_dj/center2/'
    inputs['y_ages'] = np.loadtxt('paleo_inputs/y_ages.txt')
    inputs['y'] = np.loadtxt('paleo_inputs/y_c.txt')
    inputs['Py'] = 1.*np.loadtxt('paleo_inputs/Py_c.txt')
    opt = Optimizer(inputs)
    opt.optimize(out_dir = 'transform_dj/center2/opt1/')

if flowline == 'center2_1':
    inputs['in_dir'] = 'transform_dj/center2_1/'
    inputs['y_ages'] = np.loadtxt('paleo_inputs/y_ages.txt')
    inputs['y'] = np.loadtxt('paleo_inputs/y_c.txt')
    inputs['Py'] = 1.*np.loadtxt('paleo_inputs/Py_c.txt')
    opt = Optimizer(inputs)
    opt.optimize(out_dir = 'transform_dj/center2_1/opt1/')

    
### South
#############################################################

if flowline == 'south1':
    inputs['in_dir'] = 'transform_dj/south1/'
    inputs['y_ages'] = np.loadtxt('paleo_inputs/y_ages.txt')
    inputs['y'] = np.loadtxt('paleo_inputs/y_s.txt')
    inputs['Py'] = .25*np.loadtxt('paleo_inputs/Py_s.txt')
    opt = Optimizer(inputs)
    opt.optimize(out_dir = 'transform_dj/south1/opt1/')

if flowline == 'south1_1':
    inputs['in_dir'] = 'transform_dj/south1_1/'
    inputs['y_ages'] = np.loadtxt('paleo_inputs/y_ages.txt')
    inputs['y'] = np.loadtxt('paleo_inputs/y_s.txt')
    inputs['Py'] = .25*np.loadtxt('paleo_inputs/Py_s.txt')
    opt = Optimizer(inputs)
    opt.optimize(out_dir = 'transform_dj/south1_1/opt1/')

if flowline == 'south2_1':
    inputs['in_dir'] = 'transform_dj/south2_1/'
    inputs['y_ages'] = np.loadtxt('paleo_inputs/y_ages.txt')
    inputs['y'] = np.loadtxt('paleo_inputs/y_s.txt')
    inputs['Py'] = 1.*np.loadtxt('paleo_inputs/Py_s.txt')
    opt = Optimizer(inputs)
    opt.optimize(out_dir = 'transform_dj/south2_1/opt1/')

if flowline == 'south1_opt2':
    inputs['in_dir'] = 'transform_dj/south1/'
    inputs['y_ages'] = np.loadtxt('paleo_inputs/y_ages.txt')
    inputs['y'] = np.loadtxt('paleo_inputs/y_s.txt')
    inputs['Py'] = 1.*np.loadtxt('paleo_inputs/Py_s.txt')
    opt = Optimizer(inputs)
    opt.optimize(out_dir = 'transform_dj/south1/opt2/')

if flowline == 'south2':
    inputs['in_dir'] = 'transform_dj/south2/'
    inputs['y_ages'] = np.loadtxt('paleo_inputs/y_ages.txt')
    inputs['y'] = np.loadtxt('paleo_inputs/y_s.txt')
    inputs['Py'] = 1.*np.loadtxt('paleo_inputs/Py_s.txt')
    opt = Optimizer(inputs)
    opt.optimize(out_dir = 'transform_dj/south2/opt1/')
