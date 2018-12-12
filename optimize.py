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
    inputs['in_dir'] = 'transform_early/center1'
    inputs['y_ages'] = np.loadtxt('paleo_inputs/y_ages_f.txt')
    inputs['y'] = np.loadtxt('paleo_inputs/y_c_f.txt')
    inputs['Py'] = 0.1*np.loadtxt('paleo_inputs/Py_c_f.txt')
    opt = Optimizer(inputs)
    opt.optimize(out_dir = 'transform_early/center1/opt1/')

if flowline == 'center2':
    inputs['in_dir'] = 'transform_early/center2/'
    inputs['y_ages'] = np.loadtxt('paleo_inputs/y_ages_e2.txt')
    inputs['y'] = np.loadtxt('paleo_inputs/y_c2_e.txt')
    inputs['Py'] = np.loadtxt('paleo_inputs/Py_c2_e.txt')
    opt = Optimizer(inputs)
    opt.optimize(out_dir = 'transform_early/center2/opt1/')

if flowline == 'center1_opt2':
    inputs['in_dir'] = 'transform_early/center1/'
    inputs['y_ages'] = np.loadtxt('paleo_inputs/y_ages_e2.txt')
    inputs['y'] = np.loadtxt('paleo_inputs/y_c2_e.txt')
    inputs['Py'] = 0.25*np.loadtxt('paleo_inputs/Py_c2_e.txt')
    opt = Optimizer(inputs)
    opt.optimize(out_dir = 'transform_early/center1/opt2/')

if flowline == 'center1_opt3':
    inputs['in_dir'] = 'transform_early/center1/'
    inputs['y_ages'] = np.loadtxt('paleo_inputs/y_ages_e2.txt')
    inputs['y'] = np.loadtxt('paleo_inputs/y_c2_e.txt')
    inputs['Py'] = 0.5*np.loadtxt('paleo_inputs/Py_c2_e.txt')
    opt = Optimizer(inputs)
    opt.optimize(out_dir = 'transform_early/center1/opt3/')

if flowline == 'center1_new':
    inputs['in_dir'] = 'transform_early/center1_new/'
    inputs['y_ages'] = np.loadtxt('paleo_inputs/y_ages_e2.txt')
    inputs['y'] = np.loadtxt('paleo_inputs/y_c2_e.txt')
    inputs['Py'] = 0.1*np.loadtxt('paleo_inputs/Py_c2_e.txt')
    opt = Optimizer(inputs)
    opt.optimize(out_dir = 'transform_early/center1_new/opt1/')

if flowline == 'center1_new_opt2':
    inputs['in_dir'] = 'transform_early/center1_new/'
    inputs['y_ages'] = np.loadtxt('paleo_inputs/y_ages_f.txt')
    inputs['y'] = np.loadtxt('paleo_inputs/y_c_f.txt')
    inputs['Py'] = 0.1*np.loadtxt('paleo_inputs/Py_c_f.txt')
    opt = Optimizer(inputs)
    opt.optimize(out_dir = 'transform_early/center1_new/opt2/')

if flowline == 'center2_new':
    inputs['in_dir'] = 'transform_early/center2_new/'
    inputs['y_ages'] = np.loadtxt('paleo_inputs/y_ages_e2.txt')
    inputs['y'] = np.loadtxt('paleo_inputs/y_c2_e.txt')
    inputs['Py'] = np.loadtxt('paleo_inputs/Py_c2_e.txt')
    opt = Optimizer(inputs)
    opt.optimize(out_dir = 'transform_early/center2_new/opt1/')
