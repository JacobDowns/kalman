from model.forward_model.forward_ice_model import *
from model.transient_runner import *
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import copy

class SigmaRunner(object):

    def __init__(self, input_dict):
        
        # Directory to read stuff from 
        self.in_dir = input_dict['in_dir']
        # Integer index
        self.index = input_dict['index']
        # Number of runs
        self.runs = input_dict['runs']
        # Load sigma points
        self.sigma_points = np.loadtxt(self.in_dir + 'X.txt')
        self.num_sigma_points = self.sigma_points.shape[0]
        # Load sigma times
        self.sigma_ts = np.loadtxt(self.in_dir + 'sigma_ts.txt')
        # Model input file
        self.input_file = input_dict['in_file'] 

        
        ### Model inputs
        #######################################################

        # Input dictionary
        self.inputs = {}
        # Input file name
        self.inputs['in_file'] = input_dict['in_file']
        # Time step
        self.inputs['dt'] = 1./3.
        # Number of model time steps
        self.inputs['N'] = 11590*3
        # Delta temp. function
        self.inputs['delta_temp_func'] = input_dict['delta_temp_func']
        

    # Run several sigma points
    def run(self):
        # Copy the model input dictionary
        inputs = copy.deepcopy(self.inputs)

        # Run several delta temp. sigma points through the forward model
        for i in range(self.index*self.runs, min(self.num_sigma_points, self.index*self.runs + self.runs)):
            
            # Check to make sure this one hasn't been run before
            if not os.path.isfile(self.in_dir + 'Y_' + str(i) + '.txt'):
                print i

                ### Delta temp. function
                #######################################################

                # Interpolated delta temp
                X_i = self.sigma_points[i]
                inputs['precip_param_func'] = interp1d(self.sigma_ts, X_i, kind = 'linear')
                

                ### Perform model run 
                #######################################################

                model_runner = TransientRunner(inputs)
                ages, Ls, Hs, Ps = model_runner.run()

                np.savetxt(self.in_dir + '/age_' + str(i) + '.txt', ages)
                np.savetxt(self.in_dir + '/Y_' + str(i) + '.txt', Ls)
                np.savetxt(self.in_dir + '/H_' + str(i) + '.txt', Hs)
