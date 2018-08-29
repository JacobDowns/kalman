from model.inputs.paleo_inputs1 import *
from model.forward_model.forward_ice_model import *
from model.transient_runner import *
import numpy as np
import sys
import os

class SigmaRunner(object):

    def __init__(self, input_dict):
        
        # Directory to read stuff from 
        in_dir = input_dict['in_dir']
        # Integer index
        index = input_dict['index']
        # Number of runs
        runs = input_dict['runs']
        # Load sigma points
        sigma_points = np.loadtxt(in_dir + 'X.txt')
        num_sigma_points = sigma_points.shape[0]
        # Load sigma times
        sigma_ts = np.loadtxt(in_dir + 'sigma_ts.txt')
        # Model input file
        input_file = input_dict['in_file'] 
         
        
        # Run several delta temp. sigma points through the forward model
        for i in range(index*runs, min(num_sigma_points, index*runs + runs)):
            #print i
            
            # Check to make sure this one hasn't been run before
            if not os.path.isfile(in_dir + 'Y_' + str(i) + '.txt'):

                print i
                ### Model inputs
                #######################################################

                # Input dictionary
                inputs = {}
                # Input file name
                inputs['in_file'] = input_dict['in_file']
                # Time step
                inputs['dt'] = 1./3.
                # Number of model time steps
                inputs['N'] = 4300*3
            

                ### Delta temp. function
                #######################################################
                
                # Interpolated delta temp
                X_i = sigma_points[i]
                inputs['delta_temp_func'] = interp1d(sigma_ts, X_i, kind = 'linear')
                print X_i

            
                ### Perform model run 
                #######################################################
                
                model_runner = TransientRunner(inputs)
                ages, Ls, Hs = model_runner.run()

                np.savetxt(in_dir + '/age_' + str(i) + '.txt', ages)
                np.savetxt(in_dir + '/Y_' + str(i) + '.txt', Ls)
                np.savetxt(in_dir + '/H_' + str(i) + '.txt', Hs)
