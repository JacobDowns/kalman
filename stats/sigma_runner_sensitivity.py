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
        self.X = np.loadtxt(self.in_dir + 'X.txt')
        self.num_sigma_points = self.X.shape[0]
        # Load sigma times
        self.sigma_ts = np.loadtxt(self.in_dir + 'sigma_ts.txt')
        self.N1 = len(self.sigma_ts)
        # Load sensitivity parameter names
        self.sensitivity_params = np.loadtxt(self.in_dir + 'sensitivity_params.txt', dtype = str)
        self.N2 = len(self.sensitivity_params)
        # Model input file
        self.input_file = input_dict['in_file']
        
        
        ### Model inputs
        #######################################################

        # Input dictionary
        self.inputs = {}
        # Time step
        self.inputs['dt'] = 1./3.
        # Number of model time steps
        self.inputs['N'] = 11590*3

        ### Use different steady states depending on the 
        #######################################################

        # The steady state index corresponding to each sigma point
        steady_indexes = np.zeros(self.X.shape[0], dtype = int)
        steady_indexes[self.X[:, -6:].argmin(axis = 0)] = range(1,7)
        steady_indexes[self.X[:, -6:].argmax(axis = 0)] = range(7,13)
        self.steady_indexes = steady_indexes

        print(self.X[:, -6:].argmin(axis = 0))
        print(self.X[:, -6:].argmax(axis = 0))

        x0 = self.X[0, :]
        x1 = self.X[102, :]
        x2 = self.X[51, :]

        plt.plot(x0[:-6])
        plt.plot(x1[:-6])
        plt.plot(x2[:-6])
        plt.show()

        #print(x1)
        #print(x2)
        
        quit()
        
        
    # Run several sigma points
    def run(self):

        # Run several delta temp. sigma points through the forward model
        for i in range(self.index*self.runs, min(self.num_sigma_points, self.index*self.runs + self.runs)):
            print(i)
           
            ### Perform model run if it hasn't been completed
            #######################################################
            
            if not os.path.isfile(self.in_dir + 'Y_' + str(i) + '.txt'):
                print("i", i)

                ### Load sigma point
                #######################################################

                # Copy the model input dictionary
                inputs = copy.deepcopy(self.inputs)
                # Input file
                inputs['in_file'] = self.in_dir + 'steady_states/steady_' + str(self.steady_indexes[i]) + '.h5'
                # Load the sigma point
                X_i = self.X[i]
                # Get the delta P function
                delta_P = X_i[0:self.N1]
                inputs['precip_param_func'] = interp1d(self.sigma_ts, delta_P, kind = 'linear')
                # Get the sensitivity param. values
                param_vals = X_i[self.N1:]

                # Set sensitivity params. 
                for j in range(self.N2):
                    if self.sensitivity_params[j] == 'A':
                        print('A', param_vals[j])
                        inputs['b'] = (param_vals[j]*60**2*24*365)**(-1./3.)
                    else :
                        inputs[self.sensitivity_params[j]] = param_vals[j]

                        
                ### Perform model run 
                #######################################################

                model_runner = TransientRunner(inputs)
                ages, Ls, Hs, Ps = model_runner.run()

                np.savetxt(self.in_dir + '/age_' + str(i) + '.txt', ages)
                np.savetxt(self.in_dir + '/Y_' + str(i) + '.txt', Ls)
                np.savetxt(self.in_dir + '/H_' + str(i) + '.txt', Hs)