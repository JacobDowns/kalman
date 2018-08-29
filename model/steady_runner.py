from common_runner import *
import numpy as np
from stats.scalar_ukf import *
import sys
sys.path.append('../')

class SteadyRunner(CommonRunner):

    def __init__(self, input_dict):

        super(SteadyRunner, self).__init__(input_dict)
        
        ### Steady state run options
        ############################################################################

        # Steady state file name
        self.steady_file_name = input_dict['steady_file_name']
        
        # Initial delta temp mean
        self.delta_temp_mu = -8.
        if 'delta_temp_mu' in input_dict:
            self.delta_temp_mu = input_dict['delta_temp_mu']

        # Initial delta temp variance
        self.delta_temp_sigma2 = 1.
        if 'delta_temp_sigma2' in input_dict:
            self.delta_temp_sigma2 = input_dict['delta_temp_sigma2']

        # Observation mean
        self.L_mu = input_dict['L_mu']
        
        # Observation covariance
        self.L_sigma2 = 100.**2
        if 'L_sigma2' in input_dict:
            self.L_sigma2 = input_dict['L_sigma2']
        
        # Process noise
        self.Q = 0.1**2
        if 'Q' in input_dict:
            self.Q = input_dict['Q']

            
        ### Setup the UKF
        ############################################################################

        # Process model function
        def F(xs):
            return xs

        # Measurement model function
        def H(xs):
            print "H", xs
            ys = np.zeros_like(xs)

            for i in range(len(xs)):
                ys[i] = self.model.step(xs[i], accept = False)

            return ys

        self.ukf = ScalarUKF(self.delta_temp_mu, self.delta_temp_sigma2, F, H)
        
        
    # Perform a model run
    def run(self):

        # Length at each time step
        Ls = []
        # Delta temp at each time step
        delta_temps = []

        for i in range(self.N):

            # Get the optimal delta temp dist. from the filter
            delta_temp, delta_temp_sigma2 = self.ukf.step(self.L_mu, self.L_sigma2, self.Q)
            delta_temps.append(delta_temp)
            
            if self.output:
                print "opt delta temp", delta_temp, delta_temp_sigma2

            # Do a step with the optimal delta temp
            L = self.model.step(delta_temp, accept = True)
            Ls.append(L)

            if self.output:
                print
                print "dif", L - self.L_mu
                print

        self.model.write_steady_file(self.steady_file_name)
        
