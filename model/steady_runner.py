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

        # Fixed delta temp. (Default is Buizert 11.6 temp.)
        self.delta_temp = -7.888035714285713
        if 'delta_temp' in input_dict:
            self.delta_temp = input_dict['delta_temp']
        
        # Initial precip. weight
        self.precip_param_mu = 0.
        if 'precip_param_mu' in input_dict:
            self.precip_param_mu = input_dict['precip_param_mu']

        # Initial precip param variance
        self.precip_param_sigma2 = 0.1**2
        if 'precip_param_sigma2' in input_dict:
            self.precip_param_sigma2 = input_dict['precip_param_sigma2']

        # Observation mean
        self.L_mu = input_dict['L_mu']
        
        # Observation covariance
        self.L_sigma2 = 100.**2
        if 'L_sigma2' in input_dict:
            self.L_sigma2 = input_dict['L_sigma2']
        
        # Process noise
        self.Q = 0.001**2
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
                ys[i] = self.model.step(self.delta_temp, xs[i], accept = False)

            return ys

        self.ukf = ScalarUKF(self.precip_param_mu, self.precip_param_sigma2, F, H)
        
        
    # Optimize the precip. weight
    def run(self, precip_param = 0.):
        
        # Length at each time step
        Ls = []
        # Delta temp at each time step
        precip_params = []

        for i in range(self.N):

            # Get the optimal delta temp dist. from the filter
            precip_param, precip_param_sigma2 = self.ukf.step(self.L_mu, self.L_sigma2, self.Q)
            precip_params.append(precip_param)
            
            if self.output:
                print "opt precip weight", precip_param, precip_param_sigma2

            # Do a step with the optimal param.
            L = self.model.step(self.delta_temp, precip_param, accept = True)
            Ls.append(L)

            if self.output:
                print
                print "dif", L - self.L_mu
                print

        self.model.write_steady_file(self.steady_file_name)
        
