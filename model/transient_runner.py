from common_runner import *
import numpy as np
from stats.scalar_ukf import *


class TransientRunner(object):

    def __init__(self, input_dict):
        
        ### Transient inputs
        ############################################################################

        # Delta temp as a function of age
        self.delta_temp_func = input_dict['delta_temp_func']
        # Starting age
        self.start_age = -11.6e3
        

    # Perform a model run
    def run(self):

        # Process model function
        def F(xs):
            return xs

        # Measurement model function
        def H(xs):
            print "H", xs
            ys = np.zeros_like(xs)

            for i in range(len(xs)):
                ys[i] = model.step(xs[i], accept = False)

            return ys

        ukf = ScalarUKF(delta_temp_mu, delta_temp_sigma2, F, H)
        
        # Length at each time step
        Ls = []
        # Age at each time step 
        ages = []

        for j in range(self.N):
            # Age
            age = self.start_age + self.model.t
            ages.append(age)
            # Delta temp. 
            delta_temp = self.delta_temp_func(age)

            if self.output:
                print "delta temp.", delta_temp
                print "age", age
                print age

            L = self.model.step(delta_temp, accept = True)
            Ls.append(L)

        return np.array(ages), np.array(Ls)
        
