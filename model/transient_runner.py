from common_runner import *
import numpy as np

class TransientRunner(CommonRunner):

    def __init__(self, input_dict):

        super(TransientRunner, self).__init__(input_dict)
        
        ### Transient inputs
        ############################################################################

        # Delta temp as a function of age
        self.delta_temp_func = input_dict['delta_temp_func']
        # Starting age
        self.start_age = -11.6e3
        # Snapshot interval : write out thickness vector periodically
        self.snapshot_interval = 1000
        if 'snapshot_interval' in input_dict:
            self.snapshot_interval = input_dict['snapshot_interval']
        

    # Perform a model run
    def run(self):
        # Length at each time step
        Ls = []
        # Age at each time step 
        ages = []
        # Thickness vectors through time
        Hs = []

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

            if j % self.snapshot_interval == 0:
                print "asdf"
                Hs.append(self.model.H0.vector().get_local())

        return np.array(ages), np.array(Ls), np.array(Hs)
    