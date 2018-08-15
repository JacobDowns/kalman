from inputs.paleo_inputs import *
from forward_model.forward_ice_model import *

class PaleoRunner(object):

    def __init__(self, input_dict):
        
        ### Initialize model
        ############################################################################

        # Input file
        self.in_file = input_dict['in_file']
        # Time step
        self.dt = 1./3.
        if 'dt' in input_dict:
            self.dt = input_dict['dt']
        # Output directory
        self.out_dir = 'out'
        if 'out_dir' in input_dict:
            self.out_dir = input_dict['out_dir']
        # Output file name
        self.out_file = 'paleo'
        if 'out_file' in input_dict:
            self.out_file = input_dict['paleo']
        # Model inputs
        self.model_inputs = PaleoInputs(self.in_file, dt = dt)
        # Model 
        self.model = ForwardIceModel(self.model_inputs, self.out_dir, self.out_file)
        # Delta temp as a function of age
        self.delta_temp_func = input_dict['delta_temp_func']
        # Number of time steps
        self.N = input_dict['N']
        # Starting age
        self.start_age = -11.6e3
        # Output (print stuff)?
        self.output = True
        if 'output' in input_dict:
            self.output = input_dict['output']
        

    # Perform a model run
    def run():
        # Length at each time step
        Ls = []
        # Age at each time step 
        ages = []

        for j in range(self.N):
            # Age
            age = self.start_age + model.t + dt
            ages.append(age)
            # Delta temp. 
            delta_temp = self.delta_temp_func(age)

            if self.output:
                print "delta temp.", delta_temp
                print "age", age
                print age

            L = model.step(delta_temp, accept = True)
            Ls.append(L)

        return np.array(ages), np.array(Ls)
        
