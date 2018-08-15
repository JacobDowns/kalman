from inputs.paleo_inputs import *
from forward_model.forward_ice_model import *

class PaleoRunner(object):

    def __init__(self, input_dict):
        
        ### Model inputs
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

            
        ### PDD params
        ############################################################################
        
        # PDD variance
        self.pdd_var = 5.5
        if 'pdd_var' in input_dict:
            self.pdd_var = input_dict['pdd_var']
        # Ablation rate snow
        self.lambda_snow = 0.005
        if 'lambda_snow' in input_dict:
            self.lambda_snow = input_dict['lambda_snow']
        # Ablation rate ice 
        self.lambda_ice = 0.008
        if 'lambda_ice' in input_dict:
            self.lambda_ice = input_dict['lambda_ice']
        # Precip. factor
        self.lambda_precip = 0.07
        if 'lambda_precip' in input_dict:
            self.lambda_precip = input_dict['lambda_precip']


        ### Sliding parameters
        ############################################################################

        # Overburden pressure fraction
        self.P_frac = input_dict['P_frac'] 
        # Basal traction
        self.beta2 = input_dict['beta2']
        

        ### Init. model
        ############################################################################
        
        # Model inputs
        self.model_inputs = PaleoInputs(self.in_file, dt = self.dt, pdd_var = self.pdd_var,
                                        lambda_snow = self.lambda_snow, lambda_ice = self.lambda_ice, lambda_precip = self.lambda_precip)
        # Model 
        self.model = ForwardIceModel(self.model_inputs, self.out_dir, self.out_file)
        # Assign P_frac
        self.model.P_frac.assign(self.P_frac)
        # Assign beta2
        self.model.beta2.interpolate(Constant(self.beta2))
        

    # Perform a model run
    def run(self):
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
        
