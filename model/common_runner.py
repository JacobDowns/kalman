from inputs.paleo_inputs import *
from forward_model.forward_ice_model import *

class CommonRunner(object):

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
            
        # Number of time steps
        self.N = input_dict['N']
        # Output (print stuff)?
        self.output = True
        if 'output' in input_dict:
            self.output = input_dict['output']
            
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
            
        # Basal traction
        self.beta2 = None
        if 'beta2' in input_dict:
            self.beta2 = input_dict['beta2']

        # Ice hardness
        self.b = None
        if 'b' in input_dict:
            self.b = input_dict['b']
            
        
        ### Init. model
        ############################################################################
        
        # Model input dictionary
        inputs = {}
        inputs['dt'] = self.dt
        inputs['pdd_var'] = self.pdd_var
        inputs['lambda_snow'] = self.lambda_snow
        inputs['lambda_ice'] = self.lambda_ice
        inputs['lambda_precip'] = self.lambda_precip
        inputs['beta2'] = self.beta2

        #print inputs
        #quit()

        # Create model inputs
        self.model_inputs = PaleoInputs(self.in_file, inputs)

        
        ### Set constants
        ############################################################################
        
        # Ice hardness
        if 'b' in input_dict:
            print self.model_inputs.physical_constants['b']
            print input_dict['b']
            #quit()
            self.model_inputs.physical_constants['b'] = input_dict['b']
        # Overburden pressure fraction
        if 'P_frac' in input_dict:
            self.model_inputs.physical_constants['P_frac'] = input_dict['P_frac']
        # Model
        print self.model_inputs.physical_constants
        self.model = ForwardIceModel(self.model_inputs, self.out_dir, self.out_file)
        
        
