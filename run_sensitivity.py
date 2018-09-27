from stats.sigma_runner import *

### Set the run options
###############################################################

# Parameter name
param_name = str(sys.argv[1])
# param. index
param_index = int(sys.argv[2])
# Run index
run_index = int(sys.argv[3])

# Load param value
in_dir = 'transform_long/' + param_name + '_' + str(param_index) +  '/'
params = np.loadtxt('sensitivity/' + param_name +  '/params.txt')
param_value = params[param_index]

# Sigma runner inputs
inputs = {}
# Directory to read stuff from 
inputs['in_dir'] = in_dir
# Model input file
inputs['in_file'] = in_dir + 'steady.h5'
# Integer index
inputs['index'] = run_index
# Number of runs
inputs['runs'] = 2


### Delta temp. function
#######################################################
data = np.loadtxt('paleo_data/buizert_dye3.txt')
years = -data[:,0][::-1]
temps = data[:,1][::-1]
inputs['delta_temp_func'] = interp1d(years, temps - temps[-1], kind = 'linear')


### Run some sigma points through the model
###############################################################

sr = SigmaRunner(inputs)
sr.inputs[param_name] = param_value
sr.run()
