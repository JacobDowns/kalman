from stats.sigma_runner import *
set_log_active(False)

### Set the run options
###############################################################

# Directory to read stuff from 
in_dir = sys.argv[1]
# Integer index
index = int(sys.argv[2])
# Sigma runner inputs
inputs = {}
# Directory to read stuff from 
inputs['in_dir'] = in_dir
# Model input file
inputs['in_file'] = in_dir + 'steady.h5'
# Integer index
inputs['index'] = index
# Number of runs
inputs['runs'] = 88

### Run some sigma points through the model
###############################################################
sr = SigmaRunner(inputs)
sr.inputs['beta2'] = 1.6e-3
sr.run()
