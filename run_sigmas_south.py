from model.stats.sigma_runner import *

# Directory to read stuff from 
in_dir = sys.argv[1]
# Integer index
index = int(sys.argv[2])

# Sigma runner inputs
inputs = {}
# Directory to read stuff from 
in_dir = input_dict['in_dir']
# Integer index
index = input_dict['index']
# Number of runs
runs = input_dict['runs']

sr = SigmaRunner(inputs)
