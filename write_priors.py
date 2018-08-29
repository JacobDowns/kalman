from stats.prior_writer import PriorWriter
import sys

# Flowline
out_dir = sys.argv[1]
# Input dictionary
inputs = {}
# Dimension of state vector
N = 87
# Delta - controls prior smoothness
delta = 1500.

inputs = {}
inputs['out_dir'] = out_dir
inputs['delta'] = delta
inputs['N'] = N
#inputs['delta_temp_file'] = out_dir + 'opt_m.txt'
pw = PriorWriter(inputs)

