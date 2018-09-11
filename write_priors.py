from stats.prior_writer import PriorWriter
import sys

inputs = {}

# Directory to write prior
inputs['out_dir'] = sys.argv[1]
# Delta - controls prior smoothness
inputs['delta'] = 50e3
# Dimension of state vector
inputs['N'] = 51
# Optional prior input file
if len(sys.argv) > 2:
    inputs['x'] = sys.argv[2]
    
pw = PriorWriter(inputs)

