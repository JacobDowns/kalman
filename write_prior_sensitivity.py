from stats.prior_writer_sensitivity import PriorWriter
import sys

inputs = {}

# Directory to write prior
inputs['out_dir'] = sys.argv[1]
# Delta - controls prior smoothness
inputs['delta'] = 250e3
# Prior input file
inputs['x'] = sys.argv[2]
    
pw = PriorWriter(inputs)
