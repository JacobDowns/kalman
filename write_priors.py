from stats.prior_writer import PriorWriter

### South prior 1
##########################################################################

inputs = {}
inputs['out_dir'] = 'filter/south_prior1/'
inputs['delta'] = 1500.
inputs['N'] = 87
pw = PriorWriter(inputs)

### South prior 2
##########################################################################

inputs = {}
inputs['out_dir'] = 'filter/south_prior2/'
inputs['delta'] = 1500.
inputs['N'] = 87
inputs['delta_temp_file'] = 'filter/south_prior1/opt_m.txt'
pw = PriorWriter(inputs)

### Center prior 1
##########################################################################

inputs = {}
inputs['out_dir'] = 'filter/center_prior1/'
inputs['delta'] = 1500.
inputs['N'] = 87
pw = PriorWriter(inputs)

### Center prior 2
##########################################################################

inputs = {}
inputs['out_dir'] = 'filter/center_prior2/'
inputs['delta'] = 1500.
inputs['N'] = 87
inputs['delta_temp_file'] = 'filter/center_prior1/opt_m.txt'
pw = PriorWriter(inputs)

### North prior 1
##########################################################################

inputs = {}
inputs['out_dir'] = 'filter/north_prior1/'
inputs['delta'] = 1500.
inputs['N'] = 87
pw = PriorWriter(inputs)

### North prior 2
##########################################################################

inputs = {}
inputs['out_dir'] = 'filter/north_prior2/'
inputs['delta'] = 1500.
inputs['N'] = 87
inputs['delta_temp_file'] = 'filter/north_prior1/opt_m.txt'
pw = PriorWriter(inputs)

