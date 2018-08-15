from stats.prior_writer import PriorWriter

# Flowline
flowline = sys.argv[0]
# Input dictionary
inputs = {}
# Dimension of state vector
N = 87
# Delta - controls prior smoothness
delta = 1500.

### South
##########################################################################

if flowline == 'south_prior1':
    inputs['out_dir'] = 'filter/south_prior1/'
    inputs['delta'] = delta
    inputs['N'] = N
    pw = PriorWriter(inputs)

if flowline == 'south_prior2':
    inputs['out_dir'] = 'filter/south_prior2/'
    inputs['delta'] = delta
    inputs['N'] = N
    inputs['delta_temp_file'] = 'filter/south_prior2/opt_m.txt'
    pw = PriorWriter(inputs)
    

### Center
##########################################################################

if flowline == 'center_prior1':
    inputs['out_dir'] = 'filter/center_prior1/'
    inputs['delta'] = delta
    inputs['N'] = N
    pw = PriorWriter(inputs)

if flowline == 'center_prior2':
    inputs['out_dir'] = 'filter/center_prior2/'
    inputs['delta'] = delta
    inputs['N'] = N
    inputs['delta_temp_file'] = 'filter/center_prior2/opt_m.txt'
    pw = PriorWriter(inputs)

    
### North
##########################################################################

if flowline == 'north_prior1':
    inputs['out_dir'] = 'filter/north_prior1/'
    inputs['delta'] = delta
    inputs['N'] = N
    pw = PriorWriter(inputs)

if flowline == 'north_prior2':
    inputs['out_dir'] = 'filter/north_prior2/'
    inputs['delta'] = delta
    inputs['N'] = N
    inputs['delta_temp_file'] = 'filter/north_prior2/opt_m.txt'
    pw = PriorWriter(inputs)

