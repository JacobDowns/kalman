from stats

# Flowline
flowline = sys.argv[0]
# Integer index
index = int(sys.argv[1])
# Input dictionary
inputs = {}

if flowline == 'south_prior1':
    input_dict['in_dir'] = 'filter/south_prior1/'
    input_dict['index'] = index
    input_dict['runs'] = 5
    input_dict['input_file'] =  'paleo_inputs/south_paleo_steady_11_6.h5'
    
if flowline == 'south_prior2':
    input_dict['in_dir'] = 'filter/south_prior2/'
    input_dict['index'] = index
    input_dict['runs'] = 5
    input_dict['input_file'] =  'paleo_inputs/south_paleo_steady_11_6.h5'
    
if flowline == 'center_prior1':
    input_dict['in_dir'] = 'filter/center_prior1/'
    input_dict['index'] = index
    input_dict['runs'] = 5
    input_dict['input_file'] =  'paleo_inputs/center_paleo_steady_11_6.h5'
    
if flowline == 'center_prior2':
    input_dict['in_dir'] = 'filter/center_prior2/'
    input_dict['index'] = index
    input_dict['runs'] = 5
    input_dict['input_file'] =  'paleo_inputs/center_paleo_steady_11_6.h5'
    
if flowline == 'north_prior1':
    input_dict['in_dir'] = 'filter/north_prior1/'
    input_dict['index'] = index
    input_dict['runs'] = 5
    input_dict['input_file'] =  'paleo_inputs/north_paleo_steady_11_6.h5'

if flowline == 'north_prior2':
    input_dict['in_dir'] = 'filter/north_prior2/'
    input_dict['index'] = index
    input_dict['runs'] = 5
    input_dict['input_file'] =  'paleo_inputs/north_paleo_steady_11_6.h5'
    
    
    
    
    

