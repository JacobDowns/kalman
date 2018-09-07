from stats.sigma_runner import *

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
inputs['runs'] = 3


### Delta temp. function
#######################################################
data = np.loadtxt('paleo_data/jensen_dye3.txt')
# Years before present (2000)
years = data[:,0] - 2000.0
# Temps. in K
temps = data[:,1]
# Interp. delta temp. 
inputs['delta_temp_func'] = interp1d(years, temps - temps[-1], kind = 'linear')


### Run some sigma points through the model
###############################################################
sr = SigmaRunner(inputs) 
sr.run()
