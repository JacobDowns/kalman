from dolfin import *
import numpy as np
from common_inputs import *
from scipy.interpolate import interp1d
from pdd_calculator import PDDCalculator
import matplotlib.pyplot as plt
import sys

"""
Inputs for paleo run.
"""

class PaleoInputs(CommonInputs):

    def __init__(self, input_file_name, input_dict):

        ### Set input parameters
        ########################################################################

        # Time step
        self.dt = 1.
        if 'dt' in input_dict:
            self.dt = input_dict['dt']
        
        # PDD variance
        self.pdd_var = 5.5
        if 'pdd_var' in input_dict:
            self.pdd_var = input_dict['pdd_var']

        # Ablation rate for snow (m / (degree C * day))
        self.lambda_snow = 0.005
        if 'lambda_snow' in input_dict:
            self.lambda_snow = input_dict['lambda_snow']

        # Ablation rate for ice (m / (degree C * day))
        self.lambda_ice = 0.008
        if 'lambda_ice' in input_dict:
            self.lambda_ice = input_dict['lambda_ice']

        # Precipitation parameter
        self.lambda_precip = 0.07
        if 'lambda_precip' in input_dict:
            self.lambda_precip = input_dict['lambda_precip']

        # Superimposed ice fraction
        self.super_ice_frac = 0.6
        if 'super_ice_frac' in input_dict:
            self.super_ice_frac = input_dict['super_ice_frac']

        # Elevation lapse rate (degrees C / km)
        self.lapse_rate = 5.
        if 'lapse_rate' in input_dict:
            self.lapse_rate = input_dict['lapse_rate']

        # Basal traction
        self.beta2 = None
        if 'beta2' in input_dict:
            self.beta2 = input_dict['beta2']

        # Start age
        self.start_age = -11.6e3
        if 'start_age' in input_dict:
            self.start_age = input_dict['start_age']
            
        
        ### Load monthly modern temp. and precip. fields
        ########################################################################

        additional_cg_fields = ['T' + str(i) for i in range(12)] \
         + ['P' + str(i) for i in range(12)] + ['S_ref']
        additional_interp_fields = additional_cg_fields

        input_options = {
            'additional_cg_fields' : additional_cg_fields,
            'additional_interp_fields' : additional_interp_fields
        }

        self.additional_cg_fields = additional_cg_fields

        # Override beta2 from the .h5 file?
        if self.beta2:
            input_options['beta2'] = self.beta2

        super(PaleoInputs, self).__init__(input_file_name, input_options)

        # Initialize the inputs with the correct initial length
        self.update_interp_all(float(self.input_functions['L0']))
        # Surface mass blance function
        self.adot = Function(self.V_cg)
        # Precipitation for the given year (for plotting) in m/a
        self.precip_func = Function(self.V_cg)
        # Temperature for the given year (for plotting) in C
        self.temp = Function(self.V_cg)
        # Initial glacier length
        self.L_init = float(self.input_functions['L0'])
        # Object for calculating PDD's
        self.pdd_calc = PDDCalculator(self.pdd_var)

        
        ### Load delta temp. record
        ########################################################################

        # What dt reconstruction to use?
        self.delta_temp_record = 'buizert'
        # Use seasonal or annual delta temp. record?
        self.seasonality = True
        
        #if self.delta_temp_record == 'buizert' and 'seasonality' in input_dict:
        #    self.seasonality = input_dict['seasonality']

        if self.delta_temp_record == 'buizert':
            data = np.loadtxt('paleo_data/buizert_full.txt')
            years = -data[:,0][::-1]
            temps_ann = data[:,1][::-1]
            temps_djf = data[:,2][::-1]
            temps_mam = data[:,3][::-1]
            temps_jja = data[:,4][::-1]
            temps_son = data[:,5][::-1]

            self.delta_temp_ann = interp1d(years, temps_ann - temps_ann[-1], kind = 'linear')
            self.delta_temp_djf = interp1d(years, temps_djf - temps_djf[-1], kind = 'linear')
            self.delta_temp_mam = interp1d(years, temps_mam - temps_mam[-1], kind = 'linear')
            self.delta_temp_jja = interp1d(years, temps_jja - temps_jja[-1], kind = 'linear')
            self.delta_temp_son = interp1d(years, temps_son - temps_son[-1], kind = 'linear')
        else:
            data = np.loadtxt('paleo_data/jensen_dye3.txt')
            years = data[:,0] - 2000.0
            temps = data[:,1]
            
            self.delta_temp_ann = interp1d(years, temps_ann - temps_ann[-1], kind = 'linear')

        
    """
    Adot expression used by the model.
    """
    def get_adot_exp(self, S):
        # Just return local copy of adot that gets updated as elevation changes
        self.modeled_S = S
        return self.adot


    """
    Recompute SMB as time and ice surface change.
    """
    def update_adot(self, t, precip_param = 0.0, delta_temp = None):
        age = self.start_age + t

        ### Delta temp. to use for each month
        ########################################################################

        if delta_temp:
            dt_djf = delta_temp
            dt_mam = delta_temp
            dt_jja = delta_temp
            dt_son = delta_temp            
        elif self.delta_temp_record == 'buizert' and self.seasonality:
            dt_djf = self.delta_temp_djf(age)
            dt_mam = self.delta_temp_mam(age)
            dt_jja = self.delta_temp_jja(age)
            dt_son = self.delta_temp_son(age)
        else :
            dt_ann = self.delta_temp_ann(age)
            dt_djf = dt_ann
            dt_mam = dt_ann
            dt_jja = dt_ann
            dt_son = dt_ann   
                 
        monthly_dts = [dt_djf, dt_djf, dt_djf, dt_mam, dt_mam, dt_mam,\
                   dt_jja, dt_jja, dt_jja, dt_son, dt_son, dt_son]

        print ("Age", age)
        print ("Delta", dt_djf, dt_mam, dt_jja, dt_son)
        print ("Precip", precip_param)
            

        ### Compute monthly pdd's and precip.
        ########################################################################
        
        # Get the reference elevation used by climate model
        ref_elevation_vec = self.input_functions['S_ref'].vector().get_local()
        # Get the modeled elevation
        modeled_elevation_vec = self.modeled_S.vector().get_local()
        # Compute the lapse rate correction in C
        lapse_correction = ((ref_elevation_vec - modeled_elevation_vec) / 1000.0) * self.lapse_rate
        # Total snow that has fallen for the year
        total_snowfall = np.zeros_like(self.input_functions['S_ref'].vector().get_local())
        # Total number of pdds for the year
        total_pdds = np.zeros_like(self.input_functions['S_ref'].vector().get_local())
        
        for i in range(12):
            # Compute the delta temp. adjusted / lapse rate corrected temp. for this month
            # Modern temp.  and precip. for a given month are computed as a 30 year modern average from Box
            modern_temp_vec = self.input_functions['T' + str(i)].vector().get_local()
            temp_vec = modern_temp_vec + monthly_dts[i] + lapse_correction
            # Compute the delta temp. adjusted precip.
            modern_precip_vec = self.input_functions['P' + str(i)].vector().get_local()
            # Temp. corrected precip. rate in m.w.e./a
            precip_vec = modern_precip_vec*np.e**(self.lambda_precip*(temp_vec - modern_temp_vec)) + precip_param
            # Compute pdd's for this month
            total_pdds += self.pdd_calc.get_pdd(temp_vec)
            # Fraction of precip. that falls as snow
            snowfall_frac = self.pdd_calc.get_acc_frac(temp_vec)
            # Compute snowfall for the month in m.w.e
            total_snowfall += precip_vec * (1./12.) * snowfall_frac

        # Save total snowfall for plotting
        self.precip_func.vector()[:] = total_snowfall
        

        ### Compute SMB from total snowfall and pdds
        ########################################################################
        
        # PDD's needed to melt given fraction of the snowpack 
        pdds_to_make_super_ice = (self.super_ice_frac*total_snowfall) / self.lambda_snow
        # PDD's that actually go to making superimposed ice
        pdds_super_ice = np.minimum(total_pdds, pdds_to_make_super_ice)
        total_pdds -= pdds_super_ice
        # Amount of superimposed ice in m.w.e
        super_ice = pdds_super_ice * self.lambda_snow
        # Amount of snow in m.w.e remaining after some has been converted to superimposed ice
        total_snowfall -= super_ice        
        # PDD's needed to melt all the remaining snow
        pdds_to_melt_snow = total_snowfall / self.lambda_snow
        # PDD's that actually go to melting snow
        pdds_melt_snow = np.minimum(total_pdds, pdds_to_melt_snow)
        total_pdds -= pdds_melt_snow
        # Amount of snow that remains in m.w.e
        total_snowfall -= pdds_melt_snow * self.lambda_snow
        # PDD's needed to melt the superimposed ice
        pdds_to_melt_super_ice = super_ice / self.lambda_ice
        # PDD's that actually go to melting superimposed ice
        pdds_melt_super_ice = np.minimum(total_pdds, pdds_to_melt_super_ice)
        total_pdds -= pdds_melt_super_ice
        # The amount of superimposed ice remaining
        super_ice -= pdds_melt_super_ice * self.lambda_ice
        # Compute the accumulation in m.w.e, consisting of snow and superimpsed ice
        accumulation = total_snowfall + super_ice
        # Compute the amount of ablation in m.w.e (remaining PDD's go to melting glacier ice)
        ablation = total_pdds * self.lambda_ice
        # Total yearly mass balance in m.i.e. assuming snowpack turns to ice at end of year
        smb = (accumulation - ablation) * (10./9.)
        self.adot.vector()[:] = smb
        

    # Update inputs that change with glacier length and time
    def update_inputs(self, L, age, precip_param = 0.0):
        self.update_interp_all(L)
        self.update_adot(age, precip_param)
