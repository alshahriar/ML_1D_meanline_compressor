#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = "Al Shahriar"
__copyright__ = "Copyright Danfoss Turbocor 2023, The Meanline ML Project"
__credits__ = ["Al Shahriar"]
__license__ = "Private"
__version__ = "1.0.0"
__maintainer__ = "Al Shahriar"
__email__ = "al.shahriar@danfoss.com"
__status__ = "Pre-production"

"""Detail description of TT_post_process.py

@Usage:
    Minimum and maximu range for each input parameters (features)
@Date: 
    July 5 2023
@Output
    Returns a numpy array with 18 parameters
"""

# %%
import numpy as np

def get_input_parameter_range():
    # stage 1
    TE_blade_ang_hub_s1 = [-60, 15]
    TE_blade_ang_tip_s1 = [-60, 15]
    inlet_inclination_ang_s1 = [-10, 10]
    number_of_main_blades_s1 = [5, 10]
    LE_Clearance_s1 = [0.0001, 0.0004]
    TE_Clearance_s1 = [0.0001, 0.0004]
    shroud_rad_s1 = [0.0181791, 0.0201791]
    inlet_hub_rad_s1 = [0.0035, 0.0055]
    impeller_outlet_rad_s1 = [0.0350, 0.04]
    impeller_outlet_width_s1 = [0.004, 0.005]
    
    # volute
    volute_throat_area = [0.0004, 0.0008]
    volute_exit_pipe_dia = [0.02, 0.04]
    
    # diffuser stage 1
    Rpin_s1 = [0.035944, 0.068609]
    Bpin_s1 = [0.002, 0.0045532]
    outlet_avg_rad_s1 = [0.045, 0.085]
    outlet_avg_width_s1 = [0.002, 0.004]
    
    # operating conditions
    rpm = [30000, 50000]
    mass_flow_rate = [0.4, 1.6]
    
    # return channel
    
    # stage 2
    TE_blade_ang_hub_s2 = [-60, 15]
    TE_blade_ang_tip_s2 = [-60, 15]
    inlet_inclination_ang_s2 = [-10, 10]
    number_of_main_blades_s2 = [5, 10]
    LE_Clearance_s2 = [0.0001, 0.0004]
    TE_Clearance_s2 = [0.0001, 0.0004]
    shroud_rad_s2 = [0.0181791, 0.0201791]
    inlet_hub_rad_s2 = [0.0035, 0.0055]
    impeller_outlet_rad_s2 = [0.0350, 0.04]
    impeller_outlet_width_s2 = [0.004, 0.005]
    
    # diffuser stage 2
    Rpin_s2 = [0.035944, 0.068609]
    Bpin_s2 = [0.002, 0.0045532]
    outlet_avg_rad_s2 = [0.045, 0.085]
    outlet_avg_width_s2 = [0.002, 0.004]
    
    # parameter ranges in correct order, comma separated
    param_ranges = np.array([TE_blade_ang_hub_s1,       # 0
                             TE_blade_ang_tip_s1,       # 1
                             inlet_inclination_ang_s1,  # 2
                             number_of_main_blades_s1,  # 3
                             LE_Clearance_s1,           # 4
                             TE_Clearance_s1,           # 5
                             shroud_rad_s1,             # 6
                             inlet_hub_rad_s1,          # 7
                             impeller_outlet_rad_s1,    # 8
                             impeller_outlet_width_s1,  # 9
    
                             volute_throat_area,        # 10
                             volute_exit_pipe_dia,      # 11
    
                             Rpin_s1,                   # 12
                             Bpin_s1,                   # 13
                             outlet_avg_rad_s1,         # 14
                             outlet_avg_width_s1,       # 15
    
                             rpm,                       # 16
                             mass_flow_rate,            # 17
                             ])
    return param_ranges