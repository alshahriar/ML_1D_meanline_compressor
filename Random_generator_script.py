import numpy as np
import pandas as pd
import os
import time
import json
"""
 Order of input parameters, use proper TurboTides variables paths!
 Keep this order consistent through param_ranges, parameters, and units
 ######################################### Add parameters as needed ####################################################
 1.  1d/1d_Machine/stage1/impeller/blade/hubSect/beta2b
 2.  1d/1d_Machine/stage1/impeller/blade/tipSect/beta2b
 3.  1d/1d_Machine/stage1/impeller/in/phi
 4.  1d/1d_Machine/stage1/impeller/blade/numMainBlades
 5.  1d/1d_Machine/stage1/impeller/blade/LE/clr
 6.  1d/1d_Machine/stage1/impeller/blade/TE/clr
 7.  1d/1d_Machine/stage1/impeller/in/Rs
 8.  1d/1d_Machine/stage1/impeller/in/Rh
 9.  1d/1d_Machine/stage1/impeller/out/Ra
 10. 1d/1d_Machine/stage1/impeller/out/b
 11. 1d/1d_Machine/stage1/volute/throatArea
 12. 1d/1d_Machine/stage1/volute/exitPipeDiameter
 13. 1d/1d_Machine/stage1/vaneless/Rpin_Rin
 14. 1d/1d_Machine/stage1/vaneless/Bpin_Bin
 15. 1d/1d_Machine/stage1/vaneless/Rout_Rin
 16. 1d/1d_Machine/stage1/vaneless/Bout_Bin
 17. 1d/SolverSetting/opCondition/dp/RPM
 18. 1d/SolverSetting/opCondition/dp/minlet
 #######################################################################################################################
"""

# ######################################################################################################################
# Parameter ranges:
# parameter_description_stageNum = [min_value, max_value]
# ######################################################################################################################
# Stage 1
stage1_impeller_blade_hubSect_te = [-60, 15]
stage1_impeller_blade_tipSect_te = [-60, 15]
stage1_impeller_in_inclination_angle = [-10, 10]
stage1_impeller_blade_number_of_main_blades = [5, 10]
stage1_impeller_blade_LE_LE_Clearance = [0.0001, 0.0004]
stage1_impeller_blade_TE_TE_Clearance = [0.0001, 0.0004]
stage1_impeller_in_Shroud_radius = [0.0181791, 0.0201791]
stage1_impeller_in_Hub_radius = [0.0035, 0.0055]
stage1_impeller_out_Outlet_radius = [0.0350, 0.04]
stage1_impeller_out_Outlet_width = [0.004, 0.005]

# Volute
stage1_volute_Throat_area = [0.0004, 0.0008]
stage1_volute_Exit_pipe_diameter = [0.02, 0.04]

# Diffuser stage 1
stage1_vaneless_RpinRin = [0.035944, 0.068609]
stage1_vaneless_BpinBin = [0.002, 0.0045532]
stage1_vaneless_RoutRin = [0.045, 0.085]
stage1_vaneless_BoutBin = [0.002, 0.004]

# Operating conditions
rotational_speed = [30000, 50000]
mass_flow_rate = [0.4, 1.6]

# Return channel

# Stage 2
'''
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

# Diffuser stage 2
Rpin_s2 = [0.035944, 0.068609]
Bpin_s2 = [0.002, 0.0045532]
outlet_avg_rad_s2 = [0.045, 0.085]
outlet_avg_width_s2 = [0.002, 0.004]
'''

# ######################################################################################################################
# Generate random numbers in appropriate ranges for each parameter
# ######################################################################################################################

# Generate N_samples for each parameter
N_SAMPLES = 100

# Prints samples to csv in chunks of 1000 rows
CHUNK_SIZE = 1000

# Parameter ranges in correct order, comma separated
param_ranges = np.array([stage1_impeller_blade_hubSect_te,  # 0
                         stage1_impeller_blade_tipSect_te,  # 1
                         stage1_impeller_in_inclination_angle,  # 2
                         stage1_impeller_blade_number_of_main_blades,  # 3
                         stage1_impeller_blade_LE_LE_Clearance,  # 4
                         stage1_impeller_blade_TE_TE_Clearance,  # 5
                         stage1_impeller_in_Shroud_radius,  # 6
                         stage1_impeller_in_Hub_radius,  # 7
                         stage1_impeller_out_Outlet_radius,  # 8
                         stage1_impeller_out_Outlet_width,  # 9

                         stage1_volute_Throat_area,  # 10
                         stage1_volute_Exit_pipe_diameter,  # 11

                         stage1_vaneless_RpinRin,  # 12
                         stage1_vaneless_BpinBin,  # 13
                         stage1_vaneless_RoutRin,  # 14
                         stage1_vaneless_BoutBin,  # 15

                         rotational_speed,  # 16
                         mass_flow_rate  # 17
                         ])

lower = param_ranges[:, 0]
widths = param_ranges[:, 1] - param_ranges[:, 0]
# Random samples in appropriate ranges
samples = (lower + widths * np.random.random(size=(N_SAMPLES, widths.shape[0]))).round(decimals=6)

# Make dataframe of random parameter samples
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!   USE THE CORRECT TURBOTIDES VARIABLE PATHS   !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
parameters = pd.DataFrame(samples, columns=[r'1d/1d_Machine/stage1/impeller/blade/hubSect/beta2b',
                                            r'1d/1d_Machine/stage1/impeller/blade/tipSect/beta2b',
                                            r'1d/1d_Machine/stage1/impeller/in/phi',
                                            r'1d/1d_Machine/stage1/impeller/blade/numMainBlades',
                                            r'1d/1d_Machine/stage1/impeller/blade/LE/clr',
                                            r'1d/1d_Machine/stage1/impeller/blade/TE/clr',
                                            r'1d/1d_Machine/stage1/impeller/in/Rs',
                                            r'1d/1d_Machine/stage1/impeller/in/Rh',
                                            r'1d/1d_Machine/stage1/impeller/out/Ra',
                                            r'1d/1d_Machine/stage1/impeller/out/b',
                                            r'1d/1d_Machine/stage1/volute/throatArea',
                                            r'1d/1d_Machine/stage1/volute/exitPipeDiameter',
                                            r'1d/1d_Machine/stage1/vaneless/Rpin_Rin',
                                            r'1d/1d_Machine/stage1/vaneless/Bpin_Bin',
                                            r'1d/1d_Machine/stage1/vaneless/Rout_Rin',
                                            r'1d/1d_Machine/stage1/vaneless/Bout_Bin',
                                            r'1d/SolverSetting/opCondition/dp/RPM',
                                            r'1d/SolverSetting/opCondition/dp/minlet'
                                            ])


# ######################################################################################################################
# Fix data: add rules as needed
# ######################################################################################################################

# Fix shroud s1 radius 0.0146791m above hub radius
parameters['1d/1d_Machine/stage1/impeller/in/Rs'] = parameters['1d/1d_Machine/stage1/impeller/in/Rh'] + 0.0146791

# Round number of s1 blades to closest int
parameters = parameters.round({'1d/1d_Machine/stage1/impeller/blade/numMainBlades': 0})

# Diffuser s1: Rpin<Rout; >Rin;
parameters['1d/1d_Machine/stage1/vaneless/Rpin_Rin'] = parameters['1d/1d_Machine/stage1/impeller/out/Ra'] \
                                            + np.random.random() \
                                            * (parameters['1d/1d_Machine/stage1/vaneless/Rout_Rin']
                                               - parameters['1d/1d_Machine/stage1/impeller/out/Ra'])

# Diffuser s1: Bpin>Bout; <Bin
parameters['1d/1d_Machine/stage1/vaneless/Bpin_Bin'] = parameters['1d/1d_Machine/stage1/impeller/out/b'] \
                                            + np.random.random() \
                                            * (parameters['1d/1d_Machine/stage1/vaneless/Bout_Bin']
                                               - parameters['1d/1d_Machine/stage1/impeller/out/b'])

# Diffuser s1: Ratio Rpin/Rin because Rpinch relies on the inlet Radius
parameters['1d/1d_Machine/stage1/vaneless/Rpin_Rin'] = parameters['1d/1d_Machine/stage1/vaneless/Rpin_Rin'] \
                                            / parameters['1d/1d_Machine/stage1/impeller/out/Ra']
parameters['1d/1d_Machine/stage1/vaneless/Bpin_Bin'] = parameters['1d/1d_Machine/stage1/vaneless/Bpin_Bin'] \
                                            / parameters['1d/1d_Machine/stage1/impeller/out/b']

# s1 Rout/Rin & Bout/Bin (Turbotides does not supply the Diffuser outlet
# Width and radius Inputs. This is a way to get around it)
parameters['1d/1d_Machine/stage1/vaneless/Rout_Rin'] = parameters['1d/1d_Machine/stage1/vaneless/Rout_Rin'] \
                                            / parameters['1d/1d_Machine/stage1/impeller/out/Ra']
parameters['1d/1d_Machine/stage1/vaneless/Bout_Bin'] = parameters['1d/1d_Machine/stage1/vaneless/Bout_Bin'] \
                                            / parameters['1d/1d_Machine/stage1/impeller/out/b']

# Round inlet blade angles to 2 decimal places
parameters = parameters.round({'1d/1d_Machine/stage1/impeller/blade/hubSect/beta2b': 2,
                               '1d/1d_Machine/stage1/impeller/blade/tipSect/beta2b': 2,
                               '1d/1d_Machine/stage1/impeller/in/phi': 2})


# ######################################################################################################################
# Check if parameters are feasible, delete bad rows, add rules as needed
# Rule 1: Rout/Rin > Rpin/Rin                       --> stage 1 diffuser
# Rule 2: Rpin/Rin >= 1                             --> stage 1 diffuser
# Rule 3: 1 > Bpin/Bin                              --> stage 1 diffuser
# Rule 4: Bpin/Bin >= Bout/Bin                      --> stage 1 diffuser
# ######################################################################################################################

parameters.drop(parameters[
                  (parameters['1d/1d_Machine/stage1/vaneless/Rout_Rin']         # R1,
                   > parameters['1d/1d_Machine/stage1/vaneless/Rpin_Rin'])
                & (parameters['1d/1d_Machine/stage1/vaneless/Rpin_Rin'] >= 1)   # R2,
                & (1 > parameters['1d/1d_Machine/stage1/vaneless/Bpin_Bin'])    # R3,
                & (parameters['1d/1d_Machine/stage1/vaneless/Bpin_Bin']
                   > parameters['1d/1d_Machine/stage1/vaneless/Bout_Bin'])      # R4,
                == False].index, inplace=True)


# ######################################################################################################################
# Add column that has Run1_para_study_od(1-1000)
# Add units
# ######################################################################################################################

parameters['Object Parameter Unit'] = (parameters.index % 1000) + 1    # add counter column (1 - 1000)
parameters['Object Parameter Unit'] = "Run1_para_study_od" + parameters['Object Parameter Unit'].apply(str)

# parameters.columns = pd.MultiIndex.from_tuples(list(zip(parameters.columns, units)))  # add units to columns


# ######################################################################################################################
# Create output directory if it doesn't already exist, print files in there
# Change index of dataframe to start at 1
# Pickle the dataframe, limited to 8gb, and save as csv, in batches of size CHUNK_SIZE
# ######################################################################################################################

if not os.path.exists('tt_input'):
    os.mkdir('tt_input')

exportData = parameters.to_dict('list')
with open('tt_input\\Batch_0.json', 'w') as f:
    json.dump(exportData, f, allow_nan=True)

parameters.index = parameters.index + 1     # change dataframe index to start at 1

for k, g in parameters.groupby(np.arange(len(parameters)) // CHUNK_SIZE):
    g.to_csv(f"tt_input\\Batch_{k}.csv", index=False)
    g.to_pickle(f"tt_input\\Batch_{k}.pkl")

# ######################################################################################################################
# Extras
# ######################################################################################################################

# unpickled_parameters = pd.read_pickle('random_parameters.pkl')  # unpickles the files to extract pandas dataframe

# t = time.time()
# elapsed = time.time() - t
# print(elapsed)
