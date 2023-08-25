import numpy as np
import pandas as pd
import os
import time

# Order of input parameters, use proper TurboTides variables paths!
# Keep this order consistent through param_ranges, parameters, and units
# ######################################## Add parameters as needed ####################################################
# 1.  stage1->impeller->blade->hubSect
# 2.  stage1->impeller->blade->tipSect
# 3.  stage1->impeller->in->Inclination angle
# 4.  stage1->impeller->blade->Number of main blades
# 5.  stage1->impeller->blade->LE->LE Clearance
# 6.  stage1->impeller->blade->TE->TE Clearance
# 7.  stage1->impeller->in->Shroud radius
# 8.  stage1->impeller->in->Hub radius
# 9.  stage1->impeller->out->Outlet radius (avg)
# 10. stage1->impeller->out->Outlet width
# 11. stage1->volute->Throat area
# 12. stage1->volute->Exit pipe diameter
# 13. stage1->vaneless1->Rpin/Rin
# 14. stage1->vaneless1->Bpin/Bin
# 15. stage1->vaneless1->Rout/Rin
# 16. stage1->vaneless1->Bout/Bin
# 17. Rotational speed
# 18. Mass flow rate
# ######################################################################################################################


# ######################################################################################################################
# Parameter ranges:
# parameter_description_stageNum = [min_value, max_value]
# ######################################################################################################################
# Stage 1
stage1_impeller_blade_hubSect = [-60, 15]
stage1_impeller_blade_tipSect = [-60, 15]
stage1_impeller_in_Inclination_angle = [-10, 10]
stage1_impeller_blade_Number_of_main_blades = [5, 10]
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
stage1_vaneless1_RpinRin = [0.035944, 0.068609]
stage1_vaneless1_BpinBin = [0.002, 0.0045532]
stage1_vaneless1_RoutRin = [0.045, 0.085]
stage1_vaneless1_BoutBin = [0.002, 0.004]

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
N_SAMPLES = 1000

# Prints samples to csv in chunks of 1000 rows
CHUNK_SIZE = 1000
batch_number = 180;

print("batch number: ",batch_number)

# Parameter ranges in correct order, comma separated
param_ranges = np.array([stage1_impeller_blade_hubSect,                # 0
                         stage1_impeller_blade_tipSect,                # 1
                         stage1_impeller_in_Inclination_angle,         # 2
                         stage1_impeller_blade_Number_of_main_blades,  # 3
                         stage1_impeller_blade_LE_LE_Clearance,        # 4
                         stage1_impeller_blade_TE_TE_Clearance,        # 5
                         stage1_impeller_in_Shroud_radius,             # 6
                         stage1_impeller_in_Hub_radius,                # 7
                         stage1_impeller_out_Outlet_radius,            # 8
                         stage1_impeller_out_Outlet_width,             # 9

                         stage1_volute_Throat_area,                    # 10
                         stage1_volute_Exit_pipe_diameter,             # 11

                         stage1_vaneless1_RpinRin,                     # 12
                         stage1_vaneless1_BpinBin,                     # 13
                         stage1_vaneless1_RoutRin,                     # 14
                         stage1_vaneless1_BoutBin,                     # 15

                         rotational_speed,                             # 16
                         mass_flow_rate                                # 17
                         ])

lower = param_ranges[:, 0]
widths = param_ranges[:, 1] - param_ranges[:, 0]
# Random samples in appropriate ranges
samples = (lower + widths * np.random.random(size=(N_SAMPLES, widths.shape[0]))).round(decimals=6)

# Make dataframe of random parameter samples
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!   USE THE CORRECT TURBOTIDES VARIABLE PATHS   !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
parameters = pd.DataFrame(samples, columns=['stage1->impeller->blade->hubSect->TE blade Angle',
                                            'stage1->impeller->blade->tipSect->TE blade Angle',
                                            'stage1->impeller->in->Inclination angle',
                                            'stage1->impeller->blade->Number of main blades',
                                            'stage1->impeller->blade->LE->LE Clearance',
                                            'stage1->impeller->blade->TE->TE Clearance',
                                            'stage1->impeller->in->Shroud radius',
                                            'stage1->impeller->in->Hub radius',
                                            'stage1->impeller->out->Outlet radius (avg)',
                                            'stage1->impeller->out->Outlet width',
                                            'stage1->volute->Throat area',
                                            'stage1->volute->Exit pipe diameter',
                                            'stage1->vaneless1->Rpin/Rin',
                                            'stage1->vaneless1->Bpin/Bin',
                                            'stage1->vaneless1->Rout/Rin',
                                            'stage1->vaneless1->Bout/Bin',
                                            'Rotational speed',
                                            'Mass flow rate'
                                            ])

# Set units of each parameter to be added to the table later
units = ['deg(m)',
         'deg(m)',
         'deg(delta)',
         '',
         'm',
         'm',
         'm',
         'm',
         'm',
         'm',
         'm^2',
         'm',
         '',
         '',
         '',
         '',
         'RPM',
         'kg/s',
         # leave Object Parameter Unit at the end
         '']


# ######################################################################################################################
# Fix data: add rules as needed
# ######################################################################################################################

# Fix shroud s1 radius 0.0146791m above hub radius
parameters['stage1->impeller->in->Shroud radius'] = parameters['stage1->impeller->in->Hub radius'] + 0.0146791

# Round number of s1 blades to closest int
parameters = parameters.round({'stage1->impeller->blade->Number of main blades': 0})

# Diffuser s1: Rpin<Rout; >Rin;
parameters['stage1->vaneless1->Rpin/Rin'] = parameters['stage1->impeller->out->Outlet radius (avg)'] \
                                            + np.random.random() \
                                            * (parameters['stage1->vaneless1->Rout/Rin']
                                               - parameters['stage1->impeller->out->Outlet radius (avg)'])

# Diffuser s1: Bpin>Bout; <Bin
parameters['stage1->vaneless1->Bpin/Bin'] = parameters['stage1->impeller->out->Outlet width'] \
                                            + np.random.random() \
                                            * (parameters['stage1->vaneless1->Bout/Bin']
                                               - parameters['stage1->impeller->out->Outlet width'])

# Diffuser s1: Ratio Rpin/Rin because Rpinch relies on the inlet Radius
parameters['stage1->vaneless1->Rpin/Rin'] = parameters['stage1->vaneless1->Rpin/Rin'] \
                                            / parameters['stage1->impeller->out->Outlet radius (avg)']
parameters['stage1->vaneless1->Bpin/Bin'] = parameters['stage1->vaneless1->Bpin/Bin'] \
                                            / parameters['stage1->impeller->out->Outlet width']

# s1 Rout/Rin & Bout/Bin (Turbotides does not supply the Diffuser outlet
# Width and radius Inputs. This is a way to get around it)
parameters['stage1->vaneless1->Rout/Rin'] = parameters['stage1->vaneless1->Rout/Rin'] \
                                            / parameters['stage1->impeller->out->Outlet radius (avg)']
parameters['stage1->vaneless1->Bout/Bin'] = parameters['stage1->vaneless1->Bout/Bin'] \
                                            / parameters['stage1->impeller->out->Outlet width']

# Round inlet blade angles to 2 decimal places
parameters = parameters.round({'stage1->impeller->blade->hubSect': 2,
                               'stage1->impeller->blade->tipSect': 2,
                               'stage1->impeller->in->Inclination angle': 2})


# ######################################################################################################################
# Check if parameters are feasible, delete bad rows, add rules as needed
# Rule 1: Rout/Rin > Rpin/Rin                       --> stage 1 diffuser
# Rule 2: Rpin/Rin >= 1                             --> stage 1 diffuser
# Rule 3: 1 > Bpin/Bin                              --> stage 1 diffuser
# Rule 4: Bpin/Bin >= Bout/Bin                      --> stage 1 diffuser
# ######################################################################################################################

parameters.drop(parameters[
                  (parameters['stage1->vaneless1->Rout/Rin'] > parameters['stage1->vaneless1->Rpin/Rin'])      # R1,
                & (parameters['stage1->vaneless1->Rpin/Rin'] >= 1)                                             # R2,
                & (1 > parameters['stage1->vaneless1->Bpin/Bin'])                                              # R3,
                & (parameters['stage1->vaneless1->Bpin/Bin'] > parameters['stage1->vaneless1->Bout/Bin'])      # R4,
                == False].index, inplace=True)


# ######################################################################################################################
# Add column that has Run1_para_study_od(1-1000)
# Add units
# ######################################################################################################################

parameters['Object Parameter Unit'] = (parameters.index % 1000) + 1    # add counter column (1 - 1000)
parameters['Object Parameter Unit'] = 'Run1_para_study_od' + parameters['Object Parameter Unit'].apply(str)

parameters.columns = pd.MultiIndex.from_tuples(list(zip(parameters.columns, units)))  # add units to columns


# ######################################################################################################################
# Create output directory if it doesn't already exist, print files in there
# Change index of dataframe to start at 1
# Pickle the dataframe, limited to 8gb, and save as csv, in batches of size CHUNK_SIZE
# ######################################################################################################################

# if not os.path.exists("Sample_Batches"):
#     os.mkdir("Sample_Batches")
# os.chdir("Sample_Batches")                  # change cwd

parameters.index = parameters.index + 1     # change dataframe index to start at 1

# for k, g in parameters.groupby(np.arange(len(parameters)) // CHUNK_SIZE):
#     g.to_csv(f'Batch_{k+1}.csv', index=False)
#     g.to_pickle(f'Batch_{k+1}.pkl')

file_name = str('batch_%d.csv' % batch_number)
dir_name = r"../tt_input/"
full_name = os.path.join(dir_name, file_name)

if(os.path.isfile(full_name)):
    import sys
    sys.exit("Error: change the batch number. File exists: %s" % full_name)
else:
    parameters.to_csv(full_name, index=False)


file_name = str('batch_%d.pkl' % batch_number)
dir_name = r"../tt_input/"
full_name = os.path.join(dir_name, file_name)
parameters.to_pickle(full_name)


# ######################################################################################################################
# Extras
# ######################################################################################################################

# unpickled_parameters = pd.read_pickle("random_parameters.pkl")  # unpickles the files to extract pandas dataframe

# t = time.time()
# elapsed = time.time() - t
# print(elapsed)