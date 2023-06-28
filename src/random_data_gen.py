import numpy as np
import pandas as pd
import time

t = time.time()
# Order of input parameters
# ######################################## Add parameters as needed ####################################################
# 0. TE Blade Angle (Hub)…[deg(m)]                      (stage 1)
# 1. TE Blade Angle (Tip)…[deg(m)]                      (stage 1)
# 2. Inlet Inclination Angle…[deg(delta)]               (stage 1)
# 3. Number of Main Blades                              (stage 1)
# 4. LE Clearance                                       (stage 1)
# 5. TE Clearance                                       (stage 1)
# 6. Shroud Radius                                      (stage 1)
# 7. Inlet Hub Radius                                   (stage 1)
# 8. Impeller Outlet Radius                             (stage 1)
# 9. Impeller Outlet Width                              (stage 1)

# 10. Throat Area                                       (volute)
# 11. Exit pipe Diameter                                (volute)
# 12. Rpin -> Rpin/Rin
# 13. Bpin -> Bpin/Bin
# 14. Outlet Avg.Radius (diffuser) -> Rout/Rin          (diffuser)
# 15. Outlet Width (diffuser) -> Bout/Bin               (diffuser)
# 16. RPM                                               (operating parameter)
# 17. Mass Flow Rate… [kg/s]                            (operating parameter)

# 18. TE Blade Angle (Hub)…[deg(m)]                     (stage 2)
# 19. TE Blade Angle (Tip)…[deg(m)]                     (stage 2)
# 20. Inlet Inclination Angle…[deg(delta)]              (stage 2)
# 21. Number of Main Blades                             (stage 2)
# 22. LE Clearance                                      (stage 2)
# 23. TE Clearance                                      (stage 2)
# 24. Shroud Radius                                     (stage 2)
# 25. Inlet Hub Radius                                  (stage 2)
# 26. Impeller Outlet Radius                            (stage 2)
# 27. Impeller Outlet Width                             (stage 2)
# 28. Return channel stuff                              (return channel)
# ######################################################################################################################


# ######################################################################################################################
# Parameter ranges:
# parameter_description_stageNum = [min_value, max_value]
# ######################################################################################################################


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

# generate N_samples of parameter combos
N_samples = 200000

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

lower = param_ranges[:, 0]
widths = param_ranges[:, 1] - param_ranges[:, 0]
# random samples in appropriate ranges
samples = (lower + widths*np.random.random(size=(N_samples, widths.shape[0]))).round(decimals=6)

parameters = pd.DataFrame(samples, columns=['TE_blade_ang_hub_s1',
                                            'TE_blade_ang_tip_s1',
                                            'inlet_inclination_ang_s1',
                                            'number_of_main_blades_s1',
                                            'LE_Clearance_s1',
                                            'TE_Clearance_s1',
                                            'shroud_rad_s1',
                                            'inlet_hub_rad_s1',
                                            'impeller_outlet_rad_s1',
                                            'impeller_outlet_width_s1',
                                            'volute_throat_area',
                                            'volute_exit_pipe_dia',
                                            'Rpin_s1',
                                            'Bpin_s1',
                                            'outlet_avg_rad_s1',
                                            'outlet_avg_width_s1',
                                            'rpm',
                                            'mass_flow_rate',
                                            ])


# ######################################################################################################################
# fix data: add rules as needed
# ######################################################################################################################

# fix shroud s1 radius 0.0146791m above hub radius
parameters['shroud_rad_s1'] = parameters['inlet_hub_rad_s1'] + 0.0146791

# round number of s1 blades to closest int
parameters = parameters.round({'number_of_main_blades_s1': 0})

# Diffuser s1: Rpin<Rout; >Rin;
parameters['Rpin_s1'] = parameters['impeller_outlet_rad_s1'] +\
                        np.random.random() * (parameters['outlet_avg_rad_s1'] - parameters['impeller_outlet_rad_s1'] )

# Diffuser s1: Bpin>Bout; <Bin
# Check this ##### (b - a) or (a - b)
parameters['Bpin_s1'] = parameters['impeller_outlet_width_s1'] +\
                        np.random.random() * (parameters['outlet_avg_width_s1'] - parameters['impeller_outlet_width_s1'] )

# Diffuser s1: Ratio Rpin/Rin because Rpinch relies on the inlet Radius
parameters['Rpin_s1'] = parameters['Rpin_s1']/parameters['impeller_outlet_rad_s1']
parameters['Bpin_s1'] = parameters['Bpin_s1']/parameters['impeller_outlet_width_s1']

# s1 Rout/Rin & Bout/Bin (Turbotides does not supply the Diffuser outlet
# width and radius Inputs. This is a way to get around it)
parameters['outlet_avg_rad_s1'] = parameters['outlet_avg_rad_s1']/parameters['impeller_outlet_rad_s1']
parameters['outlet_avg_width_s1'] = parameters['outlet_avg_width_s1']/parameters['impeller_outlet_width_s1']

# round s1 TE inlet blade angles to 2 decimal places
parameters = parameters.round({'TE_blade_ang_hub_s1': 2, 'TE_blade_ang_tip_s1': 2, 'inlet_inclination_ang_s1': 2})

# ######################################################################################################################
# check if diffuser parameters are feasible, delete bad rows
# rule 1: Rout/Rin > Rpin/Rin >= 1 > Bpin/Bin >= Bout/Bin --> stage 1 diffuser
# rule 2: ...
# ######################################################################################################################

parameters.drop(parameters[(parameters['outlet_avg_rad_s1'] > parameters['Rpin_s1']) &
                (parameters['Rpin_s1'] >= 1) &
                (1 > parameters['Bpin_s1']) &
                (parameters['Bpin_s1'] > parameters['outlet_avg_width_s1']) == False].index, inplace=True)

# ######################################################################################################################
# add column that has Run1_para_study_od
# ######################################################################################################################

parameters['Object Parameter Unit'] = (parameters.index % 1000) + 1
parameters['Object Parameter Unit'] = 'Run1_para_study_od' + parameters['Object Parameter Unit'].apply(str)

# ######################################################################################################################
# pickle the dataframe, limited to 8gb
# ######################################################################################################################

parameters.to_pickle("random_parameters.pkl")
unpickled_df = pd.read_pickle("random_parameters.pkl")
print(parameters, parameters.shape)
print(unpickled_df, unpickled_df.shape)

# t = time.time()
elapsed = time.time() - t
print(elapsed)

