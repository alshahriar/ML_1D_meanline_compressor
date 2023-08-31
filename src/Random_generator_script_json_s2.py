import numpy as np
import pandas as pd
import os
import time
import json
  
from param_bounds_s2 import get_param_bounds_np
from param_bounds_s2 import get_col_list
from param_bounds_s2 import get_col_list_json

batch_no = 116
batch_size = 1000;
param_ranges,units = get_param_bounds_np()
col_list = get_col_list()
col_list_json = get_col_list_json()
param_bounds_df = pd.DataFrame(param_ranges.T, columns=col_list).T
lower = param_ranges[:, 0]
widths = param_ranges[:, 1] - param_ranges[:, 0]
# Random samples in appropriate ranges
samples = (lower + widths * np.random.random(size=(batch_size, widths.shape[0]))).round(decimals=6)
parameters = pd.DataFrame(samples, columns=col_list)
parametersT = parameters.T
tag_txt = 'Object Parameter Unit'

if(0):
    s1_n_blade_txt = '1d/1d_Machine/stage1/impeller/blade/numMainBlade'
    s2_n_blade_txt = '1d/1d_Machine/stage2/impeller/blade/numMainBlade'
    rc_n_blade_txt = '1d/1d_Machine/stage1/return_channel/deswirl/blade/numMainBlade'
    
    s1_hub_angle_txt = '1d/1d_Machine/stage1/impeller/blade/hubSect/beta2b'
    s1_tip_angle_txt = '1d/1d_Machine/stage1/impeller/blade/tipSect/beta2b'
    s1_inc_angle_txt = '1d/1d_Machine/stage1/impeller/in/phi'
    
    s2_hub_angle_txt = '1d/1d_Machine/stage2/impeller/blade/hubSect/beta2b'
    s2_tip_angle_txt = '1d/1d_Machine/stage2/impeller/blade/tipSect/beta2b'
    s2_inc_angle_txt = '1d/1d_Machine/stage2/impeller/in/phi'
    
else:
    
    #for i in range(len(col_list)):
        #print(i," ",col_list[i])
    s1_n_blade_txt = col_list[2]
    s2_n_blade_txt = col_list[23]
    rc_n_blade_txt = col_list[18]
    
    s1_hub_angle_txt = col_list[3]
    s1_tip_angle_txt = col_list[4]
    s1_inc_angle_txt = col_list[5]
    
    s2_hub_angle_txt = col_list[24]
    s2_tip_angle_txt = col_list[25]
    s2_inc_angle_txt = col_list[26]
    
    s1_hub_R_txt = col_list[0]
    s1_shroud_R_txt = col_list[1]
    s2_hub_R_txt = col_list[21]
    s2_shroud_R_txt = col_list[22]

    s1_Rpin_txt = col_list[10]
    s1_Bpin_txt = col_list[11]
    s2_Rpin_txt = col_list[31]
    s2_Bpin_txt = col_list[32]

    s1_diff_radius_txt = col_list[12]
    s1_diff_width_txt = col_list[13]
    s2_diff_radius_txt = col_list[33]
    s2_diff_width_txt = col_list[34]

    s1_imp_out_radius_txt = col_list[8]
    s1_imp_out_width_txt = col_list[9]
    s2_imp_out_radius_txt = col_list[29]
    s2_imp_out_width_txt = col_list[30]

# %%
parameters = parameters.round({s1_n_blade_txt: 0})
parameters = parameters.round({rc_n_blade_txt: 0})
parameters = parameters.round({s2_n_blade_txt: 0})
parameters = parameters.round({"Rotational speed": 2})

# Round inlet blade angles to 2 decimal places
parameters = parameters.round({s1_hub_angle_txt: 2,
                               s1_tip_angle_txt: 2,
                               s1_inc_angle_txt: 2})

# Round inlet blade angles to 2 decimal places
parameters = parameters.round({s2_hub_angle_txt: 2,
                               s2_tip_angle_txt: 2,
                               s2_inc_angle_txt: 2})

parameters[s1_shroud_R_txt] = parameters[s1_hub_R_txt] + 0.0146791
parameters[s2_shroud_R_txt] = parameters[s2_hub_R_txt] + 0.0146791

parameters[s1_Rpin_txt] = parameters[s1_imp_out_radius_txt] \
                                            + np.random.random() \
                                            * (parameters[s1_diff_radius_txt]
                                               - parameters[s1_imp_out_radius_txt])
parameters[s1_Bpin_txt] = parameters[s1_imp_out_width_txt] \
                                            + np.random.random() \
                                            * (parameters[s1_diff_width_txt]
                                               - parameters[s1_imp_out_width_txt])

parameters[s2_Rpin_txt] = parameters[s2_imp_out_radius_txt] \
                                            + np.random.random() \
                                            * (parameters[s2_diff_radius_txt]
                                               - parameters[s2_imp_out_radius_txt])
# Diffuser s2: Bpin>Bout; <Bin
parameters[s2_Bpin_txt] = parameters[s2_imp_out_width_txt] \
                                            + np.random.random() \
                                            * (parameters[s2_diff_width_txt]
                                               - parameters[s2_imp_out_width_txt])

parameters[tag_txt] = (parameters.index) + 1    # add counter column (1 - 1000)
parameters[tag_txt] = "Run1_para_study_od" + parameters[tag_txt].apply(str)
# %%
if(1):
    # fixing TurboTides issue - converting in radians
    col_2_fix = s1_hub_angle_txt
    parameters[col_2_fix] = parameters[col_2_fix]/57.29569653074808
    col_2_fix = s1_tip_angle_txt
    parameters[col_2_fix] = parameters[col_2_fix]/57.29569653074808
    col_2_fix = s1_inc_angle_txt
    parameters[col_2_fix] = parameters[col_2_fix]/57.29569653074808
    
    col_2_fix = s2_hub_angle_txt
    parameters[col_2_fix] = parameters[col_2_fix]/57.29569653074808
    col_2_fix = s2_tip_angle_txt
    parameters[col_2_fix] = parameters[col_2_fix]/57.29569653074808
    col_2_fix = s2_inc_angle_txt
    parameters[col_2_fix] = parameters[col_2_fix]/57.29569653074808
# %%
parameters.to_pickle("../tt_input_s2/Batch_"+str(batch_no)+".pkl")

parameters_to_write = parameters.drop(tag_txt,axis = 1)
parameters_to_write.to_csv("../tt_input_s2/Batch_"+str(batch_no)+".csv", index=False)
# %% For continuous run
col_list_json.append(tag_txt)
exportData = parameters.to_dict('list')
with open('..\\tt_input_s2\\Batch_'+str(batch_no)+'.json', 'w') as f:
    json.dump(exportData, f, allow_nan=True)