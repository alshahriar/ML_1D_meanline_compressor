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

"""Detail description of TT_post_process_s2.py

@Usage:
    Post-processing of the second stage
    Section 1 requires user inspection
        Check 1: Batch number
        Check 2: Append option or build new model
        Check 2: File format csv(ASCII) or pickle(binary)
        Check 3: Order of the TT output columns should be the following:
@Date: 
    June 30 2023

@Files
    Required:
        TT Input files
        TT Output files
    Optional:
        compressor.py
@Output
    Files:
        Batch data (both training and testing)
        Accumulated data (both training and testing)
"""
# %% Section 0: Loading modules and libraries
# Built-in/Generic Imports
import os
import shutil
import sys
from datetime import datetime
# Libs
import numpy as np
import pandas as pd
import re
# User-owned modules
from compressor import Compressor
from get_input_parameter_range import get_input_parameter_range
from clean_data_single_cond import clean_data_single_cond
# %% Section 1: Reading the data
# Reading the inputs to TT
# Make sure that your batch number is correct
batch_number = 100
# If you want to append the current batch data to the previous batches.
# This flag is usually True
append_flag = bool(True);

print("Batch number is", batch_number)

file_format = ".pkl"
data_in = pd.read_pickle(r"../tt_input_s2/Batch_"+str(batch_number)+file_format)
data_in = data_in.droplevel(1, axis=1)
data_in = data_in.set_index("Object Parameter Unit")
# Reding the outputs of TT
# right click on performance table - select export or save as
data_out_raw = pd.read_csv(r"../tt_output_s2/batch_"+str(batch_number)+".csv",index_col='Parameter').T

# Do not change these folder and file names
train_data_dir = r"../training_data_s2"
test_data_dir = r"../testing_data_s2"
train_data_fname = "training_parameters_s2"
test_data_fname = "testing_parameters_s2"

batch_list_fname = "batch_list_s2.txt"
# %%
# Renaming the column - one column has duplicate name.
# Check if the column numbers are matching with column titles.
# The values tells how the columns are ordered in TT
# TT does not properly write the column name in the prerformance table, so we
# have to rename them.

s1_Rpin_txt_in = r"stage1->vaneless1->Rpin"
s1_Bpin_txt_in = r"stage1->vaneless1->Bpin"
#s1_LE_Cl_txt_in = r"stage1->impeller->blade->LE->LE Clearance"
#s1_TE_Cl_txt_in = r"stage1->impeller->blade->TE->TE Clearance"

s2_Rpin_txt_in = r"stage2->vaneless1->Rpin"
s2_Bpin_txt_in = r"stage2->vaneless1->Bpin"
#s2_LE_Cl_txt_in = r"stage2->impeller->blade->LE->LE Clearance"
#s2_TE_Cl_txt_in = r"stage2->impeller->blade->TE->TE Clearance"


nCol = len(data_out_raw.loc["Object"])
for i in range(0,nCol):
    if(pd.isna(data_out_raw.loc["Object"][i])):
        data_out_raw.loc["Object"][i] = data_out_raw.columns[i]
    else:
        data_out_raw.loc["Object"][i] = data_out_raw.loc["Object"][i]+"_"+data_out_raw.columns[i]

list_of_final_columns = data_out_raw.loc["Object"].to_list()
list_of_final_columns = [lines.replace(' ','_') for lines in list_of_final_columns] # removing spaces
list_of_final_columns = [re.sub('[!,*/)@#%(&$?.^-]', '', lines) for lines in list_of_final_columns]
list_of_final_columns = [re.sub('>', '_', lines) for lines in list_of_final_columns]
data_out_raw.columns = list_of_final_columns

cnt = 0
for lines in list_of_final_columns:
    print("i_"+lines+" = "+str(cnt))
    cnt  = cnt + 1

i_Inlet_mass_flow = 0
i_Machine_power = 1
i_Machine_pressure_ratio_TS = 2
i_Isentropic_machine_efficiency_TS = 3
i_stage1_Inlet_mass_flow = 4
i_stage1_Pressure_ratio_P0exP0in = 5
i_stage1_Stage_power = 6
i_stage1_Efficiency_isentropic_TT = 7
i_stage1_inlet_in_Hub_radius = 8
i_stage1_inlet_in_Shroud_radius = 9
i_stage1_inlet_out_Hub_radius = 10
i_stage1_inlet_out_Shroud_radius = 11
i_stage1_impeller_RPM = 12
i_stage1_impeller_blade_Number_of_main_blades = 13
i_stage1_impeller_blade_hubSect_TE_blade_angle = 14
i_stage1_impeller_blade_tipSect_TE_blade_angle = 15
i_stage1_impeller_in_Inclination_angle = 16
i_stage1_impeller_out_Outlet_radius_avg = 17
i_stage1_impeller_out_Outlet_width = 18
i_stage1_vaneless1_out_Avg_radius = 19
i_stage1_vaneless1_out_Width = 20
i_stage1_return_channel_RC_crossover_out_Avg_radius = 21
i_stage1_return_channel_RC_crossover_out_Width = 22
i_stage1_return_channel_RC_deswirl_out_Avg_radius = 23
i_stage1_return_channel_RC_deswirl_out_Width = 24
i_stage1_return_channel_RC_deswirl_blade_Number_of_main_blades = 25
i_stage1_return_channel_RC_exitBend_out_Hub_radius = 26
i_stage1_return_channel_RC_exitBend_out_Shroud_radius = 27
i_stage2_Pressure_ratio_PexP0in = 28
i_stage2_Stage_power = 29
i_stage2_Efficiency_isentropic_TS = 30
i_stage2_inlet_Inlet_mass_flow_rate = 31
i_stage2_inlet_out_Hub_radius = 32
i_stage2_inlet_out_Shroud_radius = 33
i_stage2_impeller_RPM = 34
i_stage2_impeller_blade_Number_of_main_blades = 35
i_stage2_impeller_blade_hubSect_TE_blade_angle = 36
i_stage2_impeller_blade_tipSect_TE_blade_angle = 37
i_stage2_impeller_in_Inclination_angle = 38
i_stage2_impeller_out_Outlet_radius_avg = 39
i_stage2_impeller_out_Outlet_width = 40
i_stage2_vaneless1_out_Avg_radius = 41
i_stage2_vaneless1_out_Width = 42
i_stage2_volute_Throat_area = 43
i_stage2_volute_Exit_pipe_length = 44
i_stage2_volute_Exit_pipe_diameter = 45

# since the following parameters cannot be included in the performance table
#cnt+=1;
i_Rpin_s1 = cnt
cnt+=1;i_Bpin_s1 = cnt
#cnt+=1;i_LE_Clearance_s1 = cnt;
#cnt+=1;i_TE_Clearance_s1 = cnt;
# since the following parameters cannot be included in the performance table
cnt+=1;i_Rpin_s2 = cnt
cnt+=1;i_Bpin_s2 = cnt
#cnt+=1;i_LE_Clearance_s2 = cnt;
#cnt+=1;i_TE_Clearance_s2 = cnt;

# Choose the upper bound and lower bound to filter out the data
pressure_ratio_s1_min = 1.0;
pressure_ratio_s1_max = 4.0;

efficiency_s1_min = 0.5;
efficiency_s1_max = 0.95;

# Choose the upper bound and lower bound to filter out the data
pressure_ratio_s2_min = 1.0;
pressure_ratio_s2_max = 4.0;

efficiency_s2_min = 0.5;
efficiency_s2_max = 0.95;
# %% Section 2: Data formatting

# Removing unnecessary rows in the TT output file
data_out = data_out_raw.drop(["Object","Unit","Run1_dp"],axis = 0, errors='ignore')
data_out_cols = data_out.columns.tolist()

#
pressure_ratio_txt = data_out_cols[i_Machine_pressure_ratio_TS]
efficiency_txt = data_out_cols[i_Isentropic_machine_efficiency_TS]

pressure_ratio_s1_txt = data_out_cols[i_stage1_Pressure_ratio_P0exP0in]
efficiency_s1_txt = data_out_cols[i_stage1_Efficiency_isentropic_TT]
pressure_ratio_s2_txt = data_out_cols[i_stage2_Pressure_ratio_PexP0in]
efficiency_s2_txt = data_out_cols[i_stage2_Efficiency_isentropic_TS]

# for Rpin Bpin conversion
impl_radius_s1_txt = data_out_cols[i_stage1_impeller_out_Outlet_radius_avg]
impl_width_s1_txt = data_out_cols[i_stage1_impeller_out_Outlet_width]
diff_radius_s1_txt = data_out_cols[i_stage1_vaneless1_out_Avg_radius]
diff_width_s1_txt = data_out_cols[i_stage1_vaneless1_out_Width]

# for Rpin Bpin conversion
impl_radius_s2_txt = data_out_cols[i_stage2_impeller_out_Outlet_radius_avg]
impl_width_s2_txt = data_out_cols[i_stage2_impeller_out_Outlet_width]
diff_radius_s2_txt = data_out_cols[i_stage2_vaneless1_out_Avg_radius]
diff_width_s2_txt = data_out_cols[i_stage2_vaneless1_out_Width]
# %% Section 3: Data indexing
# TT does not return the results for the failed cases in the parametric study.
# It returns a column with the case number added to a string for successful runs.
# We compare the input case number and output case number to figure out which
# one fails.

temp_Rpin_s1 = [];
temp_Bpin_s1 = [];
temp_LE_Clearance_s1 = [];
temp_TE_Clearance_s1 = [];

temp_Rpin_s2 = [];
temp_Bpin_s2 = [];
temp_LE_Clearance_s2 = [];
temp_TE_Clearance_s2 = [];

nRows_out = len(data_out.index)
nRows_in = len(data_in.index)
for irow_out in range(0,nRows_out):
    row_out = data_out.index[irow_out]
    case_id_txt = (row_out[18:len(row_out)])
    case_id = int(case_id_txt)-1
    #s1
    temp_Rpin_s1.append(data_in[s1_Rpin_txt_in][case_id]*data_out[impl_radius_s1_txt][irow_out])
    temp_Bpin_s1.append(data_in[s1_Bpin_txt_in][case_id]*data_out[impl_width_s1_txt][irow_out])
    #temp_LE_Clearance_s1.append(data_in[s1_LE_Cl_txt_in][case_id])
    #temp_TE_Clearance_s1.append(data_in[s1_TE_Cl_txt_in][case_id])
    # s2
    temp_Rpin_s2.append(data_in[s2_Rpin_txt_in][case_id]*data_out[impl_radius_s2_txt][irow_out])
    temp_Bpin_s2.append(data_in[s2_Bpin_txt_in][case_id]*data_out[impl_width_s2_txt][irow_out])
    #temp_LE_Clearance_s2.append(data_in[s2_LE_Cl_txt_in][case_id])
    #temp_TE_Clearance_s2.append(data_in[s2_TE_Cl_txt_in][case_id])

data_out["Rpin_s1"] = temp_Rpin_s1
data_out["Bpin_s1"] = temp_Bpin_s1
#data_out["LE_Clearance_s1"] = temp_LE_Clearance_s1
#data_out["TE_Clearance_s1"] = temp_TE_Clearance_s1

data_out["Rpin_s2"] = temp_Rpin_s2
data_out["Bpin_s2"] = temp_Bpin_s2
#data_out["LE_Clearance_s2"] = temp_LE_Clearance_s2
#data_out["TE_Clearance_s2"] = temp_TE_Clearance_s2

data_out_cols = data_out.columns.tolist()
# %% Section 4: Data cleaning
# Filtering out the unrealistic data
# stage wise
data_out_filtered = data_out
data_out_filtered,data_dropped = clean_data_single_cond(data_out_filtered,pressure_ratio_s1_min,pressure_ratio_s1_max,pressure_ratio_s1_txt)
data_out_filtered,data_dropped = clean_data_single_cond(data_out_filtered,efficiency_s1_min,efficiency_s1_max,efficiency_s1_txt)
data_out_filtered,data_dropped = clean_data_single_cond(data_out_filtered,pressure_ratio_s2_min,pressure_ratio_s2_max,pressure_ratio_s2_txt)
data_out_filtered,data_dropped = clean_data_single_cond(data_out_filtered,efficiency_s2_min,efficiency_s2_max,efficiency_s2_txt)
# %% get input parameter ranges

param_ranges = get_input_parameter_range()

# %% Section 5: Separating the testing and training data

# Ordering the columns
input_overall = [i_Inlet_mass_flow]

input_s1_list = [i_stage1_Inlet_mass_flow,i_stage1_inlet_in_Hub_radius,i_stage1_inlet_in_Shroud_radius, \
    i_stage1_inlet_out_Hub_radius,i_stage1_inlet_out_Shroud_radius, \
    i_stage1_impeller_RPM,i_stage1_impeller_blade_Number_of_main_blades, \
    i_stage1_impeller_in_Inclination_angle, i_stage1_impeller_blade_hubSect_TE_blade_angle, i_stage1_impeller_blade_tipSect_TE_blade_angle, \
    #i_LE_Clearance_s1, i_TE_Clearance_s1,\
    i_Rpin_s1, i_Bpin_s1, i_stage1_impeller_out_Outlet_radius_avg, i_stage1_impeller_out_Outlet_width,\
    i_stage1_vaneless1_out_Avg_radius,i_stage1_vaneless1_out_Width]
    
input_RC_list = [i_stage1_return_channel_RC_crossover_out_Avg_radius, i_stage1_return_channel_RC_crossover_out_Width, \
    i_stage1_return_channel_RC_deswirl_out_Avg_radius, i_stage1_return_channel_RC_deswirl_out_Width, \
    i_stage1_return_channel_RC_deswirl_blade_Number_of_main_blades, \
    i_stage1_return_channel_RC_exitBend_out_Hub_radius, i_stage1_return_channel_RC_exitBend_out_Shroud_radius]
    
input_s2_list = [i_stage2_inlet_Inlet_mass_flow_rate, \
    i_stage2_inlet_out_Hub_radius,i_stage2_inlet_out_Shroud_radius,\
    i_stage2_impeller_RPM,i_stage2_impeller_blade_Number_of_main_blades, \
    i_stage2_impeller_in_Inclination_angle, i_stage2_impeller_blade_hubSect_TE_blade_angle, i_stage2_impeller_blade_tipSect_TE_blade_angle, \
    #i_LE_Clearance_s2, i_TE_Clearance_s2,\
    i_Rpin_s2, i_Bpin_s2, i_stage2_impeller_out_Outlet_radius_avg, i_stage2_impeller_out_Outlet_width, \
    i_stage2_vaneless1_out_Avg_radius,i_stage2_vaneless1_out_Width]
    
input_volute_list = [i_stage2_volute_Throat_area, i_stage2_volute_Exit_pipe_length, i_stage2_volute_Exit_pipe_diameter]

output_overall = [i_Machine_power,i_Machine_pressure_ratio_TS,i_Isentropic_machine_efficiency_TS]
output_s1_list = [i_stage1_Pressure_ratio_P0exP0in,i_stage1_Stage_power,i_stage1_Efficiency_isentropic_TT]
output_s2_list = [i_stage2_Pressure_ratio_PexP0in,i_stage2_Stage_power,i_stage2_Efficiency_isentropic_TS]

full_list_in_order = input_overall + input_s1_list + input_RC_list + input_s2_list + input_volute_list + \
                      output_overall + output_s1_list + output_s2_list

print("Input:", len(input_overall + input_s1_list + input_RC_list + input_s2_list + input_volute_list))
print("Output:", len(output_overall + output_s1_list + output_s2_list))
#if(len(full_list_in_order)!=cnt+1):
    #sys.exit("some of the column in the performance table not included in the final analysis")
    
data_ordered = data_out_filtered.iloc[:,full_list_in_order]

# Seperating 80 percent data for training
nRows = len(data_out_filtered)
nTrain = int(0.8*nRows)
data_train = data_ordered.iloc[:nTrain,:]    
data_test = data_ordered.iloc[nTrain+1:,:]

train_full_dir = os.path.join(train_data_dir, train_data_fname)
test_full_dir = os.path.join(test_data_dir, test_data_fname)

# %% saving the current batch data
data_train.to_csv(train_full_dir+"_filtered_"+str(batch_number)+".csv", index=False)
data_test.to_csv(test_full_dir+"_filtered_"+str(batch_number)+".csv", index=False)

now = datetime.now()
temp_str = now.strftime("%Y%m%d%H%M%S")

if(append_flag):
    if(os.path.isfile(train_full_dir+file_format)):
        data_train_saved = pd.read_pickle(train_full_dir+file_format)
        data_train = pd.concat([data_train_saved,data_train])
    if(os.path.isfile(test_full_dir+file_format)):
        data_test_saved = pd.read_pickle(test_full_dir+file_format)
        data_test = pd.concat([data_test_saved,data_test])
else:
    # making a backup file of the exisitng train data
    original = train_full_dir+file_format
    target = os.path.join(train_data_dir, "backup_s2_"+temp_str+"_"+train_data_fname+file_format)
    if(os.path.isfile(original)):
        shutil.copyfile(original, target)
        # making a backup file of existing test data
    original = test_full_dir+file_format
    target = os.path.join(test_data_dir, "backup_s2_"+temp_str+"_"+test_data_fname+file_format)
    if(os.path.isfile(original)):
        shutil.copyfile(original, target)

full_dir_batch_llist = os.path.join(train_data_dir, batch_list_fname)

# Check if this batch is already added
file1 = open(full_dir_batch_llist, 'r')
Lines = file1.readlines()
file1.close()
# Strips the newline character
for line in Lines:
    #print(line)
    if int(line)==batch_number:
        sys.exit("Batch "+str(batch_number)+" is already added")
    #print("Line{}: {}".format(count, line.strip()))

file1 = open(full_dir_batch_llist, 'a')
file1.write("\n"+str(batch_number))
file1.close()

# writing the file
data_train.to_pickle(train_full_dir+file_format)
data_test.to_pickle(test_full_dir+file_format)

# data_train.columns.to_series().to_csv("list_of_final_columns.csv")
list_of_final_columns = data_train.columns.to_list()
np.savetxt('list_of_final_columns_s2.txt', list_of_final_columns, delimiter="\n", fmt="%s")