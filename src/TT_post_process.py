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
    Execute after obtaining the results from TurboTides
    Section 1 requires user inspection
        Check 1: Batch number
        Check 2: Append option or build new model
        Check 2: File format csv(ASCII) or pickle(binary)
        Check 3: Order of the TT output columns should be the following:
            ['Machine pressure ratio (T-S)',
             'Machine power',
             'Isentropic machine efficiency (T-S)',
             'Inlet mass flow',
             'Inlet mass flow corrected',
             'RPM',
             'Number of main blades',
             'TE Blade Angle Hub',
             'TE Blade Angle Tip',
             'Inclination angle',
             'Hub radius',
             'Shroud radius',
             'Impeller radius',
             'Impeller width',
             'Diffuser radius',
             'Diffuser width',
             'Throat area',
             'Exit pipe diameter',]        
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
from datetime import datetime
# Libs
import numpy as np
import pandas as pd
# User-owned modules
from compressor import Compressor
from get_input_parameter_range import get_input_parameter_range
# %% Section 1: Reading the data
# Reading the inputs to TT
# Make sure that your batch number is correct
batch_number = 0
# If you want to append the current batch data to the previous batches.
# This flag is usually True
append_flag = bool(False);

# Select the method type to handle file
# 0: for csv file
# 1: pickle file: faster and saves memory
read_method = 1;

print("Batch number is", batch_number)

if read_method==0:
    # inside TT, right click on the paramatric study table, select export as csv
    file_format = ".csv"
    data_in = pd.read_csv(r"../tt_input/batch_"+str(batch_number)+file_format).T
    read_method = 0
    # make sure the following variables and their values align with data_in
    i_TE_blade_ang_hub_in = 0;
    i_TE_blade_ang_tip_in = 1;
    i_LE_Clearance_in = 4;
    i_TE_Clearance_in = 5;
    i_Rpin_by_Rin_in = 12;
    i_Bpin_by_Rin_in = 13;
else:
    # if created using py script
    file_format = ".pkl"
    data_in = pd.read_pickle(r"../tt_input/Batch_"+str(batch_number)+file_format)
    read_method = 1

# Reding the outputs of TT
# right click on performance table - select export or save as
data_out_raw = pd.read_csv(r"../tt_output/batch_"+str(batch_number)+".csv",index_col='Parameter').T

# Renaming the column - one column has duplicate name.
# Check if the column numbers are matching with column titles.
# The values tells how the columns are ordered in TT
# TT does not properly write the column name in the prerformance table, so we
# have to rename them.
i_pressure_ratio_out = 0
i_machine_power = 1
i_efficiency_out = 2
i_mass_flow_rate_out = 3
i_RPM_out = 5
i_n_main_blades_out = 6
i_TE_blade_ang_hub_out = 7
i_TE_blade_ang_tip_out = 8
i_inclination_angle_out = 9
i_hub_radius_out = 10
i_shroud_radius_out = 11
i_impl_radius_out = 12
i_impl_width_out = 13
i_diff_radius_out = 14
i_diff_width_out = 15
i_throat_area_out = 16
i_pipe_diameter_out = 17
i_Rpin_out = 18
i_Bpin_out = 19
i_LE_Clearance_out = 20;
i_TE_Clearance_out = 21;

# Choose the upper bound and lower bound to filter out the data
pressure_ratio_min = 1.0;
pressure_ratio_max = 5.0;

efficiency_min = 0.2;
efficiency_max = 1.0;
# %% Section 2: Data formatting

# Removing unnecessary rows in the TT output file
data_out = data_out_raw.drop(["Object","Unit","Run1_dp"],axis = 0, errors='ignore')

data_out_cols = data_out.columns.tolist()
TE_blade_ang_hub_txt ='TE Blade Angle Hub'
TE_blade_ang_tip_txt ='TE Blade Angle Tip'
if data_out_cols[i_TE_blade_ang_hub_out] == 'TE blade angle': data_out_cols[i_TE_blade_ang_hub_out] = TE_blade_ang_hub_txt
if data_out_cols[i_TE_blade_ang_tip_out] == 'TE blade angle': data_out_cols[i_TE_blade_ang_tip_out] = TE_blade_ang_tip_txt
if data_out_cols[i_impl_radius_out] == 'Outlet radius (avg)': data_out_cols[i_impl_radius_out] = "Impeller radius"
if data_out_cols[i_impl_width_out] == 'Outlet width': data_out_cols[i_impl_width_out] = "Impeller width"
if data_out_cols[i_diff_radius_out] == 'Avg. radius': data_out_cols[i_diff_radius_out] = "Diffuser radius"
if data_out_cols[i_diff_width_out] == 'Width': data_out_cols[i_diff_width_out] = "Diffuser width"

data_out.columns=data_out_cols
data_out_cols = data_out.columns.tolist()
pressure_ratio_txt = data_out_cols[i_pressure_ratio_out]
efficiency_txt = data_out_cols[i_efficiency_out]
impl_radius_txt = data_out_cols[i_impl_radius_out]
impl_width_txt = data_out_cols[i_impl_width_out]
diff_radius_txt = data_out_cols[i_diff_radius_out]
diff_width_txt = data_out_cols[i_diff_width_out]

# %% Section 3: Data indexing
# TT does not return the results for the failed cases in the parametric study.
# It returns a column with the case number added to a string for successful runs.
# We compare the input case number and output case number to figure out which
# one fails.

temp_Rpin = [];
temp_Bpin = [];
temp_LE_Clearance = [];
temp_TE_Clearance = [];
temp_TE_blade_ang_hub = [];
temp_TE_blade_ang_tip = [];

data_in.columns = data_in.columns.droplevel(1)

nRows_out = len(data_out.index)
nRows_in = len(data_in.index)
for irow_out in range(0,nRows_out):
    row_out = data_out.index[irow_out]
    case_number_txt = (row_out[18:len(row_out)])
    case_number = int(case_number_txt)
    if read_method == 0:
        temp_Rpin.append(data_in.loc[case_number_txt].at[i_Rpin_by_Rin_in]*data_out[impl_radius_txt][irow_out])
        temp_Bpin.append(data_in.loc[case_number_txt].at[i_Bpin_by_Rin_in]*data_out[impl_width_txt][irow_out])
        temp_LE_Clearance.append(data_in.loc[case_number_txt].at[i_LE_Clearance_in])
        temp_TE_Clearance.append(data_in.loc[case_number_txt].at[i_TE_Clearance_in])
        temp_TE_blade_ang_hub.append(data_in.loc[case_number_txt].at[i_TE_blade_ang_hub_in])
        temp_TE_blade_ang_tip.append(data_in.loc[case_number_txt].at[i_TE_blade_ang_tip_in])
    else:
        temp_Rpin.append(data_in.loc[case_number].at[r"stage1->vaneless1->Rpin/Rin"]*data_out[impl_radius_txt][irow_out])
        temp_Bpin.append(data_in.loc[case_number].at[r"stage1->vaneless1->Bpin/Bin"]*data_out[impl_width_txt][irow_out])
        temp_LE_Clearance.append(data_in.loc[case_number].at[r"stage1->impeller->blade->LE->LE Clearance"])
        temp_TE_Clearance.append(data_in.loc[case_number].at[r"stage1->impeller->blade->TE->TE Clearance"])
        temp_TE_blade_ang_hub.append(data_in.loc[case_number].at[r"stage1->impeller->blade->hubSect"])
        temp_TE_blade_ang_tip.append(data_in.loc[case_number].at[r"stage1->impeller->blade->tipSect"])
        
data_out["Rpin"] = temp_Rpin
data_out["Bpin"] = temp_Bpin
data_out["LE Clearance"] = temp_LE_Clearance
data_out["TE Clearance"] = temp_TE_Clearance
data_out[TE_blade_ang_hub_txt] = temp_TE_blade_ang_hub
data_out[TE_blade_ang_tip_txt] = temp_TE_blade_ang_tip
data_out_cols = data_out.columns.tolist()
# %% Section 4: Data cleaning

# Filtering out the unrealistic data

nRows_out = len(data_out.index)

cases_to_remove_eta_high = [];
cases_to_remove_eta_low = [];
cases_to_remove_pr_high = [];
cases_to_remove_pr_low = [];
flag_rm_eta = np.zeros(((nRows_out,1)),dtype = bool);
flag_rm_pr = np.zeros(((nRows_out,1)),dtype = bool);

for irow_out in range(0,nRows_out):
    eta = data_out[efficiency_txt][irow_out]
    flag = bool(False)
    if eta>efficiency_max:
        cases_to_remove_eta_high.append(data_out.index[irow_out])
        flag_rm_eta[irow_out] = bool(True)
    if eta<efficiency_min:
        cases_to_remove_eta_low.append(data_out.index[irow_out])
        flag_rm_eta[irow_out] = bool(True)    

for irow_out in range(0,nRows_out):
    pr = data_out[pressure_ratio_txt][irow_out]
    if pr>pressure_ratio_max:
        cases_to_remove_pr_high.append(data_out.index[irow_out])
        flag_rm_pr[irow_out] = bool(True)
    if pr<pressure_ratio_min:
        cases_to_remove_pr_low.append(data_out.index[irow_out])
        flag_rm_pr[irow_out] = bool(True)
        
cases_to_remove_temp = cases_to_remove_eta_high + cases_to_remove_eta_low + \
    cases_to_remove_pr_high + cases_to_remove_pr_low
cases_to_remove = [*set(cases_to_remove_temp)]

flag_rm = np.logical_or(flag_rm_eta,flag_rm_pr)
data_out["Bad Samples"] = flag_rm
data_out_filtered = data_out[data_out["Bad Samples"] == False]
data_out_bad_samples = data_out[data_out["Bad Samples"] == True]


# %% get input parameter ranges
param_ranges = get_input_parameter_range()


# %% Section 5: Separating the testing and training data

# Ordering the columns exactly the same way Krista did
data_ordered = data_out_filtered.iloc[:, \
                                     [i_mass_flow_rate_out,i_RPM_out,i_n_main_blades_out, \
                                      i_TE_blade_ang_hub_out, i_TE_blade_ang_tip_out, \
                                      i_LE_Clearance_out,i_TE_Clearance_out, \
                                      i_inclination_angle_out, i_hub_radius_out, i_shroud_radius_out, \
                                      i_impl_radius_out, i_impl_width_out, i_Rpin_out, i_Bpin_out, \
                                      i_diff_radius_out, i_diff_width_out, \
                                      i_throat_area_out, i_pipe_diameter_out, \
                                      i_pressure_ratio_out,i_machine_power,i_efficiency_out]]

    
# Seperating 80 percent data for training
nRows = len(data_out_filtered)
nTrain = int(0.8*nRows)
data_train = data_ordered.iloc[:nTrain,:]    
data_test = data_ordered.iloc[nTrain+1:,:]

train_data_dir = r"../training_data"
test_data_dir = r"../testing_data"    
train_data_fname = "train_parameters"
test_data_fname = "test_parameters"
train_full_dir = os.path.join(train_data_dir, train_data_fname)
test_full_dir = os.path.join(test_data_dir, test_data_fname)

# saving the current batch data
data_train.to_csv(train_full_dir+"_filtered_"+str(batch_number)+".csv", index=False)
data_test.to_csv(test_full_dir+"_filtered_"+str(batch_number)+".csv", index=False)

now = datetime.now()
temp_str = now.strftime("%Y%m%d%H%M%S")

if(append_flag):
    if(os.path.isfile(train_full_dir+file_format)):
        if read_method==0: data_train_saved = pd.read_csv(train_full_dir+file_format)
        else: data_train_saved = pd.read_pickle(train_full_dir+file_format)
        data_train = pd.concat([data_train_saved,data_train])
    if(os.path.isfile(test_full_dir+file_format)):
        if read_method==0: data_test_saved = pd.read_csv(test_full_dir+file_format)
        else: data_test_saved = pd.read_pickle(test_full_dir+file_format)
        data_test = pd.concat([data_test_saved,data_test])
else:
    # making a backup file of the exisitng train data
    original = train_full_dir+file_format
    target = os.path.join(train_data_dir, "backup_"+temp_str+"_"+train_data_fname+file_format)
    if(os.path.isfile(original)):
        shutil.copyfile(original, target)
        # making a backup file of existing test data
    original = test_full_dir+file_format
    target = os.path.join(test_data_dir, "backup_"+temp_str+"_"+test_data_fname+file_format)
    if(os.path.isfile(original)):
        shutil.copyfile(original, target)

# Writing them in appropriate format
if read_method==0:
    # writing the file
    data_train.to_csv(train_full_dir+file_format, index=False)
    data_test.to_csv(test_full_dir+file_format, index=False)
else:
    # writing the file
    data_train.to_pickle(train_full_dir+file_format)
    data_test.to_pickle(test_full_dir+file_format)

# %% Section 6: Optional - Trying something with Compressor class
# Not necessary - optional
cpr = Compressor() 
if 0:
    # TT does not return the results for the failed cases in the parametric study
    cpr.rpm = data_out["RPM"].values
    cpr.mass_flow_rate = data_out["Inlet mass flow"].values
    cpr.isetropic_efficiency = data_out["Isentropic machine efficiency (T-S)"].values
    cpr.pressure_ratio = data_out["Machine pressure ratio (T-S)"].values
    cpr.power = data_out["Machine power"].values

    cpr.s1.inlet_inclination_ang = data_out["Inclination angle"].values
    cpr.s1.number_of_main_blades = data_out["Number of main blades"].values
    cpr.s1.shroud_rad = data_out["Shroud radius"].values
    cpr.s1.inlet_hub_rad = data_out["Hub radius"].values
    cpr.s1.impeller_outlet_rad = data_out[impl_radius_txt].values
    cpr.s1.impeller_outlet_width = data_out[impl_width_txt].values

    cpr.volute_throat_area = data_out["Throat area"].values
    cpr.volute_exit_pipe_dia = data_out["Exit pipe diameter"].values

    nRows_out = len(data_out.index)
    nRows_in = len(data_in.index)
    for irow_out in range(0,nRows_out):
        row_out = data_out.index[irow_out]
        case_number_txt = (row_out[18:len(row_out)])
        if read_method == 0:
            cpr.s1.Rpin.append(data_in.loc[case_number_txt].at[i_Rpin_by_Rin_in]*cpr.s1.impeller_outlet_rad[irow_out])
            cpr.s1.Bpin.append(data_in.loc[case_number_txt].at[i_Bpin_by_Rin_in]*cpr.s1.impeller_outlet_width[irow_out])
            cpr.s1.LE_Clearance.append(data_in.loc[case_number_txt].at[i_LE_Clearance_in])
            cpr.s1.TE_Clearance.append(data_in.loc[case_number_txt].at[i_TE_Clearance_in])
            cpr.s1.TE_blade_ang_hub.append(data_in.loc[case_number_txt].at[i_TE_blade_ang_hub_in])
            cpr.s1.TE_blade_ang_tip.append(data_in.loc[case_number_txt].at[i_TE_blade_ang_tip_in])
        else:
            case_number = int(case_number_txt)
            cpr.s1.Rpin.append(data_in.loc[case_number].at["Rpin_s1"]*cpr.s1.impeller_outlet_rad[irow_out])
            cpr.s1.Bpin.append(data_in.loc[case_number].at["Bpin_s1"]*cpr.s1.impeller_outlet_width[irow_out])
            cpr.s1.LE_Clearance.append(data_in.loc[case_number].at["LE_Clearance_s1"])
            cpr.s1.TE_Clearance.append(data_in.loc[case_number].at["TE_Clearance_s1"])
            cpr.s1.TE_blade_ang_hub.append(data_in.loc[case_number].at["TE_blade_ang_hub_s1"])
            cpr.s1.TE_blade_ang_tip.append(data_in.loc[case_number].at["TE_blade_ang_tip_s1"])
        
    # make pretty in one table
    data_out_final = pd.DataFrame({
        "TE_blade_ang_hub": cpr.s1.TE_blade_ang_hub,
        "TE_blade_ang_tip": cpr.s1.TE_blade_ang_tip,
        "inlet_inclination_ang": cpr.s1.inlet_inclination_ang,
        "number_of_main_blades": cpr.s1.number_of_main_blades,
        "LE_Clearance": cpr.s1.LE_Clearance,
        "TE_Clearance": cpr.s1.TE_Clearance,
        "shroud_rad": cpr.s1.shroud_rad,
        "inlet_hub_rad": cpr.s1.inlet_hub_rad,
        "impeller_outlet_rad": cpr.s1.impeller_outlet_rad,
        "impeller_outlet_width": cpr.s1.impeller_outlet_width,
        "Rpin": cpr.s1.Rpin,
        "Bpin": cpr.s1.Bpin,
        "outlet_avg_rad": cpr.s1.outlet_avg_rad,
        "volute_throat_area": cpr.volute_throat_area,
        "volute_exit_pipe_dia": cpr.volute_exit_pipe_dia,
        "rpm": cpr.rpm,
        "mass_flow_rate": cpr.mass_flow_rate,
        "isetropic_efficiency": cpr.isetropic_efficiency,
        "pressure_ratio": cpr.pressure_ratio,    
        "power": cpr.power
    })



