# Author: Al Shahriar
# June 28 2023
# %%
import numpy as np
import pandas as pd
from compressor import Compressor
# %% Reading the data
# Inputs to TT
batch_number = 0
if 0:
    # inside TT, right click on the paramatric study table, select export as csv
    data_in = pd.read_csv("../tt_input/batch_"+str(batch_number)+".csv").T
    read_method = 0
else:
    # if created using py script
    data_in = pd.read_pickle("random_parameters.pkl")
    read_method = 1
# Outputs of TT
data_out_raw = pd.read_csv("../tt_output/batch_"+str(batch_number)+".csv",index_col='Parameter').T
# %% 
# Removing unnecessary rows
data_out = data_out_raw.drop(["Object","Unit","Run1_dp"],axis = 0)

# renaming the column - one column has duplicate name
data_out_cols = data_out.columns.tolist()
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

# %% Data indexing
# TT does not return the results for the failed cases in the parametric study
# column number in the csv - input for TT
i_TE_blade_ang_hub_in = 0;
i_TE_blade_ang_tip_in = 1;
i_LE_Clearance_in = 4;
i_TE_Clearance_in = 5;
i_Rpin_by_Rin_in = 12;
i_Bpin_by_Rin_in = 13;

temp_Rpin = [];
temp_Bpin = [];
temp_LE_Clearance = [];
temp_TE_Clearance = [];
temp_TE_blade_ang_hub = [];
temp_TE_blade_ang_tip = [];

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
        temp_Rpin.append(data_in.iloc[case_number].at["Rpin_s1"]*data_out[impl_radius_txt][irow_out])
        temp_Bpin.append(data_in.iloc[case_number].at["Bpin_s1"]*data_out[impl_width_txt][irow_out])
        temp_LE_Clearance.append(data_in.iloc[case_number].at["LE_Clearance_s1"])
        temp_TE_Clearance.append(data_in.iloc[case_number].at["TE_Clearance_s1"])
        temp_TE_blade_ang_hub.append(data_in.iloc[case_number].at["TE_blade_ang_hub_s1"])
        temp_TE_blade_ang_tip.append(data_in.iloc[case_number].at["TE_blade_ang_tip_s1"])
        

data_out["Rpin"] = temp_Rpin
data_out["Bpin"] = temp_Bpin
data_out["LE Clearance"] = temp_LE_Clearance
data_out["TE Clearance"] = temp_TE_Clearance
data_out[TE_blade_ang_hub_txt] = temp_TE_blade_ang_hub
data_out[TE_blade_ang_tip_txt] = temp_TE_blade_ang_tip
data_out_cols = data_out.columns.tolist()
# %% Data cleaning

pressure_ratio_min = 1.0;
pressure_ratio_max = 5.0;

efficiency_min = 0.5;
efficiency_max = 1.0;

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

# %% Seperating the testing and training data
nRows = len(data_out_filtered)
nTrain = int(0.8*nRows)

data_ordered = data_out_filtered.iloc[:, \
                                     [i_mass_flow_rate_out,i_RPM_out,i_n_main_blades_out, \
                                      i_TE_blade_ang_hub_out, i_TE_blade_ang_tip_out, \
                                      i_LE_Clearance_out,i_TE_Clearance_out, \
                                      i_inclination_angle_out, i_hub_radius_out, i_shroud_radius_out, \
                                      i_impl_radius_out, i_impl_width_out, i_Rpin_out, i_Bpin_out, \
                                      i_diff_radius_out, i_diff_width_out, \
                                      i_throat_area_out, i_pipe_diameter_out, \
                                      i_pressure_ratio_out,i_machine_power,i_efficiency_out]]
data_train = data_ordered.iloc[:nTrain,:]    
data_test = data_ordered.iloc[nTrain+1:,:]
data_train.to_csv('../training_data/train_parameters.csv', index=False)
data_test.to_csv('../testing_data/test_parameters.csv', index=False)
data_train.to_pickle("../training_data/train_parameters.pkl")
data_test.to_pickle("../testing_data/test_parameters.pkl")
# %% Trying something with Compressor class
# Not necessary - optional
cpr = Compressor() 
if 1:
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
            cpr.s1.Rpin.append(data_in.iloc[case_number].at["Rpin_s1"]*cpr.s1.impeller_outlet_rad[irow_out])
            cpr.s1.Bpin.append(data_in.iloc[case_number].at["Bpin_s1"]*cpr.s1.impeller_outlet_width[irow_out])
            cpr.s1.LE_Clearance.append(data_in.iloc[case_number].at["LE_Clearance_s1"])
            cpr.s1.TE_Clearance.append(data_in.iloc[case_number].at["TE_Clearance_s1"])
            cpr.s1.TE_blade_ang_hub.append(data_in.iloc[case_number].at["TE_blade_ang_hub_s1"])
            cpr.s1.TE_blade_ang_tip.append(data_in.iloc[case_number].at["TE_blade_ang_tip_s1"])
        
    # make pretty
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



