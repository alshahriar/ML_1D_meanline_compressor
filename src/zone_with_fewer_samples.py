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

"""Detail description
@Usage:
    
@Date: 
    June 30 2023
@Files
    Required:
        Old CSV files
@Output
    Files: Accumulated old data (both training and testing) in new format
"""
# %% Section 0: Loading modules and libraries
# Built-in/Generic Imports
import os
import shutil
from datetime import datetime
# Libs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re

# User-owned modules
from get_input_parameter_range import get_input_parameter_range
from clean_data import clean_data
from clean_data_single_cond import clean_data_single_cond
from find_data_index import find_data_index_in_between
# %% Section 1: Reading old data
batch_number = 173
train_data_dir = r"../training_data"
test_data_dir = r"../testing_data"    
train_data_fname = "training_batch_"+str(batch_number)+".csv"
test_data_fname = "testing_batch_"+str(batch_number)+".csv"
train_full_dir = os.path.join(train_data_dir, train_data_fname)
test_full_dir = os.path.join(test_data_dir, test_data_fname)
data_old_train = pd.read_csv(train_full_dir,header=None)
data_old_test = pd.read_csv(test_full_dir,header=None)

with open('list_of_final_columns.txt') as f:
    list_of_final_columns = [line.rstrip('\n') for line in f]

data_old_train.columns = list_of_final_columns
data_old_test.columns = list_of_final_columns    

list_of_final_columns = data_old_train.columns.to_list()
list_of_final_columns = [lines.replace(' ','_') for lines in list_of_final_columns] # removing spaces
list_of_final_columns = [re.sub('[!,*)@#%(&$?.^-]', '', lines) for lines in list_of_final_columns] # removing special char except underscore

data_old_train.columns = list_of_final_columns
data_old_test.columns = list_of_final_columns    

fig, ax2 = plt.subplots()
ax2.plot(data_old_train.Inlet_mass_flow,data_old_train.Machine_pressure_ratio_TS,".", markersize=1)
plt.xlabel('mass flow rate [kg/s]')
plt.ylabel('PR (T-S)')
fig.savefig(r'Original_PRTS.png')


# %% 
parameter_txt = r"Inlet_mass_flow";
parameter_min = 0.8;
parameter_max = 1.6;
good_samples, defected_samples = find_data_index_in_between(data_old_train, \
               parameter_min, parameter_max, parameter_txt)
good_samples = good_samples.reset_index(drop=True)
parameter_txt = r"Isentropic_machine_efficiency_TS";
parameter_min = 0.5;
parameter_max = 0.7;
zoned_data, defected_samples = find_data_index_in_between(good_samples, \
               parameter_min, parameter_max, parameter_txt)

input_bound_np,input_bound_df = get_input_parameter_range()
# %%
nCol = len(zoned_data.columns)
for iCol in range(0,nCol-3):
    colName = zoned_data.columns[iCol]
    plt.figure()
    plt.plot(zoned_data.Inlet_mass_flow,zoned_data[colName].values,".", markersize=1)
    plt.xlabel("Inlet mass flow rate")
    plt.ylabel(input_bound_df.index[iCol])
    plt.xlim(input_bound_df.loc["Inlet mass flow"].values)
    plt.ylim(input_bound_df.iloc[iCol].values)
    fname = "zone_"+input_bound_df.index[iCol]+"flowrate.png"
    plt.savefig(fname)
    plt.close()
    
plt.figure()
plt.plot(zoned_data.Inlet_mass_flow,zoned_data.Isentropic_machine_efficiency_TS,".", markersize=1)
plt.xlabel("Inlet mass flow rate")
plt.ylabel("efficiency")
plt.xlim(input_bound_df.loc["Inlet mass flow"].values)
plt.ylim([0.5,0.95])
fname = "zone_eff_flowrate.png"
plt.savefig(fname)
plt.close()


plt.figure()
plt.plot(zoned_data.Inlet_mass_flow,zoned_data.Machine_pressure_ratio_TS,".", markersize=1)
plt.xlabel("Inlet mass flow rate")
plt.ylabel("PR TS")
plt.xlim(input_bound_df.loc["Inlet mass flow"].values)
plt.ylim([1,5])
fname = "zone_PR_flowrate.png"
plt.savefig(fname)
plt.close()


plt.figure()
plt.plot(zoned_data.Inlet_mass_flow,zoned_data.Machine_power,".", markersize=1)
plt.xlabel("Inlet mass flow rate")
plt.ylabel("Machine power")
plt.xlim(input_bound_df.loc["Inlet mass flow"].values)
# plt.ylim([0.5,0.95])
fname = "zone_Machine_power_flowrate.png"
plt.savefig(fname)
plt.close()
    
