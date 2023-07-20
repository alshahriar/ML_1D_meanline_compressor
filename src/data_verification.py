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
    Checking the original input conditions on the old dataset
@Date: 
    July 7 2023
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
# %%
import os
import shutil
from datetime import datetime
import re
# Libs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import imageio
from get_input_parameter_range import get_input_parameter_range

# %%
def clean_data_single_cond(df,minv,maxv,v_txt):
    nRows_out = len(df.index)
    flag_rm = np.zeros(((nRows_out,1)),dtype = bool);
    for irow_out in range(0,nRows_out):
        value = df[v_txt][irow_out]
        if value>maxv:
            flag_rm[irow_out] = bool(True)
        if value<minv:
            flag_rm[irow_out] = bool(True)    

    df["Outside range"] = flag_rm
    df_in_range = df[df["Outside range"] == False]
    df_out_range = df[df["Outside range"] == True]
    print("Outside range found: "+ str(len(df_out_range.index)))
    df_in_range = df_in_range.drop(["Outside range"],axis=1) # droping the extra column
    df_out_range = df_out_range.drop(["Outside range"],axis=1)
    return df_in_range, df_out_range
# %% Loading

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

titles = ["Inlet Mass Flow [kg/s]", "RPM", "Number of Main Blades", "TE Hub Blade angle [deg(m)]", 
          "TE Tip Blade angle [deg(m)]", "LE Clearance [m]", "TE Clearance [m]", "Inclination angle [deg(delta)]", 
          "Imp. Hub Radius [m]", "Imp. Shroud Radius [m]", "Imp. Outlet Radius [m]", "Imp. Outlet Width [m]",
          "Pinch radius (Rpin) [m]", "Pinch width (Bpin) [m]", "Diff. Outlet Radius [m]", "Diff. Outlet Width [m]",
          "Volute Throat Area [m^2]", "Volute Exit Diameter [m]", "Pressure Ratio", "Power (W)", "Efficiency", "eta_group", "pr_group"]


# %%
