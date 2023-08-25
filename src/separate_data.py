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

"""Detail description of separate_data.py
@Usage:
    Created new train and test data based on imposed conditions
@Date: 
    June 30 2023
@Files
    Required:
        Accumulated Pickle file that contains all data
@Output
    Files: Accumulated data (both training and testing) in new format
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
from get_input_parameter_range import get_input_parameter_range
from clean_data import clean_data
# %% Section 1: Reading compiled data
train_data_dir = r"../training_data"
test_data_dir = r"../testing_data"    
train_data_fname = "training_parameters.pkl"
test_data_fname = "testing_parameters.pkl"

train_full_dir = os.path.join(train_data_dir, train_data_fname)
test_full_dir = os.path.join(test_data_dir, test_data_fname)
train_data_df_raw = pd.read_pickle(train_full_dir)
test_data_df_raw = pd.read_pickle(test_full_dir)

# %% Filtering and cleaning data
pressure_ratio_txt = r"Machine pressure ratio (T-S)";
efficiency_txt = r"Isentropic machine efficiency (T-S)"

train_data_df_raw = train_data_df_raw.reset_index()
test_data_df_raw = test_data_df_raw.reset_index()
train_data_df_raw = train_data_df_raw.drop("index", axis=1)
test_data_df_raw = test_data_df_raw.drop("index", axis=1)


pressure_ratio_min = 1.0;
pressure_ratio_max = 5.0;
efficiency_min = 0.5;
efficiency_max = 0.95;
train_data_df, data_bad_train = clean_data(train_data_df_raw,efficiency_min,efficiency_max, \
               pressure_ratio_min, pressure_ratio_max, \
                   efficiency_txt, pressure_ratio_txt)
test_data_df, data_bad_test = clean_data(test_data_df_raw,efficiency_min,efficiency_max, \
               pressure_ratio_min, pressure_ratio_max, \
                   efficiency_txt, pressure_ratio_txt)

# %% Section 2: Saving
file_format  = ".pkl"
tag = "_95"
# writing the file
train_data_dir = r"../training_data"
test_data_dir = r"../testing_data"    
train_data_fname = "training_parameters"+tag+".pkl"
test_data_fname = "testing_parameters"+tag+".pkl"
train_full_dir = os.path.join(train_data_dir, train_data_fname)
test_full_dir = os.path.join(test_data_dir, test_data_fname)
train_data_df.to_pickle(train_full_dir)
test_data_df.to_pickle(test_full_dir)