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

"""Detail description of convert_old_data_to_new_format.py
@Usage:
    Convert the csv files to pickle format
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
# User-owned modules
from get_input_parameter_range import get_input_parameter_range
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
# %% Section 2: Saving
file_format  = ".pkl"
# writing the file
train_data_dir = r"../training_data"
test_data_dir = r"../testing_data"    
train_data_fname = "training_parameters.pkl"
test_data_fname = "testing_parameters.pkl"
train_full_dir = os.path.join(train_data_dir, train_data_fname)
test_full_dir = os.path.join(test_data_dir, test_data_fname)
data_old_train.to_pickle(train_full_dir)
data_old_test.to_pickle(test_full_dir)