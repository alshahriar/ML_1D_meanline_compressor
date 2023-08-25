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
"""
# %%
# Libs
import numpy as np
import pandas as pd
# %%
def clean_data_single_cond(df,minv,maxv,v_txt):
    nRows_out = len(df.index)
    flag_rm = np.zeros(((nRows_out,1)),dtype = bool);
    value_all = df[v_txt].values
    for irow_out in range(0,nRows_out):
        value = value_all[irow_out]
        # print(value, maxv, minv)
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