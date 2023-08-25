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
    Data cleaning
@Date: 
    July 30 2023
"""
# %% Section 0: Loading modules and libraries
# Libs
import numpy as np
import pandas as pd

# %% functions

def clean_data(df,efficiency_min,efficiency_max, \
               pressure_ratio_min, pressure_ratio_max, \
                   efficiency_txt, pressure_ratio_txt):
    nRows_out = len(df.index)

    flag_rm_eta = np.zeros(((nRows_out,1)),dtype = bool);
    flag_rm_pr = np.zeros(((nRows_out,1)),dtype = bool);

    for irow_out in range(0,nRows_out):
        eta = df[efficiency_txt][irow_out]
        if eta>efficiency_max:
            flag_rm_eta[irow_out] = bool(True)
        if eta<efficiency_min:
            flag_rm_eta[irow_out] = bool(True)    

    for irow_out in range(0,nRows_out):
        pr = df[pressure_ratio_txt][irow_out]
        if pr>pressure_ratio_max:
            flag_rm_pr[irow_out] = bool(True)
        if pr<pressure_ratio_min:
            flag_rm_pr[irow_out] = bool(True)

    flag_rm = np.logical_or(flag_rm_eta,flag_rm_pr)
    df["Bad Samples"] = flag_rm
    data_good = df[df["Bad Samples"] == False]
    data_bad = df[df["Bad Samples"] == True]
    print("Bad samples found: "+ str(len(data_bad.index)))
    data_good = data_good.drop(["Bad Samples"],axis=1) # droping the extra column
    data_bad = data_bad.drop(["Bad Samples"],axis=1)
    return data_good, data_bad