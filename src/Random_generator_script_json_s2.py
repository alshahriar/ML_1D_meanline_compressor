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
param_bounds_df = pd.DataFrame(param_ranges.T, columns=col_list_json).T
lower = param_ranges[:, 0]
widths = param_ranges[:, 1] - param_ranges[:, 0]
# Random samples in appropriate ranges
samples = (lower + widths * np.random.random(size=(batch_size, widths.shape[0]))).round(decimals=6)
parameters = pd.DataFrame(samples, columns=col_list_json)
parametersT = parameters.T
tag_txt = 'Object Parameter Unit' 
# %%
parameters = parameters.round({'1d/1d_Machine/stage1/impeller/blade/numMainBlade': 0})
parameters = parameters.round({'1d/1d_Machine/stage1/return_channel/deswirl/blade/numMainBlade': 0})
parameters = parameters.round({'1d/1d_Machine/stage2/impeller/blade/numMainBlade': 0})

# Round inlet blade angles to 2 decimal places
parameters = parameters.round({'1d/1d_Machine/stage1/impeller/blade/hubSect/beta2b': 2,
                               '1d/1d_Machine/stage1/impeller/blade/tipSect/beta2b': 2,
                               '1d/1d_Machine/stage1/impeller/in/phi': 2})

# Round number of s2 blades to the nearest int
parameters = parameters.round({'1d/1d_Machine/stage2/impeller/blade/hubSect/beta2b': 2,
                               '1d/1d_Machine/stage2/impeller/blade/tipSect/beta2b': 2,
                               '1d/1d_Machine/stage2/impeller/in/phi': 2})

parameters[tag_txt] = (parameters.index) + 1    # add counter column (1 - 1000)
parameters[tag_txt] = "Run1_para_study_od" + parameters[tag_txt].apply(str)
# %%
if(1):
    # fixing TurboTides issue - converting in radians
    col_2_fix = "1d/1d_Machine/stage1/impeller/blade/hubSect/beta2b"
    parameters[col_2_fix] = parameters[col_2_fix]/57.29569653074808
    col_2_fix = "1d/1d_Machine/stage1/impeller/blade/tipSect/beta2b"
    parameters[col_2_fix] = parameters[col_2_fix]/57.29569653074808
    col_2_fix = "1d/1d_Machine/stage1/impeller/in/phi"
    parameters[col_2_fix] = parameters[col_2_fix]/57.29569653074808
    
    col_2_fix = "1d/1d_Machine/stage2/impeller/blade/hubSect/beta2b"
    parameters[col_2_fix] = parameters[col_2_fix]/57.29569653074808
    col_2_fix = "1d/1d_Machine/stage2/impeller/blade/tipSect/beta2b"
    parameters[col_2_fix] = parameters[col_2_fix]/57.29569653074808
    col_2_fix = "1d/1d_Machine/stage2/impeller/in/phi"
    parameters[col_2_fix] = parameters[col_2_fix]/57.29569653074808


# %%
exportData = parameters.to_dict('list')
with open('..\\tt_input_s2\\Batch_'+str(batch_no)+'.json', 'w') as f:
    json.dump(exportData, f, allow_nan=True)

if not os.path.exists('Sample_Batches'):
    os.mkdir('Sample_Batches')
parameters.index = parameters.index + 1     # change dataframe index to start at 1
parameters.to_pickle("../tt_input_s2/Batch_json_"+str(batch_no)+".pkl")

col_list.insert(0, tag_txt)
parameters.columns = col_list
parameters.to_pickle("../tt_input_s2/Batch_"+str(batch_no)+".pkl")

parameters = parameters.drop(tag_txt,axis = 1)
parameters.to_csv("../tt_input_s2/Batch_"+str(batch_no)+".csv", index=False)