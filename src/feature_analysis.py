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
    Histogram to check if there are any biases in the features
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
from mpl_toolkits import mplot3d
# import imageio
# %% Loading

n_features = 18 #Number of input columns
n_labels = 3 #Number of output columns

train_data_dir = r"../training_data"
test_data_dir = r"../testing_data"    
train_data_fname = r"training_parameters.pkl"
test_data_fname = r"testing_parameters.pkl"
train_full_dir = os.path.join(train_data_dir, train_data_fname)
test_full_dir = os.path.join(test_data_dir, test_data_fname)
read_method  = 1

train_data_df = pd.read_pickle(train_full_dir)
test_data_df = pd.read_pickle(test_full_dir)
# train_data = train_data_df.to_numpy()
# test_data = test_data_df.to_numpy()
list_of_final_columns = train_data_df.columns.to_list()
list_of_final_columns = [lines.replace(' ','_') for lines in list_of_final_columns] # removing spaces
list_of_final_columns = [re.sub('[!,*)@#%(&$?.^-]', '', lines) for lines in list_of_final_columns] # removing special char except underscore
train_data_df.columns = list_of_final_columns;
test_data_df.columns = list_of_final_columns;

titles = ["Inlet Mass Flow [kg/s]", "RPM", "Number of Main Blades", "TE Hub Blade angle [deg(m)]", 
          "TE Tip Blade angle [deg(m)]", "LE Clearance [m]", "TE Clearance [m]", "Inclination angle [deg(delta)]", 
          "Imp. Hub Radius [m]", "Imp. Shroud Radius [m]", "Imp. Outlet Radius [m]", "Imp. Outlet Width [m]",
          "Pinch radius (Rpin) [m]", "Pinch width (Bpin) [m]", "Diff. Outlet Radius [m]", "Diff. Outlet Width [m]",
          "Volute Throat Area [m^2]", "Volute Exit Diameter [m]", "Pressure Ratio", "Power (W)", "Efficiency", "eta_group", "pr_group"]

# %% Plotting histograms - eta group
n_groups = 3;
max_eta = train_data_df.Isentropic_machine_efficiency_TS.max()
min_eta = train_data_df.Isentropic_machine_efficiency_TS.min()
delta_eta = np.linspace(min_eta, max_eta, n_groups+1)
eta_values = np.array(train_data_df.Isentropic_machine_efficiency_TS.values)

range1_txt = str(round(delta_eta[0],2))+"-"+str(round(delta_eta[1],2))
range2_txt = str(round(delta_eta[1],2))+"-"+str(round(delta_eta[2],2))
range3_txt = str(round(delta_eta[2],2))+"-"+str(round(delta_eta[3],2))

bool_val = (eta_values < delta_eta[1])
eta_group = np.multiply(bool_val, 1)

cond1 = ((eta_values >= delta_eta[1]))
cond2 = (eta_values < delta_eta[2])
bool_val = np.all([cond1,cond2],  axis=0)
eta_group = eta_group + np.multiply(bool_val, 2)

cond1 = ((eta_values >= delta_eta[2]))
cond2 = (eta_values <= delta_eta[3])
bool_val = np.all([cond1,cond2],  axis=0)
eta_group = eta_group + np.multiply(bool_val, 3)

zero_values = np.argwhere(eta_group == 0)

train_data_df["eta_group"] = eta_group
# fig = train_data_df.pivot(columns='eta_group').Inlet_mass_flow.plot(kind = 'hist', stacked=True, bins=10, edgecolor="black")
new_df = pd.DataFrame()
#new_df["Index"] = range(len(train_data_df))
#for (columnName, columnData) in train_data_df.iteritems():
for i, columnName in enumerate(train_data_df):
    columnData = np.array(train_data_df[columnName].values)
    print('Column Name : ', columnName)
    new_df["eta_group"] = np.array(eta_group)
    new_df["data"] = columnData
    new_df["data"] = new_df["data"].astype(float) # important
    fig = new_df.pivot(columns='eta_group').data.plot(kind = 'hist', stacked=True, bins=10, edgecolor="black")
    # print('Column Contents : ', columnData.values)
    # plt.hist(columnData.values,edgecolor="black",bins=10)
    xlbl_txt = titles[i]
    plt.xlabel(xlbl_txt)
    plt.ylabel('count')
    plt.legend([r"$\eta$ = "+range1_txt,r"$\eta$ = "+range2_txt,r"$\eta$ = "+range3_txt])
    plt.savefig(os.path.join(r"hist_figures", "eta", columnName+r".png"))
    plt.close()

# %% Input vs efficiency
for i, columnName in enumerate(train_data_df):
    columnData = train_data_df[columnName]
    print('Column Name : ', columnName)
    xlbl_txt = titles[i]
    plt.xlabel(xlbl_txt)
    plt.ylabel('$\eta$')
    fig, ax2 = plt.subplots()
    ax2.plot(columnData,train_data_df.Isentropic_machine_efficiency_TS,".", markersize=1)
    plt.savefig(r"input_eta"+columnName+r".png")
    plt.close()

# %% Input parameters relations - PR groups

n_groups = 3;
max_pr = train_data_df.Machine_pressure_ratio_TS.max()
min_pr = train_data_df.Machine_pressure_ratio_TS.min()
delta_pr = np.linspace(min_pr, max_pr, n_groups+1)
pr_values = np.array(train_data_df.Machine_pressure_ratio_TS.values)

range1_txt = str(round(delta_pr[0],2))+"-"+str(round(delta_pr[1],2))
range2_txt = str(round(delta_pr[1],2))+"-"+str(round(delta_pr[2],2))
range3_txt = str(round(delta_pr[2],2))+"-"+str(round(delta_pr[3],2))

bool_val = (pr_values < delta_pr[1])
pr_group = np.multiply(bool_val, 1)

cond1 = ((pr_values >= delta_pr[1]))
cond2 = (pr_values < delta_pr[2])
bool_val = np.all([cond1,cond2],  axis=0)
pr_group = pr_group + np.multiply(bool_val, 2)

cond1 = ((pr_values >= delta_pr[2]))
cond2 = (pr_values <= delta_pr[3])
bool_val = np.all([cond1,cond2],  axis=0)
pr_group = pr_group + np.multiply(bool_val, 3)

zero_values = np.argwhere(pr_group == 0)

train_data_df["pr_group"] = pr_group
# fig = train_data_df.pivot(columns='pr_group').Inlet_mass_flow.plot(kind = 'hist', stacked=True, bins=10, edgecolor="black")

for i, columnName in enumerate(train_data_df):
    columnData = train_data_df[columnName].values
    print('Column Name : ', columnName)
    new_df["pr_group"] = pr_group
    new_df["data"] = columnData
    new_df["data"] = new_df["data"].astype(float)    
    fig = new_df.pivot(columns='pr_group').data.plot(kind = 'hist', stacked=True, bins=10, edgecolor="black")
    # print('Column Contents : ', columnData.values)
    # plt.hist(columnData.values,edgecolor="black",bins=10)
    xlbl_txt = titles[i]
    plt.xlabel(xlbl_txt)
    plt.ylabel('count')
    plt.legend([r"PR = "+range1_txt,r"PR = "+range2_txt,r"PR = "+range3_txt])
    plt.savefig(os.path.join(r"hist_figures", "pr", columnName+r".png"))
    plt.close()
    
# %% Output value relations
fig, ax2 = plt.subplots()
ax2.plot(train_data_df.Inlet_mass_flow,train_data_df.Isentropic_machine_efficiency_TS,".", markersize=1)
plt.xlabel('mass flow rate [kg/s]')
plt.ylabel('$\eta$ (T-S)')
fig.savefig(r'etaTS.png')

fig, ax2 = plt.subplots()
ax2.plot(train_data_df.Inlet_mass_flow,train_data_df.Machine_pressure_ratio_TS,".", markersize=1)
plt.xlabel('mass flow rate [kg/s]')
plt.ylabel('PR (T-S)')
fig.savefig(r'PRTS.png')

fig, ax2 = plt.subplots()
ax2.plot(train_data_df.Inlet_mass_flow,train_data_df.Machine_power,".", markersize=1)
plt.xlabel('mass flow rate [kg/s]')
plt.ylabel('Power [W]')
fig.savefig(r'Power.png')

# %%

colorV = "RPM"
x = train_data_df.Isentropic_machine_efficiency_TS.values
y = train_data_df.Machine_pressure_ratio_TS.values
z = train_data_df.Machine_power.values
v = train_data_df[colorV].values

fig = plt.figure(figsize=(8, 8))
ax =  plt.axes(projection = '3d')
img = ax.scatter(x, y, z, c=v,alpha = 0.2, cmap = 'jet')
ax.set_xlabel('$\eta$')
ax.set_ylabel('PR-TS')
ax.set_zlabel('Power')
cbar = plt.colorbar(img)
cbar.set_label(colorV)
ax.view_init(0, 180)
