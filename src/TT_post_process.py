import numpy as np
import pandas as pd

# %%
class Compressor:
    # compressor information
    def __init__(self):
        # create stages
        self.s1 = self.Stage()
        #self.s2 = self.Stage()
        self.volute_throat_area = 1.0
        self.volute_exit_pipe_dia = 1.0
        self.rpm = 10000.0
        self.mass_flow_rate = 1.0
        self.isetropic_efficiency = 0.8
        self.pressure_ratio = 2.0
        self.power = 1
    # stage information
    class Stage:
        def __init__(self):
            # create stage information
            self.TE_blade_ang_hub = []
            self.TE_blade_ang_tip = []
            self.inlet_inclination_ang = 1.0
            self.number_of_main_blades = 1
            self.LE_Clearance = []
            self.TE_Clearance = []
            self.shroud_rad = 1.0
            self.inlet_hub_rad = 1.0
            self.impeller_outlet_rad = 1.0
            self.impeller_outlet_width = 1.0
            self.Rpin = []
            self.Bpin = []
            self.outlet_avg_rad = 1.0

cpr = Compressor() 

# %%
batch_number = 0
data_in = pd.read_csv("../tt_input/batch_"+str(batch_number)+".csv").T
data_out_raw = pd.read_csv("../tt_output/batch_"+str(batch_number)+".csv",index_col='Parameter').T
# Rows to remove
data_out = data_out_raw.drop(["Object","Unit","Run1_dp"],axis = 0)
# %%

cpr.rpm = data_out["RPM"].values
cpr.mass_flow_rate = data_out["Inlet mass flow"].values
cpr.isetropic_efficiency = data_out["Isentropic machine efficiency (T-S)"].values
cpr.pressure_ratio = data_out["Machine pressure ratio (T-S)"].values
cpr.power = data_out["Machine power"].values

cpr.s1.inlet_inclination_ang = data_out["Inclination angle"].values
cpr.s1.number_of_main_blades = data_out["Number of main blades"].values
cpr.s1.shroud_rad = data_out["Shroud radius"].values
cpr.s1.inlet_hub_rad = data_out["Hub radius"].values
cpr.s1.impeller_outlet_rad = data_out["Outlet radius (avg)"].values
cpr.s1.impeller_outlet_width = data_out["Outlet width"].values

cpr.volute_throat_area = data_out["Throat area"].values
cpr.volute_exit_pipe_dia = data_out["Exit pipe diameter"].values

# column number in the csv - input for TT
TE_blade_ang_hub_index = 0;
TE_blade_ang_tip_index = 1;
LE_Clearance_index = 4;
TE_Clearance_index = 5;
Rpin_Rin_index = 12;
Bpin_Rin_index = 13;

nRows = len(data_out.index)
for irow in range(0,nRows):
    row = data_out.index[irow]
    nL = len(row)
    case_number = int(row[18:nL])
    #print(case_number)
    cpr.s1.Rpin.append(data_in.loc[str(case_number)].at[Rpin_Rin_index]*cpr.s1.impeller_outlet_rad[irow])
    cpr.s1.Bpin.append(data_in.loc[str(case_number)].at[Bpin_Rin_index]*cpr.s1.impeller_outlet_width[irow])
    cpr.s1.LE_Clearance.append(data_in.loc[str(case_number)].at[LE_Clearance_index])
    cpr.s1.TE_Clearance.append(data_in.loc[str(case_number)].at[TE_Clearance_index])
    cpr.s1.TE_blade_ang_hub.append(data_in.loc[str(case_number)].at[TE_blade_ang_hub_index])
    cpr.s1.TE_blade_ang_tip.append(data_in.loc[str(case_number)].at[TE_blade_ang_tip_index])

# make pretty
df = pd.DataFrame({
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
