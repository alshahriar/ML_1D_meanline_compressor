import numpy as np
import pandas as pd

def get_param_bounds_np():
    stage1_inlet_in_hub_radius = [0.004204, 0.005138]
    stage1_inlet_in_shroud_radius = [0.017415, 0.021285]
    
    stage1_impeller_blade_number_of_main_blades = [6, 10]
    stage1_impeller_blade_hubSect_TE_blade_ang = [-60, 15]
    stage1_impeller_blade_tipSect_TE_blade_ang = [-60, 15]
    stage1_impeller_in_inclination_angle = [-10, 10]
    stage1_LE_Clearance = [0.0001, 0.0004]
    stage1_TE_Clearance = [0.0001, 0.0004]
    stage1_impeller_out_outlet_radius = [0.03235, 0.039538]
    stage1_impeller_out_outlet_width = [0.004098, 0.005009]
    
    # Diffuser stage 1
    stage1_vaneless1_Rpin = [0.03235, 0.039538]
    stage1_vaneless1_Bpin = [0.004098, 0.005009]
    stage1_vaneless1_outlet_rad = [0.061748, 0.07547]
    stage1_vaneless1_outlet_width = [0.0025, 0.0035]
    
    # Return channel
    stage1_return_channel_rc_crossover_roc = [0.0035, 0.0045]
    stage1_return_channel_rc_deswirl_le_width = [0.003011, 0.003457]
    stage1_return_channel_rc_deswirl_te_width = [0.004228, 0.005167]
    stage1_return_channel_rc_deswirl_te_rad = [0.030875, 0.037736]
    stage1_return_channel_rc_deswirl_blade_num_blades = [13, 30]
    stage1_return_channel_rc_outlet_bend_axial_len = [0.019913, 0.024339]
    stage1_return_channel_rc_outlet_width = [0.01323, 0.01617]
    
    # Stage 2
    stage2_inlet_in_hub_radius = [0.004204, 0.005138]   ####
    stage2_inlet_in_shroud_radius = [0.017415, 0.021285] #### maybe delete these
    
    stage2_impeller_blade_number_of_main_blades = [6, 10]
    stage2_impeller_blade_hubSect_TE_blade_ang = [-60, 15]
    stage2_impeller_blade_tipSect_TE_blade_ang = [-60, 15]
    stage2_impeller_in_inclination_angle = [-10, 10]
    stage2_LE_Clearance = [0.0001, 0.0004]
    stage2_TE_Clearance = [0.0001, 0.0004]    
    stage2_impeller_out_outlet_radius = [0.03235, 0.039538]
    stage2_impeller_out_outlet_width = [0.004098, 0.005009]
    
    # Diffuser stage 1
    stage2_vaneless1_Rpin = [0.03235, 0.039538]
    stage2_vaneless1_Bpin = [0.004098, 0.005009]
    stage2_vaneless1_outlet_rad = [0.061748, 0.07547]
    stage2_vaneless1_outlet_width = [0.002, 0.004]
    
    # Volute
    stage2_volute_throat_area = [0.000523, 0.000639]
    stage2_volute_exit_pipe_length = [0.09027, 0.11033]
    stage2_volute_exit_pipe_diameter = [0.027, 0.033]
    
    # operation condition
    rpm = [30000, 50000]
    mass_flow_rate = [0.4, 1.6]
    
    # ######################################################################################################################
    # Generate random numbers in appropriate ranges for each parameter
    # ######################################################################################################################
    
    # Parameter ranges in correct order, comma separated
    param_ranges = np.array([stage1_inlet_in_hub_radius,
        stage1_inlet_in_shroud_radius,
           
        stage1_impeller_blade_number_of_main_blades,
        stage1_impeller_blade_hubSect_TE_blade_ang,
        stage1_impeller_blade_tipSect_TE_blade_ang,
        stage1_impeller_in_inclination_angle,
        stage1_LE_Clearance,
        stage1_TE_Clearance,
        stage1_impeller_out_outlet_radius,
        stage1_impeller_out_outlet_width,
           
        stage1_vaneless1_Rpin,
        stage1_vaneless1_Bpin,
        stage1_vaneless1_outlet_rad,
        stage1_vaneless1_outlet_width,
           
        stage1_return_channel_rc_crossover_roc,
        stage1_return_channel_rc_deswirl_le_width,
        stage1_return_channel_rc_deswirl_te_width,
        stage1_return_channel_rc_deswirl_te_rad,
        stage1_return_channel_rc_deswirl_blade_num_blades,
        stage1_return_channel_rc_outlet_bend_axial_len,
        stage1_return_channel_rc_outlet_width,
           
        stage2_inlet_in_hub_radius,
        stage2_inlet_in_shroud_radius,
           
        stage2_impeller_blade_number_of_main_blades,
        stage2_impeller_blade_hubSect_TE_blade_ang,
        stage2_impeller_blade_tipSect_TE_blade_ang,
        stage2_impeller_in_inclination_angle,
        stage2_LE_Clearance,
        stage2_TE_Clearance,
        stage2_impeller_out_outlet_radius,
        stage2_impeller_out_outlet_width,
           
        stage2_vaneless1_Rpin,
        stage2_vaneless1_Bpin,
        stage2_vaneless1_outlet_rad,
        stage2_vaneless1_outlet_width,
           
        stage2_volute_throat_area,
        stage2_volute_exit_pipe_length,
        stage2_volute_exit_pipe_diameter,
           
        rpm,
        mass_flow_rate
    ])
    
    units = ['m',
    'm',
      
    '-',
    'deg',
    'deg',
    'deg',
    'm',
    'm',
    'm',
    'm',
      
    'm',
    'm',
    'm',
    'm',
      
    'm',
    'm',
    'm',
    'm',
    '-',
    'm',
    'm',
      
    'm',
    'm',
      
    '-',
    'deg',
    'deg',
    'deg',
    'm',
    'm',
    'm',
    'm',
      
    'm',
    'm',
    'm',
    'm',
      
    'm^2',
    'm',
    'm',
      
    'rpm',
    'kg/s',
    # leave this at the end
    'Run1_dp']
    return param_ranges,units

def get_col_list(): 
    col_list = ['stage1->impeller->in->Hub radius',
    'stage1->impeller->in->Shroud radius',
    
    'stage1->impeller->blade->Number of main blades',
    'stage1->impeller->blade->hubSect->TE blade Angle',
    'stage1->impeller->blade->tipSect->TE blade Angle',
    'stage1->impeller->in->Inclination angle',
    'stage1->impeller->blade->LE->LE Clearance',
    'stage1->impeller->blade->TE->TE Clearance',    
    'stage1->impeller->out->Outlet radius (avg)',
    'stage1->impeller->out->Outlet width',
    
    'stage1->vaneless1->Rpin',
    'stage1->vaneless1->Bpin',
    'stage1->vaneless1->out->Outlet radius (avg)',
    'stage1->vaneless1->out->Outlet width',
    
    'stage1->return_channel->Crossover R.O.C. at hub',
    'stage1->return_channel->Deswirl LE width',
    'stage1->return_channel->Deswirl TE width',
    'stage1->return_channel->Deswirl TE radius',
    'stage1->return_channel->deswirl->blade->Number of main blades',
    'stage1->return_channel->Outlet bend axial length',
    'stage1->return_channel->Outlet width',
    
    'stage2->impeller->in->Hub radius',
    'stage2->impeller->in->Shroud radius',
    
    'stage2->impeller->blade->Number of main blades',
    'stage2->impeller->blade->hubSect->TE blade Angle',
    'stage2->impeller->blade->tipSect->TE blade Angle',
    'stage2->impeller->in->Inclination angle',
    'stage2->impeller->blade->LE->LE Clearance',
    'stage2->impeller->blade->TE->TE Clearance',        
    'stage2->impeller->out->Outlet radius (avg)',
    'stage2->impeller->out->Outlet width',
    
    'stage2->vaneless1->Rpin',
    'stage2->vaneless1->Bpin',
    'stage2->vaneless1->out->Outlet radius (avg)',
    'stage2->vaneless1->out->Outlet width',
    
    'stage2->volute->Throat area',
    'stage2->volute->Exit pipe length',
    'stage2->volute->Exit pipe diameter',
    
    'Rotational speed',
    'Mass flow rate'
    ]
    return col_list


def get_col_list_json(): 
    col_list = [r'1d/1d_Machine/stage1/impeller/in/Rh',
    r'1d/1d_Machine/stage1/impeller/in/Rs',
    
    r'1d/1d_Machine/stage1/impeller/blade/numMainBlade',
    r'1d/1d_Machine/stage1/impeller/blade/hubSect/beta2b',
    r'1d/1d_Machine/stage1/impeller/blade/tipSect/beta2b',
    r'1d/1d_Machine/stage1/impeller/in/phi',
    r'1d/1d_Machine/stage1/impeller/blade/LE/clr',
    r'1d/1d_Machine/stage1/impeller/blade/TE/clr',
    r'1d/1d_Machine/stage1/impeller/out/Ra',
    r'1d/1d_Machine/stage1/impeller/out/b',
    
    r'1d/1d_Machine/stage1/vaneless1/Rpin',
    r'1d/1d_Machine/stage1/vaneless1/Bpin',
    r'1d/1d_Machine/stage1/vaneless1/out/Ra',
    r'1d/1d_Machine/stage1/vaneless1/out/b',
    
    r'1d/1d_Machine/stage1/return_channel/R_bend',
    r'1d/1d_Machine/stage1/return_channel/Ble',
    r'1d/1d_Machine/stage1/return_channel/Bte',
    r'1d/1d_Machine/stage1/return_channel/Rte',
    r'1d/1d_Machine/stage1/return_channel/deswirl/blade/numMainBlade',
    r'1d/1d_Machine/stage1/return_channel/L_exit',
    r'1d/1d_Machine/stage1/return_channel/Bex',
    
    r'1d/1d_Machine/stage2/impeller/in/Rh',
    r'1d/1d_Machine/stage2/impeller/in/Rs',
    
    r'1d/1d_Machine/stage2/impeller/blade/numMainBlade',
    r'1d/1d_Machine/stage2/impeller/blade/hubSect/beta2b',
    r'1d/1d_Machine/stage2/impeller/blade/tipSect/beta2b',
    r'1d/1d_Machine/stage2/impeller/in/phi',
    r'1d/1d_Machine/stage2/impeller/blade/LE/clr',
    r'1d/1d_Machine/stage2/impeller/blade/TE/clr',    
    r'1d/1d_Machine/stage2/impeller/out/Ra',
    r'1d/1d_Machine/stage2/impeller/out/b',
    
    r'1d/1d_Machine/stage2/vaneless1/Rpin',
    r'1d/1d_Machine/stage2/vaneless1/Bpin',
    r'1d/1d_Machine/stage2/vaneless1/out/Ra',
    r'1d/1d_Machine/stage2/vaneless1/out/b',
    
    r'1d/1d_Machine/stage2/volute/throatArea',
    r'1d/1d_Machine/stage2/volute/exitPipeLength',
    r'1d/1d_Machine/stage2/volute/exitPipeDiameter',
    
    r'1d/SolverSetting/opCondition/dp/RPM',
    r'1d/SolverSetting/opCondition/dp/minlet'
    ]
    return col_list