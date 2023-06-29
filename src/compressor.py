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