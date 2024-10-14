from abel.classes.rf_accelerator.rf_accelerator import RFAccelerator
from abel.classes.cost_modeled import CostModeled
import scipy.constants as SI
import numpy as np
import copy
import matplotlib.pyplot as plt
from types import SimpleNamespace

class RFAcceleratorCLICopti(RFAccelerator):

    default_num_rf_cells = 24 # [m]
    default_fill_factor = 0.71
    default_rf_frequency = 2e9 # [Hz]

    default_a_n = 0.110022947942206
    default_a_n_delta = 0.016003337882503*2
    default_d_n = 0.160233420548558
    default_d_n_delta = 0.040208386429788*2

    default_num_integration_points = 1000
    
    
    def __init__(self, length=None, nom_energy_gain=None, fill_factor=default_fill_factor, rf_frequency=default_rf_frequency, num_rf_cells=default_num_rf_cells):
        
        # run base class constructor
        super().__init__(length=length, nom_energy_gain=nom_energy_gain, fill_factor=fill_factor, rf_frequency=rf_frequency)

        self.fill_factor = fill_factor
        self.num_rf_cells = num_rf_cells

        self.a_n = self.default_a_n
        self.a_n_delta = self.default_a_n_delta
        self.d_n = self.default_d_n
        self.d_n_delta = self.default_d_n_delta

        self.num_integration_points = self.default_num_integration_points

        self.structure = SimpleNamespace()
        self.structure.length = None
        self.structure.rise_time = None
        self.structure.fill_time = None
        self.structure.pulse_length_total = None
        self.structure.power_unloaded = None
        self.structure.power_loaded = None
        self.structure.power = None
        self.structure.pulse_length_max = None
        self.structure.power_max_allowable = None
        self.structure.voltage_unloaded = None
        self.structure.flattop_efficiency = None
        self.structure.pulse_energy = None
        self.structure.rf_efficiency = None
            
        self.structure.power_profile = SimpleNamespace()
        self.structure.power_profile.t = None
        self.structure.power_profile.P = None

        self.structure.gradient_profile = SimpleNamespace()
        self.structure.gradient_profile.z = None
        self.structure.gradient_profile.Ez = None

        self.structure.database = SimpleNamespace()
        self.structure.database.grid = None
        self.structure.database.cell_first = None
        self.structure.database.cell_mid = None
        self.structure.database.cell_last = None


        

    @RFAccelerator.structure_length.getter
    def structure_length(self) -> float:
        "Gets the length of each individual RF structure [m]"
        return self.structure.length
        
    def track(self, beam0, savedepth=0, runnable=None, verbose=False):

        # store data used power-flow/power use/etc modelling
        self.bunch_charge = beam0.abs_charge()
        if self.bunch_separation is None:
            self.bunch_separation = beam0.bunch_separation
        if self.num_bunches_in_train is None:
            self.num_bunches_in_train = beam0.num_bunches_in_train

        # remake the structure based on updated numbers
        
        import CLICopti
        
        # make database
        cellbase = '/pfs/lustrep2/projappl/project_465000445/software/clicopti/cellBase/TD_12GHz_v1.dat' # TODO: fix this
        database = CLICopti.CellBase.CellBase_linearInterpolation_freqScaling(cellbase, ("a_n","d_n"), self.rf_frequency/1e9)

        # make structure
        RF_structure = CLICopti.RFStructure.AccelStructure_paramSet2_noPsi(database, self.num_rf_cells, self.a_n, self.a_n_delta, self.d_n, self.d_n_delta)
        
        # DB v2 constructed ignoring P/C limit
        RF_structure.uselimit_PC = False

        # perform calculation
        RF_structure.calc_g_integrals()

        # extract structure length
        self.structure.length = RF_structure.getL()

        # extract maximum voltage
        if self.num_bunches_in_train == 1:
            self.structure.power_max_allowable = RF_structure.getMaxAllowablePower_beamTimeFixed(0.0, self.train_duration)
            self.structure.voltage_unloaded = RF_structure.getVoltageUnloaded(self.structure.power_max_allowable)
        else:
            self.structure.power_max_allowable = RF_structure.getMaxAllowablePower_beamTimeFixed(self.average_current_train, self.train_duration)
            self.structure.voltage_unloaded = RF_structure.getVoltageLoaded(self.structure.power_max_allowable, self.average_current_train)
        self.structure.voltage_max = self.structure.voltage_unloaded
            
        # auto optimize gradient
        if self.length is None:
            self.nom_accel_gradient = self.fill_factor*self.structure.voltage_max/self.structure.length

        # extract rise and fill times
        self.structure.rise_time = RF_structure.getTrise()
        self.structure.fill_time = RF_structure.getTfill()
        self.structure.pulse_length_total = 2 * (self.structure.rise_time + self.structure.fill_time) + self.train_duration

        # extract structure power
        self.structure.power_unloaded = RF_structure.getPowerUnloaded(self.voltage_structure)
        self.structure.power_loaded = RF_structure.getPowerLoaded(self.voltage_structure, self.average_current_train)
        if self.num_bunches_in_train == 1:
            self.structure.power = self.structure.power_unloaded
            self.structure.pulse_length_max = RF_structure.getMaxAllowableBeamTime(self.structure.power, 0.0)
        else:
            self.structure.power = self.structure.power_loaded
            self.structure.pulse_length_max = RF_structure.getMaxAllowableBeamTime(self.structure.power, self.average_current_train)

        # extract energy and efficiency
        if self.average_current_train is not None:
            self.structure.flattop_efficiency = RF_structure.getFlattopEfficiency(self.structure.power, self.average_current_train)
        self.structure.pulse_energy = (self.structure.rise_time + self.structure.fill_time + self.train_duration) * self.structure.power
        if self.num_bunches_in_train == 1:
            self.structure.rf_efficiency = self.voltage_structure * self.beam_charge / self.structure.pulse_energy
        else:
            self.structure.rf_efficiency = RF_structure.getTotalEfficiency(self.structure.power, self.average_current_train, self.train_duration)

        # extract power profile
        self.structure.power_profile.t = np.linspace(0,self.structure.pulse_length_total, 100)
        self.structure.power_profile.P = RF_structure.getP_t(self.structure.power_profile.t, self.structure.power, self.train_duration, self.average_current_train)

        # extract gradient profile
        self.structure.gradient_profile.z = RF_structure.getZ_all()
        if self.average_current_train is None:
            self.structure.gradient_profile.Ez = RF_structure.getEz_unloaded_all(self.structure.power)
        else:
            self.structure.gradient_profile.Ez = RF_structure.getEz_loaded_all(self.structure.power, self.average_current_train)

        # extract database grid
        self.structure.database.grid, _ = database.getGrid_meshgrid()
        self.structure.database.a_n = (RF_structure.getCellFirst().a_n, RF_structure.getCellMid().a_n, RF_structure.getCellLast().a_n)
        self.structure.database.d_n = (RF_structure.getCellFirst().d_n, RF_structure.getCellMid().d_n, RF_structure.getCellLast().d_n)
        
        # check the voltage
        if self.voltage_structure is not None and self.voltage_structure > self.get_structure_voltage_max():
            print(f'The structure gradient ({(self.voltage_structure/self.structure.length)/1e6:.1f} MV/m) is too high (max {(self.structure.voltage_max/self.structure.length)/1e6:.1f} MV/m)')
        
        # perform energy increase
        beam = copy.deepcopy(beam0)
        beam.set_Es(beam0.Es() + self.nom_energy_gain)
        
        return super().track(beam, savedepth, runnable, verbose)

    
    def get_structure_pulse_energy(self) -> float:
        "Get the energy requirements for a single pulse for one structure"
        #Note: Model is assuming a pulse like from plotPowerProfile(),
        #      which assumes a drive beam driven pulse in that pulse end is a mirror of the start,
        #      and that the pusle shape can be easily "reorganized" to a square pulse.
        #      As long as Tfill is short, this doesn't matter.
        return self.structure.pulse_energy

    def get_structure_power(self) -> float:
        "Get the peak power [W] required for a single RF structure in the given configuration."
        return self.structure.power

    def get_rf_efficiency(self) -> float:
        """
        Get the RF->beam efficiency as a number between 0 and 1, including the effect due to pulse shape
        Calculated with different methods depending on wehter num_bunches_in_train = 1 or > 1.
        """
        return self.structure.rf_efficiency

    def get_rf_efficiency_flattop(self) -> float:
        "Get the RF->beam efficiency as a number between 0 and 1, ignoring the fill time etc."
        return self.structure.flattop_efficiency

    def get_fill_time(self) -> float:
        "Get the filling time [s] of the structure, i.e. the time for a signal to propagate through the whole structure."
        return self.structure.fill_time

    def get_pulse_length_total(self) -> float:
        "Get the total RF pulse length [s], including rise time, filling time, beam time, and rampdown (drive-beam style)"
        return self.structure.pulse_length_total

    def get_pulse_length_max(self) -> float:
        "Calculates the max train duration [s] before exceeding gradient limits, given power and average_current_train [A]"
        return self.structure.pulse_length_max

    def get_structure_voltage_max(self) -> float:
        "Calculates the maximum structure voltage before exceeding breakdown limts, given the currently selected pulse length and beam current"
        return self.structure.voltage_max
    
    
    #---------------------------------------------------------------------#
    # Plots (CLICopti-based)                                              #
    #=====================================================================#

    def plot_database_points(self, bgData=None):
        "Plot the cell parameters in the database"

        plt.subplots()
        plt.scatter(self.structure.database.grid[0], self.structure.database.grid[1], marker='*', color='red')
        plt.xlabel(r'$a/\lambda$')
        plt.ylabel(r'$d/h$')

        plt.plot(self.structure.database.a_n, self.structure.database.d_n, 's')
        plt.annotate("",(self.structure.database.a_n[0],self.structure.database.d_n[0]), (self.structure.database.a_n[1],self.structure.database.d_n[1]),arrowprops=dict(arrowstyle="->"))
        plt.annotate("",(self.structure.database.a_n[1],self.structure.database.d_n[1]), (self.structure.database.a_n[2],self.structure.database.d_n[2]),arrowprops=dict(arrowstyle="->"))

    def plot_gradient_profile(self) -> float:
        plt.subplots()
        plt.plot(self.structure.gradient_profile.z*1e3, self.structure.gradient_profile.Ez/1e6)
        plt.xlabel('$z$ [mm]')
        plt.ylabel('$E_z$ [MV/m]')

        plt.title(self.make_plot_title())

    def plot_power_profile(self) -> float:
        plt.subplots()
        plt.plot(self.structure.power_profile.t*1e9, self.structure.power_profile.P/1e6)
        plt.xlabel('Time [ns]')
        plt.ylabel('$P_{in}$ [MW]')
        plt.title(self.make_plot_title())

    def make_plot_title(self) -> str:
        "Create a standardized title string for plots"
        tit = self.make_structure_title()
        tit += "\n"
        tit += f"V={self.voltage_structure/1e6:.1f} [MV]"
        tit += f", I={self.average_current_train:.1f} [A]"
        tit += f", <G>={self.gradient_structure/1e6:.1f} [MV/m]"
        return tit

    def make_structure_title(self):
        tit = "DBv2 structure"
        tit += f", N={self.num_rf_cells}"
        tit += f", f0={self.rf_frequency/1e9:.1f} [GHz]"
        tit += f", a_n={self.a_n:.3f} delta={self.a_n_delta:.3f}"
        tit += f", d_n={self.d_n:.3f} delta={self.d_n_delta:.3f}"
        return tit

    
    
    def energy_usage_cooling(self) -> float:
        "Energy usage per shot for cooling [J]"
        
        # calculate cooling efficiency (Carnot engine efficiency)
        room_temperature = 300 # [K]
        efficiency_cooling = self.operating_temperature/room_temperature
        
        energy_left_in_structures_per_pulse = 0 # TODO: calculate based on Q-factor and duration
        wallplug_energy_per_pulse_cooling = energy_left_in_structures_per_pulse / efficiency_cooling
        
        # return energy usage per bunch
        return wallplug_energy_per_pulse_cooling / self.num_bunches_in_train
        
        
    def energy_usage_rf(self) -> float:
        "Energy usage per bunch [J]"
        return (self.get_structure_pulse_energy() / self.efficiency_wallplug_to_rf) * self.get_num_structures() / self.num_bunches_in_train

    
    def energy_usage(self) -> float: # TODO: improve the estimate of number of klystrons required
        "Energy usage per shot [J]"
        
        # return energy usage per bunch
        return self.energy_usage_rf() + self.energy_usage_cooling()
        
    def get_cost_structures(self):
       return self.num_structures * self.structure_length * CostModeled.cost_per_length_rf_structure_normalconducting # TODO 
        

