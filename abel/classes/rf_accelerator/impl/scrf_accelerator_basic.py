from abel.classes.rf_accelerator.rf_accelerator import RFAccelerator
from abel.classes.cost_modeled import CostModeled
import scipy.constants as SI
import numpy as np

class SCRFAcceleratorBasic(RFAccelerator):

    #NOTE: DIFFERENT ORDER OF OF PARAMETERS!
    def __init__(self, length=None, nom_energy_gain=None, nom_accel_gradient=31.5e6, rep_rate_trains=None, bunch_separation=None, num_bunches_in_train=None, fill_factor=0.711, rf_frequency=1.3e9, structure_length=1.038, peak_power_klystron=9.822e6, operating_temperature=2):
        
        self.peak_power_klystron = peak_power_klystron
        self.operating_temperature = operating_temperature

        self.bunch_charge = None
        
        # run base class constructor
        num_structures = None
        if (length != None and structure_length != None and fill_factor != None):
            num_structures = int(np.ceil(length/structure_length)/fill_factor)
        #super().__init__(length=length, nom_energy_gain=nom_energy_gain, structure_length=structure_length, fill_factor=fill_factor, nom_accel_gradient=nom_accel_gradient, rf_frequency=rf_frequency, rep_rate_trains=rep_rate_trains, num_bunches_in_train=num_bunches_in_train, bunch_separation=bunch_separation)
        super().__init__(length=length, structure_length=structure_length, num_structures=num_structures, nom_energy_gain=nom_energy_gain)
        self.rf_frequency=rf_frequency
        self.rep_rate_trains = rep_rate_trains #To Trackable

    def track(self, beam, savedepth=0, runnable=None, verbose=False):
        b = super().track(beam, savedepth, runnable, verbose)

        # calculate the number of klystrons required
        self.energy_usage()

        return b


    def optimize_linac_geometry_and_gradient(self,fill_factor=1.0):
        """
        Find the right structure voltage based on the physics, then set the number of structures
        and the overall linac length so that the total energy gain is respected.

        See RFAccelerator_TW for example

        Returns (Vmax, Vstruct, Ntotal_int)
        """
        raise NotImplementedError()

    def get_cost_structures(self):
        "Cost of the RF structures [ILC units]"
        return self.length * CostModeled.cost_per_length_rf_structure_superconducting
        
    
    # implement required abstract methods

    def energy_usage_cooling(self) -> float:
        "Energy usage per shot for cooling [J]"
        
        # cavity filling time (approx.)
        filling_time = 925e-6 # [s]
        
        # beam loading efficiency
        train_duration = self.train_duration
        peak_power_duration = filling_time + train_duration
        efficiency_beamloading = train_duration/peak_power_duration
        
        # adjust the number of klystrons per structure
        self.num_structures_per_klystron = 39

        self.num_structures_per_cryounit = 189

        # heat loads for cooling
        static_heat_load_per_structure = 1.32/9 # [W/structure]
        dynamic_heat_load_per_structure = 9.79/9 # [W/structure] TODO: normalize by the current
        heat_load_per_structure = static_heat_load_per_structure + dynamic_heat_load_per_structure
        heat_load_total = heat_load_per_structure*self.num_structures
        
        # calculate cooling efficiency (Carnot engine efficiency)
        room_temperature = 309 # [K]
        efficiency_carnot = self.operating_temperature/room_temperature
        carnot_fraction = 0.22
        efficiency_cooling = efficiency_carnot * carnot_fraction
        
        # average power usage for cooling
        total_cooling_power = heat_load_total / efficiency_cooling
        wallplug_efficiency_cooling = 0.5089
        wallplug_power_cooling = total_cooling_power / wallplug_efficiency_cooling
        energy_per_bunch_cooling = wallplug_power_cooling / self.get_rep_rate_average()
        
        # return energy usage per bunch
        
        return energy_per_bunch_cooling

        
    def energy_usage_klystrons(self) -> float: # TODO: improve the estimate of number of klystrons required
        "Energy usage per shot [J]"
        
        # cavity filling time (approx.)
        filling_time = 925e-6 # [s]
        
        # beam loading efficiency
        train_duration = self.train_duration
        peak_power_duration = filling_time + train_duration
        efficiency_beamloading = train_duration/peak_power_duration

        # adjust the number of klystrons per structure
        self.num_structures_per_klystron = 39

        # average power usage for klystrons/modulators
        energy_per_train_per_klystron = self.peak_power_klystron * peak_power_duration
        energy_per_bunch_per_klystron = energy_per_train_per_klystron / self.num_bunches_in_train
        efficiency_klystron_wallplug_to_rf = 0.65
        wallplug_energy_per_bunch_per_klystron = energy_per_bunch_per_klystron / efficiency_klystron_wallplug_to_rf
        energy_per_bunch_all_klystrons = wallplug_energy_per_bunch_per_klystron * self.get_num_klystrons()
        
        # return energy usage per bunch
        return energy_per_bunch_all_klystrons

        
    def energy_usage(self) -> float: # TODO: improve the estimate of number of klystrons required
        "Energy usage per shot [J]"
        
        # cavity filling time (approx.)
        filling_time = 925e-6 # [s]
        
        # beam loading efficiency
        train_duration = self.train_duration
        peak_power_duration = filling_time + train_duration
        efficiency_beamloading = train_duration/peak_power_duration

        self.num_structures_per_klystron
        
        # adjust the number of klystrons per structure
        self.num_structures_per_klystron = 39

        self.num_structures_per_cryounit = 189

        # heat loads for cooling
        static_heat_load_per_structure = 1.32/9 # [W/structure]
        dynamic_heat_load_per_structure = 9.79/9 # [W/structure] TODO: normalize by the current
        heat_load_per_structure = static_heat_load_per_structure + dynamic_heat_load_per_structure
        heat_load_total = heat_load_per_structure*self.num_structures
        
        # calculate cooling efficiency (Carnot engine efficiency)
        room_temperature = 309 # [K]
        efficiency_carnot = self.operating_temperature/room_temperature
        carnot_fraction = 0.22
        efficiency_cooling = efficiency_carnot * carnot_fraction
        
        # average power usage for cooling
        total_cooling_power = heat_load_total / efficiency_cooling
        wallplug_efficiency_cooling = 0.5089
        wallplug_power_cooling = total_cooling_power / wallplug_efficiency_cooling
        energy_per_bunch_cooling = wallplug_power_cooling / self.get_rep_rate_average()
        
        # average power usage for klystrons/modulators
        energy_per_train_per_klystron = self.peak_power_klystron * peak_power_duration
        energy_per_bunch_per_klystron = energy_per_train_per_klystron / self.num_bunches_in_train
        efficiency_klystron_wallplug_to_rf = 0.65
        wallplug_energy_per_bunch_per_klystron = energy_per_bunch_per_klystron / efficiency_klystron_wallplug_to_rf
        energy_per_bunch_all_klystrons = wallplug_energy_per_bunch_per_klystron * self.get_num_klystrons()
        
        # combined RF and cooling energy
        wallplug_energy_per_bunch = energy_per_bunch_all_klystrons + energy_per_bunch_cooling
        
        # return energy usage per bunch
        return wallplug_energy_per_bunch
        
