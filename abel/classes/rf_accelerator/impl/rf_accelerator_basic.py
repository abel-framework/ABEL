from abel.classes.rf_accelerator.rf_accelerator import RFAccelerator
from abel.classes.cost_modeled import CostModeled
import scipy.constants as SI
import numpy as np

class RFAcceleratorBasic(RFAccelerator):

    def __init__(self, length=None, nom_energy_gain=None, nom_accel_gradient=20e6, fill_factor=0.71, rf_frequency=2e9, structure_length=0.5, peak_power_klystron=50e6, operating_temperature=300):
        
        self.peak_power_klystron = peak_power_klystron
        self.operating_temperature = operating_temperature
        
        # run base class constructor
        super().__init__(length=length, nom_energy_gain=nom_energy_gain, structure_length=structure_length)
        
        self.rf_frequency = rf_frequency
        #IDK what to do whith these.
        #self.nom_accel_gradient=nom_accel_gradient
        #self.fill_factor = fill_factor

    def track(self, beam, savedepth=0, runnable=None, verbose=False):
        b = super().track(beam, savedepth, runnable, verbose)

        # calculate the number of klystrons required
        self.energy_usage()

        return b


    def get_cost_structures(self):
        "Cost of the RF structures [ILC units]"
        return self.length * CostModeled.cost_per_length_rf_structure_normalconducting

    def optimize_linac_geometry_and_gradient(self,fill_factor=1.0):
        """
        Find the right structure voltage based on the physics, then set the number of structures
        and the overall linac length so that the total energy gain is respected.

        See RFAccelerator_TW for example

        Returns (Vmax, Vstruct, Ntotal_int)
        """
        raise NotImplementedError()
    
    # implement required abstract methods
    
    def energy_usage_cooling(self) -> float:
        "Energy usage per shot for cooling [J]"
        
        # cavity R/Q per length (shape dependent only)
        norm_R_upon_Q = 1/(SI.c*SI.epsilon_0)*(self.rf_frequency/SI.c)
        
        # energy per length in structures
        structure_energy_per_length = self.gradient_structure**2/(2*np.pi*self.rf_frequency*norm_R_upon_Q)

        # peak power required in the structure # [W]
        peak_power_structure_min = 10e6
        peak_power_structure = max(peak_power_structure_min, abs(self.bunch_charge) * self.voltage_structure / self.bunch_separation)
        
        # cavity filling time (approx.)
        filling_time = 2 * structure_energy_per_length * self.structure_length / peak_power_structure
        
        # beam loading efficiency
        train_duration = self.train_duration
        peak_power_duration = filling_time + train_duration
        
        # adjust the number of klystrons per structure
        self.num_structures_per_klystron = self.peak_power_klystron / peak_power_structure
        
        # calculate cooling efficiency (Carnot engine efficiency)
        room_temperature = 300 # [K]
        efficiency_cooling = self.operating_temperature/room_temperature
        
        energy_left_in_structures_per_pulse = 0 # TODO: calculate based on Q-factor and duration
        wallplug_energy_per_pulse_cooling = energy_left_in_structures_per_pulse / efficiency_cooling
        
        # return energy usage per bunch
        return wallplug_energy_per_pulse_cooling / self.num_bunches_in_train
        
        
    def energy_usage_klystrons(self) -> float: # TODO: improve the estimate of number of klystrons required
        "Energy usage per shot [J]"
        
        # cavity R/Q per length (shape dependent only)
        norm_R_upon_Q = 1/(SI.c*SI.epsilon_0)*(self.rf_frequency/SI.c)
        
        # energy per length in structures
        structure_energy_per_length = self.gradient_structure**2/(2*np.pi*self.rf_frequency*norm_R_upon_Q)

        # peak power required in the structure # [W]
        peak_power_structure_min = 10e6
        peak_power_structure = max(peak_power_structure_min, abs(self.bunch_charge) * self.voltage_structure / self.bunch_separation)
        
        # cavity filling time (approx.)
        filling_time = 2 * structure_energy_per_length * self.structure_length / peak_power_structure
        
        # beam loading efficiency
        train_duration = self.train_duration
        peak_power_duration = filling_time + train_duration
        efficiency_beamloading = train_duration/peak_power_duration
        
        # adjust the number of klystrons per structure
        self.num_structures_per_klystron = self.peak_power_klystron / peak_power_structure
        
        # wallplug power usage
        energy_per_pulse_rf = peak_power_duration * self.peak_power_klystron * self.get_num_klystrons()
        wallplug_energy_per_pulse_rf = energy_per_pulse_rf / self.efficiency_wallplug_to_rf
        
        return wallplug_energy_per_pulse_rf / self.num_bunches_in_train

    
    def energy_usage(self) -> float: # TODO: improve the estimate of number of klystrons required
        "Energy usage per shot [J]"
        
        # return energy usage per bunch
        return self.energy_usage_klystrons() + self.energy_usage_cooling()
        
