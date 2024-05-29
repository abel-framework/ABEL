from abel import RFAccelerator
import scipy.constants as SI
import numpy as np

class RFAcceleratorBasic(RFAccelerator):
    
    def __init__(self, length=None, nom_energy_gain=None, accel_gradient=20e6, filling_factor=0.9, cavity_frequency=1e9, cell_length=2.0, peak_power_klystrons=50e6, operating_temperature=300):
        super().__init__(length, nom_energy_gain, accel_gradient, filling_factor, cavity_frequency, cell_length, peak_power_klystrons, operating_temperature)

    
    def track(self, beam, savedepth=0, runnable=None, verbose=False):

        # add energy to all particles
        beam.accelerate(self.nom_energy_gain);
        
        return super().track(beam, savedepth, runnable, verbose)


    # energy usage per shot
    def energy_usage(self):

        # TODO: something is wrong with the energy usage
        
        # cavity R/Q per length (shape dependent only)
        norm_R_upon_Q = 1/(SI.c*SI.epsilon_0)*(self.cavity_frequency/SI.c)

        # cavity energy per length
        cavity_energy_per_length = self.accel_gradient**2/(2*np.pi*self.cavity_frequency*norm_R_upon_Q)

        # cavity filling time
        filling_time = 2 * cavity_energy_per_length * self.cell_length / self.peak_power_klystrons

        # beam loading efficiency
        peak_power_duration = filling_time + self.train_duration()
        efficiency_beamloading = self.train_duration()/peak_power_duration

        # calculate cooling efficiency (Carnot engine efficiency)
        room_temperature = 300 # [K]
        efficiency_cooling = self.operating_temperature/room_temperature
        
        # calculate overall efficiency
        efficiency_total = self.efficiency_wallplug_to_rf / (1/efficiency_beamloading +  (1-efficiency_beamloading)/efficiency_cooling)

        # wallplug power usage
        wallplug_power = peak_power_duration * self.peak_power_klystrons * self.rep_rate_trains / efficiency_total

        # return energy usage per bunch
        return wallplug_power/self.rep_rate_average()
                    