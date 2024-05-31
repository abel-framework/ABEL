from abel import RFAccelerator
import scipy.constants as SI
import numpy as np

class RFAcceleratorBasic(RFAccelerator):

    def __init__(self, length=None, nom_energy_gain=None, accel_gradient=20e6, filling_factor=0.9, cavity_frequency=1e9, cell_length=2.0, peak_power_klystrons=50e6, operating_temperature=300):

        self.cavity_frequency = cavity_frequency
        self.cell_length = cell_length
        self.peak_power_klystron = peak_power_klystrons
        self.operating_temperature = operating_temperature

        if length == None:
            raise ValueError("length must be set")

        #Figure out the approximate number of structures, assuming single-cell structures
        # (A cell in RF structures = one "bubble" of a longer structure). Not the same as a "plasma cell"!
        num_structures = int(round(filling_factor*length/cell_length))

        #Baseclass expects gradient OR total voltage but not both. It will now recalculate the gradient.
        super().__init__(length=length, num_structures=num_structures, gradient=accel_gradient, voltage_total=None,  bunch_separation=None, num_bunches_in_train=1, rep_rate_trains=None)

        #Verify that we didn't mess up the gradient by the round
        if nom_energy_gain != None:
            if (self.get_voltage_total()-nom_energy_gain)/nom_energy_gain > 0.05:
                raise ValueError(f"Difference between computed total voltage * e and demanded nominal energy gain > 5%; V={self.getVoltageTotal()/1e6} [MV], nom_energy_gain={nom_energy_gain/1e6} [MeV]")

    def track(self, beam, savedepth=0, runnable=None, verbose=False):
        # add energy to all particles
        beam.accelerate(self.self.get_voltage_total())
        return super().track(beam, savedepth, runnable, verbose)

    #Implement required abstract methods

    def get_RF_structure_length(self) -> float:
        "Gets the length of each individual RF structure [m]"
        return self.cell_length

    def get_RF_frequency(self) -> float:
        "Get the RF frequency of the RF structures [1/s]"
        return self.cavity_frequency

    def energy_usage(self):
        "Energy usage per shot"

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
