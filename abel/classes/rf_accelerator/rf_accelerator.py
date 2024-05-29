from abc import abstractmethod
from abel import Trackable

class RFAccelerator(Trackable):
    
    @abstractmethod
    def __init__(self, length, nom_energy_gain, accel_gradient=None, filling_factor=1, cavity_frequency=None, cell_length=None, peak_power_klystrons=None, operating_temperature=None, bunch_separation=None, num_bunches_in_train=1, rep_rate_trains=None):
        
        self.length = length
        self.nom_energy_gain = nom_energy_gain
        self.cavity_frequency = cavity_frequency
        self.accel_gradient = accel_gradient
        self.cell_length = cell_length
        self.peak_power_klystrons = peak_power_klystrons
        self.filling_factor = filling_factor
        self.operating_temperature = operating_temperature

        # bunch train pattern
        self.bunch_separation = bunch_separation # [s]
        self.num_bunches_in_train = num_bunches_in_train
        self.rep_rate_trains = rep_rate_trains # [Hz]

        self.efficiency_wallplug_to_rf = 0.60
        
        self.cost_per_length = 0.22e6 # [ILCU/m] not including klystrons (ILC is 0.24e6 with power)

    @abstractmethod   
    def track(self, beam, savedepth=0, runnable=None, verbose=False):
        return super().track(beam, savedepth, runnable, verbose)
    
    def get_length(self):
        if self.length is None and self.accel_gradient is not None:
            return self.nom_energy_gain/(self.accel_gradient*self.filling_factor)
        else:
            return self.length

    def get_cost(self):
        return self.get_length() * self.cost_per_length

    def rep_rate_average(self):
        return self.num_bunches_in_train * self.rep_rate_trains
        
    def rep_rate_intratrain(self):
        if self.bunch_separation is not None:
            return 1/self.bunch_separation
        else:
            return None

    def train_duration(self):
        if self.bunch_separation is not None:
            return self.bunch_separation * (self.num_bunches_in_train-1)
        else:
            return None

    @abstractmethod   
    def energy_usage(self):
        pass

    def wallplug_power(self):
        return self.energy_usage() * self.rep_rate_average()
    
    def get_nom_energy_gain(self):
        return self.nom_energy_gain
    
    def survey_object(self):
        return patches.Rectangle((0, -1), self.get_length(), 2)