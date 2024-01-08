from abc import abstractmethod
from matplotlib import patches
from abel import Trackable

class Source(Trackable):
    
    @abstractmethod
    def __init__(self):
        pass
        
    def track(self, beam, savedepth=0, runnable=None, verbose=False):
        beam.location = 0
        beam.stage_number = 0
        beam.trackable_number = -1
        return super().track(beam, savedepth, runnable, verbose)
    
    @abstractmethod
    def get_length(self):
        pass
    
    @abstractmethod
    def get_energy(self):
        pass
        
    @abstractmethod
    def energy_efficiency(self):
        pass
    
    @abstractmethod
    def get_charge(self):
        pass
    
    def energy_usage(self):
        return self.get_energy()*abs(self.get_charge())/self.energy_efficiency()
    
    def survey_object(self):
        rect = patches.Rectangle((0, -0.5), self.get_length(), 1)
        return rect
    